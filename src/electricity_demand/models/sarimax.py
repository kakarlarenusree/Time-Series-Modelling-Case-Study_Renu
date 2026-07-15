import time
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ------------------------------------------------------------
# 4.1 Fetch daily Berlin temperature from Open-Meteo
# ------------------------------------------------------------
def get_open_meteo_temperature(latitude=52.52, longitude=13.41, start_date="2015-01-01", end_date="2020-12-31"):
    """Download daily mean temperature (deg C) for a location from the Open-Meteo archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["daily"]
    temp = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temperature_2m_mean": data["temperature_2m_mean"],
    }).set_index("date")
    # The API 'time' strings are YYYY-MM-DD (naive). We want them as YYYY-MM-DD 00:00:00+00:00 UTC
    # to align with weekly.index, assuming the API values are for the corresponding *UTC* day.
    # The 'Europe/Berlin' timezone in params specifies what the temperature *values* refer to.
    temp.index = pd.to_datetime(data["time"]).tz_localize('UTC')
    return temp


temp_daily = get_open_meteo_temperature(
    # Fetch data starting 2 weeks earlier to ensure full weekly bins are available
    start_date=str((weekly.index.min() - pd.Timedelta(weeks=2)).date()),
    end_date=str(weekly.index.max().date()),
)
temp_daily.head()

# ------------------------------------------------------------
# 4.2 Aggregate to weekly temperature & degree-day features
#     IMPORTANT: resample('W')-mean/min/max/sum only ever look *inside* the
#     current week -> no future information leaks into week t's features.
# ------------------------------------------------------------

# Use 'W-SUN' to match the weekly.index frequency
temp_mean_resampled = temp_daily["temperature_2m_mean"].resample("W-SUN").mean()
temp_min_resampled = temp_daily["temperature_2m_mean"].resample("W-SUN").min()
temp_max_resampled = temp_daily["temperature_2m_mean"].resample("W-SUN").max()

base_heat, base_cool = 15.5, 22.0
heating_degree_days_resampled = (
    np.maximum(base_heat - temp_daily["temperature_2m_mean"], 0).resample("W-SUN").sum()
)
cooling_degree_days_resampled = (
    np.maximum(temp_daily["temperature_2m_mean"] - base_cool, 0).resample("W-SUN").sum()
)

temp_weekly = pd.DataFrame({
    "temp_mean": temp_mean_resampled,
    "temp_min": temp_min_resampled,
    "temp_max": temp_max_resampled,
    "heating_degree_days": heating_degree_days_resampled,
    "cooling_degree_days": cooling_degree_days_resampled,
})

# Reindex to ensure alignment with weekly.index, then interpolate and drop NaNs
temp_weekly = temp_weekly.reindex(weekly.index)
temp_weekly = temp_weekly.interpolate("time")
temp_weekly.head()

# ------------------------------------------------------------
# 4.3 German public-holiday feature
# ------------------------------------------------------------
try:
    import holidays
except ImportError:
    import sys, subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "holidays"], check=True)
    import holidays

de_holidays = holidays.Germany(years=range(weekly.index.min().year, weekly.index.max().year + 1))
holiday_dates = pd.to_datetime(sorted(de_holidays.keys()))

daily_index = pd.date_range(weekly.index.min() - pd.Timedelta(days=6), weekly.index.max(), freq="D")
is_holiday_daily = pd.Series(daily_index.isin(holiday_dates).astype(int), index=daily_index)

holiday_weekly = pd.DataFrame(index=weekly.index)
holiday_weekly["holiday_days"] = is_holiday_daily.resample("W").sum()
holiday_weekly["has_holiday"] = (holiday_weekly["holiday_days"] > 0).astype(int)
holiday_weekly.head()

# ------------------------------------------------------------
# 4.4 Assemble the feature table and fit SARIMAX with exogenous regressors
# ------------------------------------------------------------
feature_df = pd.DataFrame({"load_gw": weekly})
feature_df = feature_df.join(temp_weekly).join(holiday_weekly)

exog_cols = ["temp_mean", "heating_degree_days", "cooling_degree_days", "holiday_days", "has_holiday"]

# Explicitly fill NaNs in exogenous columns with 0, then interpolate and drop any remaining NaNs
feature_df[exog_cols] = feature_df[exog_cols].fillna(0)
feature_df = feature_df.interpolate("time").dropna()

y_x = feature_df["load_gw"]
X_x = feature_df[exog_cols]

y_train_x, y_test_x = y_x.iloc[:-TEST_WEEKS], y_x.iloc[-TEST_WEEKS:]
X_train_x, X_test_x = X_x.iloc[:-TEST_WEEKS], X_x.iloc[-TEST_WEEKS:]

sarimax_x = SARIMAX(
    y_train_x,
    exog=X_train_x,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarimax_x_fit = sarimax_x.fit(disp=False)

sarimax_x_fc = sarimax_x_fit.get_forecast(steps=len(y_test_x), exog=X_test_x)
sarimax_x_mean = sarimax_x_fc.predicted_mean
sarimax_x_ci = sarimax_x_fc.conf_int(alpha=0.05)
sarimax_x_mean.index = y_test_x.index
sarimax_x_ci.index = y_test_x.index

forecasts["SARIMAX (temp+holiday, conditional)"] = sarimax_x_mean
results.append(evaluate_forecast("SARIMAX (temp+holiday, conditional)", y_test_x, sarimax_x_mean, y_train_x))

print(sarimax_x_fit.summary().tables[1])  # coefficient table incl. exogenous regressors

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_train_x.index[-160:], y_train_x.iloc[-160:], label="Training data", color="black", linewidth=1.2)
ax.plot(y_test_x.index, y_test_x, label="Actual (test)", color="black", linewidth=2)
ax.plot(y_test_x.index, sarima_mean.reindex(y_test_x.index), label="SARIMA (no exog)", color="tab:purple", linestyle="--")
ax.plot(y_test_x.index, sarimax_x_mean, label="SARIMAX (+temp/holiday)", color="tab:brown")
ax.fill_between(y_test_x.index, sarimax_x_ci.iloc[:, 0], sarimax_x_ci.iloc[:, 1],
                alpha=0.15, color="tab:brown", label="SARIMAX 95% interval")
ax.axvline(y_train_x.index[-1], color="grey", linestyle=":")
ax.set_title("SARIMAX with temperature + holiday exogenous regressors (conditional forecast)")
ax.set_ylabel("Load, GW")
ax.legend(ncol=2)
plt.tight_layout()
plt.show()

print(f"SARIMAX (+exog) test RMSE = {rmse(y_test_x, sarimax_x_mean):.3f} GW   "
      f"vs SARIMA (no exog) RMSE = {rmse(test, sarima_mean):.3f} GW")
