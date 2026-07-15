
# ------------------------------------------------------------
# Train / test split — 2-year (104-week) forecast horizon
# ------------------------------------------------------------
y = weekly.copy()

train = y.iloc[:-TEST_WEEKS]
test = y.iloc[-TEST_WEEKS:]
h = len(test)

print(f"Training period: {train.index.min().date()} -> {train.index.max().date()}  ({len(train)} weeks)")
print(f"Test period:     {test.index.min().date()} -> {test.index.max().date()}  ({len(test)} weeks)")

# ------------------------------------------------------------
# Evaluation utilities (used throughout the notebook)
# ------------------------------------------------------------
def rmse(y_true, y_pred):
    # sklearn >=1.4 removed the `squared` kwarg from mean_squared_error in favour of
    # a dedicated root_mean_squared_error function; computing it manually works on any version.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mase(y_true, y_pred, y_train, seasonality=52):
    """Mean Absolute Scaled Error, scaled by the in-sample seasonal-naive MAE."""
    naive_errors = np.abs(y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values)
    scale = naive_errors.mean()
    return np.mean(np.abs(y_true - y_pred)) / scale

def evaluate_forecast(name, y_true, y_pred, y_train, seasonality=52):
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).reindex(y_true.index).astype(float)
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "MASE": mase(y_true, y_pred, y_train, seasonality),
        "Bias": np.mean(y_pred - y_true),
    }

results = []
forecasts = {}

# ------------------------------------------------------------
# 2.1 Mean forecast
# ------------------------------------------------------------
forecasts["Mean"] = pd.Series(train.mean(), index=test.index)

# ------------------------------------------------------------
# 2.2 Naive forecast (last observation carried forward)
# ------------------------------------------------------------
forecasts["Naive"] = pd.Series(train.iloc[-1], index=test.index)

# ------------------------------------------------------------
# 2.3 Seasonal naive forecast (value from the same week 52 weeks earlier)
#     True multi-step seasonal-naive: each forecast reuses the last *known*
#     value from one year back, without peeking at the test set.
# ------------------------------------------------------------
seasonal_naive_values = []
for i, date in enumerate(test.index):
    # source index is 52 weeks before the *forecast* date -> falls in train for h<=52,
    # and reuses the (already forecast) seasonal-naive value for h>52 (proper h-step-ahead)
    lookback_date = date - pd.DateOffset(weeks=52)
    if lookback_date in train.index:
        seasonal_naive_values.append(train.loc[lookback_date])
    else:
        # for the second forecast year, reuse the value produced 52 steps earlier
        # in this very forecast (no test-set leakage)
        seasonal_naive_values.append(seasonal_naive_values[i - 52])

forecasts["Seasonal Naive"] = pd.Series(seasonal_naive_values, index=test.index)

# ------------------------------------------------------------
# 2.4 Drift forecast
# ------------------------------------------------------------
drift_slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
forecasts["Drift"] = pd.Series(
    train.iloc[-1] + drift_slope * np.arange(1, h + 1),
    index=test.index,
)

for name, pred in forecasts.items():
    results.append(evaluate_forecast(name, test, pred, train))

pd.DataFrame(results).sort_values("MASE").reset_index(drop=True).round(3)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-160:], train.iloc[-160:], label="Training data", color="black", linewidth=1.2)
ax.plot(test.index, test, label="Actual (test)", color="black", linewidth=2, linestyle="-")

colors = {"Mean": "tab:blue", "Naive": "tab:orange", "Seasonal Naive": "tab:green", "Drift": "tab:red"}
for name, pred in forecasts.items():
    ax.plot(test.index, pred, label=name, color=colors[name], linestyle="--", linewidth=1.3)

ax.axvline(train.index[-1], color="grey", linestyle=":")
ax.set_title("Benchmark forecasts vs. actual — 2-year (104-week) horizon")
ax.set_ylabel("Load, GW")
ax.legend(ncol=3)
plt.tight_layout()
plt.show()
