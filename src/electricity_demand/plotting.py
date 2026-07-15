
# Zoom on 2 representative years to inspect the annual cycle & weekday pattern more closely
fig, ax = plt.subplots(figsize=(12, 4))
weekly.loc["2017":"2018"].plot(ax=ax, color="firebrick")
ax.set_title("Weekly load, 2017-2018 (illustrating the annual seasonal cycle)")
ax.set_ylabel("GW")
plt.tight_layout()
plt.show()

# Boxplot by ISO week to visualise the seasonal profile & its variance
week_of_year = weekly.index.isocalendar().week.astype(int)
box_df = pd.DataFrame({"load_gw": weekly.values, "week": week_of_year.values})
fig, ax = plt.subplots(figsize=(13, 4))
box_df.boxplot(column="load_gw", by="week", ax=ax, showfliers=False)
ax.set_title("Distribution of weekly load by ISO week-of-year")
ax.set_xlabel("ISO week")
ax.set_ylabel("GW")
plt.suptitle("")
plt.tight_layout()
plt.show()

stl = STL(weekly, period=52, robust=True)
stl_result = stl.fit()

fig = stl_result.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# Quantify the relative strength of trend and seasonality (Hyndman & Athanasopoulos formulas)
resid = stl_result.resid
trend = stl_result.trend
season = stl_result.seasonal

F_trend = max(0, 1 - resid.var() / (trend + resid).var())
F_season = max(0, 1 - resid.var() / (season + resid).var())

print(f"Strength of trend    : {F_trend:.3f}")
print(f"Strength of seasonality: {F_season:.3f}")
