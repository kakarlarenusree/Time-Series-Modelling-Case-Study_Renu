
# ------------------------------------------------------------
# 1.2 Resample to daily and weekly means (in GW)
# ------------------------------------------------------------
daily = hourly.resample("D").mean() / 1000.0
daily.name = "load_gw"

weekly = hourly.resample("W").mean() / 1000.0
weekly = weekly.asfreq("W")
weekly = weekly.interpolate("time")   # fill the odd partial/missing week at resample boundaries
weekly.name = "load_gw"

print("Daily :", daily.index.min().date(), "->", daily.index.max().date(), f"({len(daily)} obs)")
print("Weekly:", weekly.index.min().date(), "->", weekly.index.max().date(), f"({len(weekly)} obs)")

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)

axes[0].plot(hourly.index, hourly / 1000.0, linewidth=0.3, color="steelblue")
axes[0].set_title("Hourly German electricity load (GW)")
axes[0].set_ylabel("GW")

axes[1].plot(daily.index, daily, linewidth=0.6, color="darkorange")
axes[1].set_title("Daily mean load (GW)")
axes[1].set_ylabel("GW")

axes[2].plot(weekly.index, weekly, linewidth=1.2, color="firebrick")
axes[2].set_title("Weekly mean load (GW)")
axes[2].set_ylabel("GW")
axes[2].set_xlabel("Date")

plt.tight_layout()
plt.show()
