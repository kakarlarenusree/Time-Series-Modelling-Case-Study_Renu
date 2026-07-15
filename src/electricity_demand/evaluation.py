def stationarity_report(series, label):
    series = series.dropna()

    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")

    # KPSS's lookup table only covers p in [0.01, 0.10]; when the statistic falls
    # outside that range statsmodels emits an InterpolationWarning and clips the
    # p-value to the nearest bound. We catch that warning and report the bound
    # honestly (">0.10" / "<0.01") instead of printing a misleading exact number.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
        clipped = any("InterpolationWarning" in str(w.category) for w in caught)

    if clipped:
        kpss_p_display = f">{kpss_p:.2f}" if kpss_p >= 0.10 else f"<{kpss_p:.2f}"
    else:
        kpss_p_display = f"{kpss_p:.4f}"

    print(f"--- {label} ---")
    print(f"ADF  statistic = {adf_stat:8.3f}   p-value = {adf_p:.4f}  "
          f"-> {'stationary' if adf_p < 0.05 else 'non-stationary'} (reject H0 if p<0.05)")
    print(f"KPSS statistic = {kpss_stat:8.3f}   p-value {kpss_p_display}  "
          f"-> {'non-stationary' if (not clipped and kpss_p < 0.05) else 'stationary'} "
          f"(reject H0 if p<0.05; capped p-values mean \'clearly on the stationary side\' here)")
    print()

level = weekly.copy()
diff1 = weekly.diff(1).dropna()
diff1_seasonal = weekly.diff(1).diff(52).dropna()

stationarity_report(level, "Level (no differencing)")
stationarity_report(diff1, "First difference (d=1)")
stationarity_report(diff1_seasonal, "First diff + seasonal diff (d=1, D=1, s=52)")

fig, axes = plt.subplots(3, 2, figsize=(13, 10))

for row, (series, label) in enumerate(
    [(level, "Level"), (diff1, "First difference (d=1)"), (diff1_seasonal, "d=1, D=1 (s=52)")]
):
    plot_acf(series, lags=60, ax=axes[row, 0], title=f"ACF — {label}")
    plot_pacf(series, lags=60, ax=axes[row, 1], method="ywm", title=f"PACF — {label}")

plt.tight_layout()
plt.show()
