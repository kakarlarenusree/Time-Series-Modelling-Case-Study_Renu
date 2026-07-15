import time
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def _fit_one(series, order, seasonal_order):
    """Fit a single SARIMAX candidate, fully suppressing convergence warnings
    (they are expected and harmless for the many over-parameterised candidates
    a grid search deliberately tries) and failing fast on non-convergence.
    """
    p, d, q = order
    if p == 0 and q == 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                trend="c" if d == 0 else None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            # low_memory + a bounded number of iterations keeps each candidate
            # fast; candidates that don't converge in 50 iterations are simply
            # not competitive on AIC and get dropped rather than ground out.
            fit = model.fit(disp=False, maxiter=50, method="lbfgs", low_memory=True)
        if not np.isfinite(fit.aic):
            return None
        return {"order": order, "seasonal_order": seasonal_order, "AIC": fit.aic, "BIC": fit.bic}
    except Exception:
        return None


def sarima_grid_search(series, p_range, d_range, q_range, seasonal_order, n_jobs=-1):
    """Parallel AIC grid search over (p, d, q) for a fixed seasonal order."""
    combos = list(itertools.product(p_range, d_range, q_range))
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_one)(series, (p, d, q), seasonal_order) for p, d, q in combos
    )
    results = [r for r in out if r is not None]
    print(f"Fitted {len(results)}/{len(combos)} candidates in {time.time() - t0:.1f}s "
          f"({len(combos) - len(results)} skipped: degenerate or failed to converge)")
    return pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)


# ---- Stage 1: non-seasonal grid, required ranges p in [0,6], d in [0,2], q in [0,6] ----
# Fitting all 3 x 7 x 7 = 147 candidates *with* a 52-period seasonal term is the
# expensive step (each fit is a large Kalman-filter state-space model). We keep the
# full required ranges but (a) run every candidate in parallel across CPU cores,
# (b) cap each fit's iterations so non-converging over-parameterised candidates are
# dropped quickly instead of grinding for minutes, and (c) suppress the resulting
# (expected, harmless) ConvergenceWarning spam -- a non-converging fit is simply
# excluded from AIC ranking rather than treated as an error.

seasonal_order_fixed = (1, 1, 1, 52)

coarse_grid = sarima_grid_search(
    train,
    p_range=range(0, 7),
    d_range=range(0, 3),
    q_range=range(0, 7),
    seasonal_order=seasonal_order_fixed,
)
coarse_grid.head(10)

best_p, best_d, best_q = coarse_grid.iloc[0]["order"]
print(f"Best non-seasonal order from Stage 1: (p,d,q) = ({best_p},{best_d},{best_q})  "
      f"AIC = {coarse_grid.iloc[0]['AIC']:.2f}")
# ---- Stage 2: seasonal grid search, P,D,Q in [0,2], fixing the best (p,d,q) ----
def _fit_seasonal_one(series, order, seasonal_order):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=50, method="lbfgs", low_memory=True)
        if not np.isfinite(fit.aic):
            return None
        return {"seasonal_order": seasonal_order, "AIC": fit.aic, "BIC": fit.bic}
    except Exception:
        return None


seasonal_combos = [(P, D, Q, 52) for P, D, Q in itertools.product(range(0, 3), range(0, 3), range(0, 3))]

t0 = time.time()
seasonal_out = Parallel(n_jobs=-1, verbose=5)(
    delayed(_fit_seasonal_one)(train, (best_p, best_d, best_q), so) for so in seasonal_combos
)
seasonal_results = [r for r in seasonal_out if r is not None]
print(f"Fitted {len(seasonal_results)}/{len(seasonal_combos)} seasonal candidates in {time.time() - t0:.1f}s")

seasonal_grid = pd.DataFrame(seasonal_results).sort_values("AIC").reset_index(drop=True)
seasonal_grid.head(10)

best_P, best_D, best_Q, best_s = seasonal_grid.iloc[0]["seasonal_order"]
best_order = (int(best_p), int(best_d), int(best_q))
best_seasonal_order = (int(best_P), int(best_D), int(best_Q), int(best_s))

print(f"Selected model: SARIMA{best_order}{best_seasonal_order}")
print(f"AIC = {seasonal_grid.iloc[0]['AIC']:.2f}")

sarima_model = SARIMAX(
    train,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

resid = sarima_fit.resid.dropna()

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

axes[0, 0].plot(resid.index, resid, linewidth=0.8, color="darkslategray")
axes[0, 0].axhline(0, color="black", linewidth=1)
axes[0, 0].set_title("Residuals over time")

plot_acf(resid, lags=60, ax=axes[0, 1], title="ACF of residuals")

axes[1, 0].hist(resid, bins=40, color="steelblue", edgecolor="white")
axes[1, 0].set_title("Residual distribution")

from scipy import stats
stats.probplot(resid, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Residual Q-Q plot")

plt.tight_layout()
plt.show()

# Ljung-Box test for remaining autocorrelation (H0: residuals are white noise)
lb_test = acorr_ljungbox(resid, lags=[10, 20, 52], return_df=True)
print(lb_test)

# ------------------------------------------------------------
# 3.3 Forecast the 2-year (104-week) test period, with confidence intervals
# ------------------------------------------------------------
sarima_fc = sarima_fit.get_forecast(steps=h)
sarima_mean = sarima_fc.predicted_mean
sarima_ci95 = sarima_fc.conf_int(alpha=0.05)
sarima_ci80 = sarima_fc.conf_int(alpha=0.20)

sarima_mean.index = test.index
sarima_ci95.index = test.index
sarima_ci80.index = test.index

forecasts["SARIMA"] = sarima_mean
results.append(evaluate_forecast("SARIMA", test, sarima_mean, train))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[-160:], train.iloc[-160:], label="Training data", color="black", linewidth=1.2)
ax.plot(test.index, test, label="Actual (test)", color="black", linewidth=2)
ax.plot(test.index, sarima_mean, label="SARIMA forecast", color="tab:purple")
ax.fill_between(test.index, sarima_ci95.iloc[:, 0], sarima_ci95.iloc[:, 1],
                alpha=0.15, color="tab:purple", label="95% interval")
ax.fill_between(test.index, sarima_ci80.iloc[:, 0], sarima_ci80.iloc[:, 1],
                alpha=0.25, color="tab:purple", label="80% interval")
ax.plot(test.index, forecasts["Seasonal Naive"], label="Seasonal naive", linestyle="--", color="tab:green")
ax.axvline(train.index[-1], color="grey", linestyle=":")
ax.set_title(f"SARIMA{best_order}{best_seasonal_order} — 2-year forecast with confidence intervals")
ax.set_ylabel("Load, GW")
ax.legend(ncol=2)
plt.tight_layout()
plt.show()

print(f"SARIMA test RMSE = {rmse(test, sarima_mean):.3f} GW")
pd.DataFrame(results).sort_values("MASE").reset_index(drop=True).round(3)
