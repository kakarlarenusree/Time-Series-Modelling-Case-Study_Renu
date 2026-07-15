
def make_ml_table(df, target="load_gw", max_lag=52):
    """Build a supervised-learning table for the weekly forecasting task.
    All load-derived features use shift(1)+rolling so no future information leaks in.
    """
    feat = df.copy()
    y_ = feat[target]

    for lag in [1, 2, 4, 8, 13, 26, 52]:
        feat[f"lag_{lag}"] = y_.shift(lag)

    feat["roll_mean_4"] = y_.shift(1).rolling(4).mean()
    feat["roll_mean_13"] = y_.shift(1).rolling(13).mean()
    feat["roll_mean_52"] = y_.shift(1).rolling(52).mean()
    feat["roll_std_13"] = y_.shift(1).rolling(13).std()

    week = feat.index.isocalendar().week.astype(int)
    feat["week"] = week
    feat["year"] = feat.index.year
    for k in range(1, 4):
        feat[f"sin_{k}"] = np.sin(2 * np.pi * k * week / 52)
        feat[f"cos_{k}"] = np.cos(2 * np.pi * k * week / 52)

    return feat.dropna()


ml_table = make_ml_table(feature_df)
ml_train = ml_table.iloc[:-TEST_WEEKS]
ml_test = ml_table.iloc[-TEST_WEEKS:]

feature_cols = [c for c in ml_table.columns if c != "load_gw"]
X_train_ml, y_train_ml = ml_train[feature_cols], ml_train["load_gw"]
X_test_ml, y_test_ml = ml_test[feature_cols], ml_test["load_gw"]

print(f"Features used ({len(feature_cols)}): {feature_cols}")

gbr = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.03,
    max_leaf_nodes=15,
    l2_regularization=0.1,
    random_state=RANDOM_STATE,
)
gbr.fit(X_train_ml, y_train_ml)

ml_forecast = pd.Series(gbr.predict(X_test_ml), index=y_test_ml.index, name="Feature model")

forecasts["Feature model (GBR)"] = ml_forecast
results.append(evaluate_forecast("Feature model (GBR)", y_test_ml, ml_forecast, y_train_ml))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_train_ml.index[-160:], y_train_ml.iloc[-160:], label="Training data", color="black", linewidth=1.2)
ax.plot(y_test_ml.index, y_test_ml, label="Actual (test)", color="black", linewidth=2)
ax.plot(y_test_ml.index, ml_forecast, label="Feature model (GBR)", color="tab:cyan")
ax.plot(y_test_ml.index, sarimax_x_mean.reindex(y_test_ml.index), label="SARIMAX (+exog)", linestyle="--", color="tab:brown")
ax.axvline(y_train_ml.index[-1], color="grey", linestyle=":")
ax.set_title("Feature-based Gradient Boosting forecast — 2-year horizon")
ax.set_ylabel("Load, GW")
ax.legend(ncol=2)
plt.tight_layout()
plt.show()

print(f"Feature model test RMSE = {rmse(y_test_ml, ml_forecast):.3f} GW")

# Permutation feature importance -- interpretability of the feature model
perm = permutation_importance(gbr, X_test_ml, y_test_ml, n_repeats=20, random_state=RANDOM_STATE)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
top = importance_df.head(12)
ax.barh(top["feature"][::-1], top["importance_mean"][::-1], xerr=top["importance_std"][::-1])
ax.set_title("Permutation feature importance — Gradient Boosting model")
ax.set_xlabel("Increase in test RMSE when feature is permuted")
plt.tight_layout()
plt.show()

importance_df.head(10)
