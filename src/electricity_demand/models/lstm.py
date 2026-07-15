
# ------------------------------------------------------------
# 6.1 Prepare hourly data with calendar + temperature features
# ------------------------------------------------------------
try:
    import tensorflow as tf
except ImportError:
    import sys, subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "tensorflow-cpu"], check=True)
    import tensorflow as tf

from sklearn.preprocessing import StandardScaler

tf.random.set_seed(RANDOM_STATE)

hourly_gw = (hourly / 1000.0).asfreq("H").interpolate("time")
hourly_gw.name = "load_gw"

# Hourly temperature via linear interpolation of the daily Open-Meteo series
# (a genuine deployment would instead pull an hourly weather forecast product)
temp_hourly = temp_daily["temperature_2m_mean"].reindex(
    pd.date_range(temp_daily.index.min(), temp_daily.index.max() + pd.Timedelta(hours=23), freq="H")
).interpolate("time")

hourly_df = pd.DataFrame({"load_gw": hourly_gw}).join(temp_hourly.rename("temp"))
hourly_df["hour"] = hourly_df.index.hour
hourly_df["dow"] = hourly_df.index.dayofweek
hourly_df["month"] = hourly_df.index.month
hourly_df["sin_hour"] = np.sin(2 * np.pi * hourly_df["hour"] / 24)
hourly_df["cos_hour"] = np.cos(2 * np.pi * hourly_df["hour"] / 24)
hourly_df["sin_dow"] = np.sin(2 * np.pi * hourly_df["dow"] / 7)
hourly_df["cos_dow"] = np.cos(2 * np.pi * hourly_df["dow"] / 7)
hourly_df["sin_month"] = np.sin(2 * np.pi * hourly_df["month"] / 12)
hourly_df["cos_month"] = np.cos(2 * np.pi * hourly_df["month"] / 12)
hourly_df = hourly_df.dropna()

print(hourly_df.shape)
hourly_df.head()

# ------------------------------------------------------------
# 6.2 Train / test split -- last ~2 years held out, and scale on TRAIN ONLY
# ------------------------------------------------------------
TEST_HOURS = TEST_DAYS_HOURLY * 24

hourly_train_df = hourly_df.iloc[:-TEST_HOURS]
hourly_test_df = hourly_df.iloc[-TEST_HOURS:]

feature_cols_h = ["load_gw", "temp", "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month"]

scaler = StandardScaler().fit(hourly_train_df[feature_cols_h])
train_scaled = pd.DataFrame(scaler.transform(hourly_train_df[feature_cols_h]),
                             index=hourly_train_df.index, columns=feature_cols_h)
test_scaled = pd.DataFrame(scaler.transform(hourly_test_df[feature_cols_h]),
                            index=hourly_test_df.index, columns=feature_cols_h)

LOOKBACK, HORIZON = 168, 24  # 7 days in, 1 day out

def make_seq2seq_windows(df_scaled, lookback=LOOKBACK, horizon=HORIZON, target_col="load_gw"):
    values = df_scaled.values
    target_idx = df_scaled.columns.get_loc(target_col)
    X, Y = [], []
    for start in range(0, len(values) - lookback - horizon + 1):
        X.append(values[start: start + lookback])
        Y.append(values[start + lookback: start + lookback + horizon, target_idx])
    return np.array(X), np.array(Y)

X_train_seq, Y_train_seq = make_seq2seq_windows(train_scaled)
print("Training windows:", X_train_seq.shape, Y_train_seq.shape)

# ------------------------------------------------------------
# 6.3 Build & hyper-tune the LSTM (simple manual search over a small grid --
#     a full-scale hyperband/Bayesian search is impractical for a class notebook,
#     so we tune the highest-leverage hyperparameters: hidden units, layers, dropout).
#     Trained on day-stride windows (see 6.2) with a short epoch budget purely
#     to RANK the 3 configs -- the winning config is then given a proper,
#     longer final fit below, so this stage doesn't need to fully converge.
# ------------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

n_features = X_train_seq.shape[2]

# Hold out the final 10% of training windows as a validation set (chronological, no leakage)
n_val = int(0.1 * len(X_train_seq))
X_tr, Y_tr = X_train_seq[:-n_val], Y_train_seq[:-n_val]
X_val, Y_val = X_train_seq[-n_val:], Y_train_seq[-n_val:]

def build_lstm(hidden_units=64, n_layers=1, dropout=0.2):
    model = Sequential()
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        if i == 0:
            model.add(LSTM(hidden_units, return_sequences=return_seq, input_shape=(LOOKBACK, n_features)))
        else:
            model.add(LSTM(hidden_units, return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(RepeatVector(HORIZON))
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer="adam", loss="mse")
    return model

import time

hp_grid = [
    {"hidden_units": 32, "n_layers": 1, "dropout": 0.1},
    {"hidden_units": 64, "n_layers": 1, "dropout": 0.2},
    {"hidden_units": 64, "n_layers": 2, "dropout": 0.2},
]

# epochs=10 + patience=2 + batch_size=256 is deliberately fast: this stage only
# needs to *rank* the 3 configs relative to each other, not fully train the
# winning one -- the winning config gets a proper, longer final fit below.
hp_results = []
for hp in hp_grid:
    t0 = time.time()
    model = build_lstm(**hp)
    es = EarlyStopping(patience=2, restore_best_weights=True)
    history = model.fit(
        X_tr, Y_tr[..., np.newaxis],
        validation_data=(X_val, Y_val[..., np.newaxis]),
        epochs=10, batch_size=256, verbose=0, callbacks=[es],
    )
    val_loss = min(history.history["val_loss"])
    hp_results.append({**hp, "val_mse_scaled": val_loss})
    print(hp, f"-> val MSE (scaled): {val_loss:.5f}  ({time.time() - t0:.1f}s)")

hp_results_df = pd.DataFrame(hp_results).sort_values("val_mse_scaled").reset_index(drop=True)
hp_results_df

best_hp = hp_results_df.iloc[0][["hidden_units", "n_layers", "dropout"]].to_dict()
best_hp["hidden_units"] = int(best_hp["hidden_units"])
best_hp["n_layers"] = int(best_hp["n_layers"])
print("Selected hyperparameters:", best_hp)

final_lstm = build_lstm(**best_hp)
es = EarlyStopping(patience=5, restore_best_weights=True)
final_history = final_lstm.fit(
    X_tr, Y_tr[..., np.newaxis],
    validation_data=(X_val, Y_val[..., np.newaxis]),
    epochs=60, batch_size=128, verbose=0, callbacks=[es],
)

plt.plot(final_history.history["loss"], label="train loss")
plt.plot(final_history.history["val_loss"], label="val loss")
plt.title("LSTM training curve (scaled MSE)")
plt.xlabel("Epoch"); plt.ylabel("MSE (scaled)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6.4 Roll the 24h-ahead model forward across the full 2-year test period
#     Each block uses only genuinely known-at-origin inputs:
#     - past load (own history, real observed up to the block start)
#     - calendar features (known in advance, by definition)
#     - temperature (again *observed*, so this is a conditional forecast --
#       see the same caveat as Parts 4-5)
# ------------------------------------------------------------
full_scaled = pd.concat([train_scaled, test_scaled])
test_start_pos = len(train_scaled)

n_blocks = TEST_HOURS // HORIZON
target_idx = feature_cols_h.index("load_gw")

lstm_preds_scaled = []
for b in range(n_blocks):
    origin = test_start_pos + b * HORIZON
    window = full_scaled.values[origin - LOOKBACK: origin]
    pred_block = final_lstm.predict(window[np.newaxis, ...], verbose=0)[0, :, 0]
    lstm_preds_scaled.append(pred_block)

lstm_preds_scaled = np.concatenate(lstm_preds_scaled)

# Inverse-transform just the load column
dummy = np.zeros((len(lstm_preds_scaled), len(feature_cols_h)))
dummy[:, target_idx] = lstm_preds_scaled
lstm_preds_gw = scaler.inverse_transform(dummy)[:, target_idx]

lstm_forecast_hourly = pd.Series(
    lstm_preds_gw, index=hourly_test_df.index[:len(lstm_preds_gw)], name="LSTM"
)

# ------------------------------------------------------------
# 6.5 Evaluate the hourly LSTM, and aggregate to weekly for a like-for-like comparison
# ------------------------------------------------------------
actual_hourly_test = hourly_test_df["load_gw"].iloc[:len(lstm_forecast_hourly)]

lstm_mae = mean_absolute_error(actual_hourly_test, lstm_forecast_hourly)
lstm_rmse = rmse(actual_hourly_test, lstm_forecast_hourly)
lstm_mape = np.mean(np.abs((actual_hourly_test - lstm_forecast_hourly) / actual_hourly_test)) * 100

print(f"Hourly LSTM  -> MAE={lstm_mae:.3f} GW  RMSE={lstm_rmse:.3f} GW  MAPE={lstm_mape:.2f}%")

fig, ax = plt.subplots(figsize=(12, 5))
window_show = slice(0, 24 * 14)  # first 2 weeks of the test period
ax.plot(actual_hourly_test.index[window_show], actual_hourly_test.iloc[window_show], label="Actual", color="black")
ax.plot(lstm_forecast_hourly.index[window_show], lstm_forecast_hourly.iloc[window_show], label="LSTM forecast", color="tab:red")
ax.set_title("Hourly LSTM — first 2 weeks of the 2-year test period")
ax.set_ylabel("Load, GW")
ax.legend()
plt.tight_layout()
plt.show()

# Aggregate the hourly LSTM forecast to weekly so it appears in the same
# comparison table/plot as the other (weekly) models
lstm_weekly = lstm_forecast_hourly.resample("W").mean()
lstm_weekly = lstm_weekly.reindex(test.index).interpolate("time")

forecasts["LSTM (hourly, aggregated to weekly)"] = lstm_weekly
results.append(evaluate_forecast("LSTM (hourly, aggregated to weekly)", test, lstm_weekly, train))
