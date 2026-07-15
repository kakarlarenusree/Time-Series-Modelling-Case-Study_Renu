
# ============================================================
# Global setup
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import itertools
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

TEST_WEEKS = 104   # two-year forecast horizon for weekly models
TEST_DAYS_HOURLY = 2 * 365  # ~2 years for the hourly LSTM (Part 6)

# ------------------------------------------------------------
# 1.1 Download hourly OPSD data for Germany
# ------------------------------------------------------------
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

raw = pd.read_csv(
    OPSD_URL,
    usecols=["utc_timestamp", "DE_load_actual_entsoe_transparency"],
    parse_dates=["utc_timestamp"],
)

raw = raw.rename(columns={
    "utc_timestamp": "date",
    "DE_load_actual_entsoe_transparency": "load_mw",
})

raw = raw.set_index("date").sort_index()

hourly = raw["load_mw"].astype(float)
hourly = hourly[hourly.notna()]

# Keep only 2015-01-01 through the end of the file (Oct 2020), as instructed
hourly = hourly.loc["2015-01-01":]

print(f"Hourly series: {hourly.index.min()}  ->  {hourly.index.max()}")
print(f"Observations : {len(hourly):,}")
hourly.head()
