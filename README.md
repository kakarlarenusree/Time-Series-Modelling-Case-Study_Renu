# Forecasting Weekly & Hourly German Electricity Demand

Time series forecasting of German electricity demand using the [Open Power System Data](https://open-power-system-data.org/) (OPSD) hourly load series, with Berlin temperature (Open-Meteo) as an exogenous weather regressor. Built for Advanced Research Topics — Assignment 1.

Six forecasting approaches are compared on a common 104-week (2-year) held-out test period: Mean, Naive, Seasonal Naive, and Drift benchmarks, SARIMA, SARIMAX (temperature + holiday regressors), a Gradient Boosting feature model, and an hourly LSTM.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/125aoZ03-P6ZcTBXtX10AiKe8aAe3Iw66)

Developed and run in Google Colab. No local Python environment is required — click the badge above to open and run the notebook directly in your browser.

## Repository structure

```
.
├── notebooks/
│   └── A1_electricity_demand_forecasting.ipynb   # main analysis notebook (run top to bottom)
├── report/
│   └── report.pdf                                # written report (6-8 pages)
├── figures/                                       # exported figures used in the report
├── requirements.txt
├── README.md
└── LICENSE
```

## Data

| Source | Description | Access |
|---|---|---|
| [OPSD Time Series](https://data.open-power-system-data.org/time_series/) | Hourly German electricity load, `2015-01-01` to `2020-10-06` | Downloaded automatically in Section 1.1 of the notebook |
| [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive) | Daily mean temperature for Berlin (52.52°N, 13.41°E) | Downloaded automatically in Section 4.1 |
| [`holidays`](https://pypi.org/project/holidays/) (Python package) | German public holiday calendar | Generated in Section 4.3 |

No manual downloads are required — both external datasets are fetched over HTTP when the notebook is run. An internet connection is needed the first time; results are cached in-memory for the rest of the run.

## Models

| Model | Section | Notes |
|---|---|---|
| Mean / Naive / Seasonal Naive / Drift | Part 2 | Classical forecasting benchmarks |
| SARIMA | Part 3 | Two-stage AIC grid search over `(p,d,q)(P,D,Q)_52` |
| SARIMAX | Part 4 | SARIMA + weekly temperature and holiday exogenous regressors |
| Gradient Boosting (`HistGradientBoostingRegressor`) | Part 5 | Lag/rolling/Fourier/calendar features, evaluated with permutation importance |
| LSTM (seq2seq) | Part 6 | 168h lookback → 24h forecast horizon, rolled forward across the hourly test period |

## Getting started

### Option A — Google Colab (recommended, no setup needed)

1. Click the **Open In Colab** badge above (or open `notebooks/A1_electricity_demand_forecasting.ipynb` from this repo directly at [colab.research.google.com](https://colab.research.google.com) via `File > Open notebook > GitHub`).
2. Run all cells in order (`Runtime > Run all`). The notebook installs any packages not already present in the Colab environment (e.g. `holidays`, `tensorflow-cpu`) via `pip` at the point they're first needed, so no separate install step is required.
3. Colab's free tier is sufficient for the whole notebook, though the SARIMA grid search (Part 3.1) and LSTM hyperparameter search (Part 6.3) are the slowest steps — expect roughly 15-30 minutes end to end depending on the allocated runtime.

### Option B — Run locally instead

```bash
git clone https://github.com/<kakarlarenusree>/electricity-demand-forecasting.git
cd electricity-demand-forecasting
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/A1_electricity_demand_forecasting.ipynb
```

Then run all cells top to bottom (`Kernel > Restart & Run All`).

Random seeds are fixed (`RANDOM_STATE = 0`) for the Gradient Boosting model and the LSTM, so results should be reproducible between runs on the same environment (Colab or local).

## Results

Test-set evaluation metrics (2-year weekly test horizon; the hourly LSTM is aggregated to weekly for comparison) are computed in the notebook's final comparison section and reported in full in `report/report.pdf`. A plot of all model forecasts against actual demand over the test period is generated at the end of the notebook and exported to `figures/`.

## Requirements.txt

```
numpy
pandas
matplotlib
requests
statsmodels
scikit-learn
holidays
tensorflow-cpu
joblib
jupyter
```

## License

This project is submitted as coursework for [module/course name]. Code is shared under the MIT License (see `LICENSE`) unless otherwise required by your institution's policy — check before reusing.

## Author

Renusree — 24087145— MSc Data Science

## Acknowledgements / Data attribution

- Electricity demand data: Open Power System Data, https://open-power-system-data.org/
- Weather data: Open-Meteo, https://open-meteo.com/ (CC BY 4.0)
- Public holiday data: `holidays` Python package
