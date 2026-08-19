# Sales Forecasting — Predictive Analytics Project

Forecasting daily sales 90 days into the future using time-series and
regression modeling, with a full data-cleaning and evaluation pipeline.

## Overview

- **Data:** 4 years of simulated daily sales data (2022–2025), built to
  resemble a real retail dataset — includes a growth trend, weekly and
  yearly seasonality, promotional spikes, and deliberately injected
  messiness (missing values, duplicate rows, outliers) to clean.
  *(Synthetic data, generated for this project — not real company data.)*
- **Goal:** clean the raw data, explore its trend/seasonality, train two
  different forecasting approaches, evaluate them on a held-out test
  period, and forecast the next 90 days.

## Pipeline

1. **Data cleaning** — removed duplicate rows, filled missing values via
   linear interpolation, capped outliers using a rolling-median residual
   method
2. **EDA** — STL decomposition (trend/seasonal/residual), weekly and
   monthly seasonality breakdown
3. **Feature engineering** — calendar features (day-of-week, month,
   weekend flag, cyclical day-of-year encoding) for the regression model
4. **Chronological train/test split** — last 90 days held out (no
   shuffling, to avoid leaking future data into training)
5. **Two models compared:**
   - Linear Regression (with engineered calendar features)
   - SARIMA(1,1,1)(1,1,1,7) — seasonal ARIMA, weekly seasonal order
6. **Evaluation** — MAE, RMSE, MAPE, R² on the 90-day test set
7. **Final forecast** — best model retrained on full history, projected
   90 days into the future

## Results

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Linear Regression** | **42.94** | **52.29** | **5.78%** | **0.076** |
| SARIMA(1,1,1)(1,1,1,7) | 60.20 | 71.28 | 8.56% | -0.718 |

**Linear Regression won.** This is a genuinely useful result, not just a
clean win for the fancier model — SARIMA's seasonal order here only
captures a *weekly* (7-day) cycle, so it never saw the *yearly* pattern
that actually drives most of the variation in this data. The regression
model, by contrast, was handed that yearly signal directly through
engineered calendar features (`doy_sin`, `doy_cos`, month dummies).
SARIMA's negative R² means it performed worse than simply predicting the
average value every day — a reminder that a more complex, "proper"
time-series model isn't automatically better than a well-featured simple
one, and that both should always be compared before picking a production
model.

## Files

- `predictive_analytics.ipynb` — full notebook: cleaning, EDA, modeling,
  evaluation, forecasting
- `historical_sales.csv` — the raw (synthetic) input data
- `forecast_next_90_days.csv` — the final 90-day forecast output

## Tech stack

Python · pandas · NumPy · scikit-learn · statsmodels · matplotlib ·
Jupyter

## Possible extensions

- Swap in a real dataset (e.g. from Kaggle)
- Give SARIMA a yearly seasonal order (`seasonal_order=(1,1,1,365)`) or
  add exogenous regressors (promotions, holidays)
- Try Prophet or a gradient-boosted model (XGBoost/LightGBM) with lag
  features
- Wrap the forecast in a small Streamlit app for an interactive demo
