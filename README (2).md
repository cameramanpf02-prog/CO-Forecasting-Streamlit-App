# 🌿 CO₂ Emission Forecasting — Thailand 3 Sectors
**Academic Conference Presentation App**

## Overview
A Streamlit web application for forecasting CO₂ emissions across **Power, Transport, and Industry** sectors in Thailand using 7 time series models.

| Parameter | Value |
|-----------|-------|
| Train period | 2010–2023 (168 months) |
| Test period | 2024–2025 (24 months) |
| Forecast | Configurable 6–36 months beyond Dec 2025 |
| COVID dummy | Apr 2020 – Jun 2021 |
| Data source | EPPO Thailand (monthly) |

## Models
| Model | Type |
|-------|------|
| ARIMA | Baseline non-seasonal |
| SARIMAX+COVID | Seasonal + exogenous COVID dummy |
| ETS | Holt-Winters Exponential Smoothing |
| Prophet | Meta's decomposition model |
| Hybrid 1 | SARIMAX + Ridge Regression |
| Hybrid 2 | ETS + Ridge Regression |
| Hybrid 3 | SARIMAX + Prophet Ensemble |

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then upload `Eppo_Out_CO2_from_Power__Transport__Industry_Dataset.xlsx` via the sidebar.

## App Tabs
1. **Overview** — EDA: time series, fuel mix, statistics
2. **Forecast — All Sectors** — All 3 sectors on one page with selectable models
3. **Performance Metrics** — MAE, RMSE, MAPE tables + radar charts
4. **P-value Analysis** — SARIMAX coefficient significance
5. **Data Table** — Raw + forecast data export (CSV)
