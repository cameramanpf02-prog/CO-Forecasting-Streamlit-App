"""
CO₂ Emission Forecasting Dashboard
3 Sectors: Power Generation | Transport | Industry
Academic Conference Edition
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Emission Forecasting | 3 Sectors",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Academic Conference Style
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e8edf5;
}

/* ── Header Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d2137 0%, #102a45 40%, #0e2038 100%);
    border: 1px solid rgba(56, 139, 220, 0.35);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,139,220,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e8f4ff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #7db3e0;
    margin: 0 0 14px 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(56,139,220,0.15);
    border: 1px solid rgba(56,139,220,0.4);
    color: #7db3e0;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.3px;
}
.badge-green {
    background: rgba(34,197,94,0.12);
    border-color: rgba(34,197,94,0.35);
    color: #6ee7a6;
}

/* ── Sector Cards ── */
.sector-header {
    border-radius: 10px;
    padding: 12px 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.sector-power   { background: linear-gradient(90deg,rgba(46,134,193,0.25),rgba(46,134,193,0.05)); border-left: 4px solid #2E86C1; color: #7ac8f5; }
.sector-transport { background: linear-gradient(90deg,rgba(230,126,34,0.25),rgba(230,126,34,0.05)); border-left: 4px solid #E67E22; color: #f5b97a; }
.sector-industry  { background: linear-gradient(90deg,rgba(30,132,73,0.25),rgba(30,132,73,0.05)); border-left: 4px solid #1E8449; color: #7ae8a4; }

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(56,139,220,0.4); }
.metric-label {
    font-size: 0.72rem;
    color: #7db3e0;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8f4ff;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-unit {
    font-size: 0.7rem;
    color: #5d8aad;
    margin-top: 2px;
}
.metric-good  { color: #6ee7a6; }
.metric-warn  { color: #fbbf24; }
.metric-bad   { color: #f87171; }

/* ── Section Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(56,139,220,0.15);
    margin: 28px 0;
}

/* ── Info Box ── */
.info-box {
    background: rgba(56,139,220,0.08);
    border: 1px solid rgba(56,139,220,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #a8caec;
    margin-bottom: 16px;
}

/* ── Table ── */
.stDataFrame { background: transparent !important; }
.stDataFrame td, .stDataFrame th {
    background: rgba(13,27,42,0.8) !important;
    color: #e8edf5 !important;
    border-color: rgba(56,139,220,0.15) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #090d18 !important;
    border-right: 1px solid rgba(56,139,220,0.15) !important;
}
section[data-testid="stSidebar"] * { color: #c5d8ee !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #7db3e0 !important;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Spinner ── */
.stSpinner > div { border-color: #2E86C1 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #3d5c7a;
    font-size: 0.78rem;
    margin-top: 48px;
    padding: 20px 0;
    border-top: 1px solid rgba(56,139,220,0.1);
}

/* ── Best model tag ── */
.best-tag {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    color: #6ee7a6;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin-left: 6px;
    vertical-align: middle;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(56,139,220,0.12);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7db3e0 !important;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(46,134,193,0.25) !important;
    color: #e8f4ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TRAIN_START = '2010-01'
TRAIN_END   = '2023-12'
TEST_START  = '2024-01'
TEST_END    = '2025-12'
COVID_START = '2020-04'
COVID_END   = '2021-06'

MONTH_MAP = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}

SECTOR_CFG = {
    'Power'    : {'color':'#2E86C1','icon':'⚡','css':'sector-power',   'fuel_cols':['oil','coal','gas']},
    'Transport': {'color':'#E67E22','icon':'🚗','css':'sector-transport','fuel_cols':['oil','gas']},
    'Industry' : {'color':'#1E8449','icon':'🏭','css':'sector-industry', 'fuel_cols':['oil','coal','gas']},
}

MODELS_LIST = ['ARIMA','SARIMAX','ETS','Hybrid1(SARIMAX+Ridge)','Hybrid2(ETS+Ridge)']

BEST_MODELS = {
    'Power'    : 'Hybrid1(SARIMAX+Ridge)',
    'Transport': 'Hybrid2(ETS+Ridge)',
    'Industry' : 'SARIMAX',
}

MODEL_COLORS = {
    'Actual'               : '#e8edf5',
    'ARIMA'                : '#e74c3c',
    'SARIMAX'              : '#1abc9c',
    'ETS'                  : '#3498db',
    'Hybrid1(SARIMAX+Ridge)': '#f1c40f',
    'Hybrid2(ETS+Ridge)'   : '#e67e22',
}

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def make_exog(index, use_covid=True):
    if not use_covid:
        return None
    dummy = ((index >= COVID_START) & (index <= COVID_END)).astype(int)
    return pd.DataFrame({'covid': dummy}, index=index)

def clamp_ci(ci):
    c = ci.copy()
    c.iloc[:, 0] = c.iloc[:, 0].clip(lower=0)
    return c

def compute_metrics(actual, predicted):
    idx = actual.index.intersection(predicted.index)
    a, p = actual[idx].values, predicted[idx].values
    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mape = np.mean(np.abs((a - p) / np.maximum(a, 1))) * 100
    r2   = 1 - np.sum((a - p)**2) / np.sum((a - np.mean(a))**2)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}

def mape_grade(mape):
    if mape < 5:  return 'metric-good'
    if mape < 10: return 'metric-warn'
    return 'metric-bad'

def parse_eppo_sheet(uploaded_bytes, sheet_name, col_names):
    raw = pd.read_excel(uploaded_bytes, sheet_name=sheet_name, header=None)
    records, yr = [], None
    for _, row in raw.iterrows():
        cell = str(row.iloc[0]).strip()
        try:
            v = int(float(cell))
            if 1990 <= v <= 2030:
                yr = v
            continue
        except:
            pass
        if yr and cell.lower() in MONTH_MAP:
            month = MONTH_MAP[cell.lower()]
            try:
                vals = [float(row.iloc[i]) for i in range(1, len(col_names)+1)]
                rec  = {'date': pd.Timestamp(year=yr, month=month, day=1)}
                rec.update(dict(zip(col_names, vals)))
                records.append(rec)
            except:
                pass
    df = pd.DataFrame(records).sort_values('date').set_index('date')
    return df

# ─────────────────────────────────────────────
# Forecasting Engines
# ─────────────────────────────────────────────
def run_arima(train, test, fc_steps):
    import pmdarima as pm
    ar = pm.auto_arima(train, d=1, seasonal=False, max_p=3, max_q=3,
                       stepwise=True, information_criterion='bic', trace=False)
    order = ar.order
    fit  = ARIMA(train, order=order).fit()
    test_pred = fit.get_forecast(steps=len(test)).predicted_mean
    test_pred.index = test.index

    full_ts = pd.concat([train, test])
    fit_full = ARIMA(full_ts, order=order).fit()
    fc_obj  = fit_full.get_forecast(steps=fc_steps)
    fc_pred = fc_obj.predicted_mean
    fc_index = pd.date_range(full_ts.index[-1] + pd.DateOffset(months=1), periods=fc_steps, freq='MS')
    fc_pred.index = fc_index
    fc_ci = clamp_ci(fc_obj.conf_int())
    fc_ci.index = fc_index
    m = compute_metrics(test, test_pred)
    m['order'] = str(order)
    return test_pred, fc_pred, fc_ci, m

def run_sarimax(train, test, fc_steps, sector_name):
    import pmdarima as pm
    FALLBACK = {'Industry': ((1,1,1),(1,1,1,12))}
    if sector_name in FALLBACK:
        s_order, s_seas = FALLBACK[sector_name]
    else:
        sa = pm.auto_arima(train, d=1, D=1, seasonal=True, m=12,
                           max_p=3, max_q=3, max_P=2, max_Q=2, max_D=1,
                           stepwise=True, information_criterion='bic', trace=False)
        s_order = sa.order
        s_seas  = sa.seasonal_order

    exog_tr = make_exog(train.index)
    exog_te = make_exog(test.index)
    fit = SARIMAX(train, exog=exog_tr, order=s_order, seasonal_order=s_seas).fit(disp=False)
    test_pred = fit.get_forecast(steps=len(test), exog=exog_te).predicted_mean
    test_pred.index = test.index

    full_ts = pd.concat([train, test])
    exog_fu = make_exog(full_ts.index)
    fc_index = pd.date_range(full_ts.index[-1] + pd.DateOffset(months=1), periods=fc_steps, freq='MS')
    exog_fc = make_exog(fc_index)
    fit_full = SARIMAX(full_ts, exog=exog_fu, order=s_order, seasonal_order=s_seas).fit(disp=False)
    fc_obj   = fit_full.get_forecast(steps=fc_steps, exog=exog_fc)
    fc_pred  = fc_obj.predicted_mean
    fc_pred.index = fc_index
    fc_ci = clamp_ci(fc_obj.conf_int())
    fc_ci.index = fc_index

    m = compute_metrics(test, test_pred)
    m['order'] = f'{s_order}x{s_seas}'
    return test_pred, fc_pred, fc_ci, m

def run_ets(train, test, fc_steps):
    fit = ExponentialSmoothing(train, trend='add', seasonal='add',
                               seasonal_periods=12, damped_trend=True).fit(optimized=True)
    test_pred = pd.Series(fit.forecast(len(test)), index=test.index)

    full_ts = pd.concat([train, test])
    fit_full = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                    seasonal_periods=12, damped_trend=True).fit(optimized=True)
    fc_index = pd.date_range(full_ts.index[-1] + pd.DateOffset(months=1), periods=fc_steps, freq='MS')
    fc_pred  = pd.Series(fit_full.forecast(fc_steps), index=fc_index)
    resid_std = np.std(fit.resid)
    z95 = 1.96
    fc_ci = pd.DataFrame({
        'lower': (fc_pred - z95 * resid_std * np.sqrt(np.arange(1, fc_steps+1))).clip(lower=0).values,
        'upper': (fc_pred + z95 * resid_std * np.sqrt(np.arange(1, fc_steps+1))).values,
    }, index=fc_index)
    m = compute_metrics(test, test_pred)
    m['order'] = 'add,add,12'
    return test_pred, fc_pred, fc_ci, m

def run_hybrid1(train, test, fc_steps, sector_name):
    """SARIMAX base + Ridge correction on lagged features"""
    import pmdarima as pm
    FALLBACK = {'Industry': ((1,1,1),(1,1,1,12))}
    if sector_name in FALLBACK:
        s_order, s_seas = FALLBACK[sector_name]
    else:
        sa = pm.auto_arima(train, d=1, D=1, seasonal=True, m=12,
                           max_p=3, max_q=3, max_P=2, max_Q=2, max_D=1,
                           stepwise=True, information_criterion='bic', trace=False)
        s_order, s_seas = sa.order, sa.seasonal_order

    exog_tr = make_exog(train.index)
    exog_te = make_exog(test.index)
    fit = SARIMAX(train, exog=exog_tr, order=s_order, seasonal_order=s_seas).fit(disp=False)
    sarima_fitted  = fit.fittedvalues
    sarima_resid   = train - sarima_fitted
    sarima_test    = fit.get_forecast(steps=len(test), exog=exog_te).predicted_mean
    sarima_test.index = test.index

    # Build lag features for Ridge on residuals
    def make_lag_features(series, n_lags=6):
        df_feat = pd.DataFrame({'y': series})
        for lag in range(1, n_lags+1):
            df_feat[f'lag_{lag}'] = df_feat['y'].shift(lag)
        df_feat['month'] = df_feat.index.month
        df_feat['sin12'] = np.sin(2*np.pi*df_feat.index.month/12)
        df_feat['cos12'] = np.cos(2*np.pi*df_feat.index.month/12)
        return df_feat.dropna()

    resid_feat = make_lag_features(sarima_resid, n_lags=6)
    X_train = resid_feat.drop(columns='y').values
    y_train = resid_feat['y'].values
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)

    # Predict residuals for test iteratively
    resid_history = sarima_resid.copy()
    ridge_resid_test = []
    for i in range(len(test)):
        lags = [resid_history.iloc[-(l)] for l in range(1, 7)]
        month_v = test.index[i].month
        feat = lags + [month_v, np.sin(2*np.pi*month_v/12), np.cos(2*np.pi*month_v/12)]
        feat_s = scaler.transform([feat])
        pred_resid = ridge.predict(feat_s)[0]
        ridge_resid_test.append(pred_resid)
        resid_history = pd.concat([resid_history, pd.Series([pred_resid], index=[test.index[i]])])

    test_pred = sarima_test + pd.Series(ridge_resid_test, index=test.index)

    # Forecast
    full_ts  = pd.concat([train, test])
    exog_fu  = make_exog(full_ts.index)
    fit_full = SARIMAX(full_ts, exog=exog_fu, order=s_order, seasonal_order=s_seas).fit(disp=False)
    fc_index = pd.date_range(full_ts.index[-1] + pd.DateOffset(months=1), periods=fc_steps, freq='MS')
    exog_fc  = make_exog(fc_index)
    fc_obj   = fit_full.get_forecast(steps=fc_steps, exog=exog_fc)
    fc_base  = fc_obj.predicted_mean
    fc_base.index = fc_index
    fc_ci = clamp_ci(fc_obj.conf_int())
    fc_ci.index = fc_index

    # Ridge forecast (residuals → decay to 0)
    decay_resid = [r * (0.85**i) for i, r in enumerate(ridge_resid_test[-fc_steps:][:fc_steps]
                    + [0]*(max(0, fc_steps - len(ridge_resid_test))))]
    if len(decay_resid) < fc_steps:
        decay_resid += [0] * (fc_steps - len(decay_resid))
    fc_pred = fc_base + pd.Series(decay_resid[:fc_steps], index=fc_index)
    fc_pred = fc_pred.clip(lower=0)

    m = compute_metrics(test, test_pred)
    m['order'] = f'SARIMAX+Ridge'
    return test_pred, fc_pred, fc_ci, m

def run_hybrid2(train, test, fc_steps):
    """ETS base + Ridge correction"""
    fit_ets = ExponentialSmoothing(train, trend='add', seasonal='add',
                                   seasonal_periods=12, damped_trend=True).fit(optimized=True)
    ets_fitted  = fit_ets.fittedvalues
    ets_resid   = train - ets_fitted
    ets_test    = pd.Series(fit_ets.forecast(len(test)), index=test.index)

    def make_lag_features(series, n_lags=6):
        df_feat = pd.DataFrame({'y': series})
        for lag in range(1, n_lags+1):
            df_feat[f'lag_{lag}'] = df_feat['y'].shift(lag)
        df_feat['month']  = df_feat.index.month
        df_feat['sin12']  = np.sin(2*np.pi*df_feat.index.month/12)
        df_feat['cos12']  = np.cos(2*np.pi*df_feat.index.month/12)
        return df_feat.dropna()

    resid_feat = make_lag_features(ets_resid, n_lags=6)
    X_tr = resid_feat.drop(columns='y').values
    y_tr = resid_feat['y'].values
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr)

    resid_history = ets_resid.copy()
    ridge_resid_test = []
    for i in range(len(test)):
        lags = [resid_history.iloc[-(l)] for l in range(1, 7)]
        month_v = test.index[i].month
        feat  = lags + [month_v, np.sin(2*np.pi*month_v/12), np.cos(2*np.pi*month_v/12)]
        feat_s = scaler.transform([feat])
        pred_r = ridge.predict(feat_s)[0]
        ridge_resid_test.append(pred_r)
        resid_history = pd.concat([resid_history, pd.Series([pred_r], index=[test.index[i]])])

    test_pred = ets_test + pd.Series(ridge_resid_test, index=test.index)

    # Forecast with ETS
    full_ts  = pd.concat([train, test])
    fit_full = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                    seasonal_periods=12, damped_trend=True).fit(optimized=True)
    fc_index = pd.date_range(full_ts.index[-1] + pd.DateOffset(months=1), periods=fc_steps, freq='MS')
    fc_pred  = pd.Series(fit_full.forecast(fc_steps), index=fc_index)
    resid_std = np.std(fit_ets.resid)
    fc_ci = pd.DataFrame({
        'lower': (fc_pred - 1.96*resid_std*np.sqrt(np.arange(1, fc_steps+1))).clip(lower=0).values,
        'upper': (fc_pred + 1.96*resid_std*np.sqrt(np.arange(1, fc_steps+1))).values,
    }, index=fc_index)
    # Ridge decay on forecast
    decay = [r*(0.85**i) for i,r in enumerate(ridge_resid_test[-fc_steps:][:fc_steps]
              + [0]*max(0, fc_steps-len(ridge_resid_test)))]
    if len(decay) < fc_steps:
        decay += [0]*(fc_steps - len(decay))
    fc_pred = fc_pred + pd.Series(decay[:fc_steps], index=fc_index)
    fc_pred = fc_pred.clip(lower=0)

    m = compute_metrics(test, test_pred)
    m['order'] = 'ETS+Ridge'
    return test_pred, fc_pred, fc_ci, m

# ─────────────────────────────────────────────
# Run All Models for ONE sector
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_all_models(train_vals, train_idx, test_vals, test_idx, fc_steps, sector_name):
    train = pd.Series(train_vals, index=pd.DatetimeIndex(train_idx))
    test  = pd.Series(test_vals,  index=pd.DatetimeIndex(test_idx))
    results = {}

    try:
        tp, fp, fc, m = run_arima(train, test, fc_steps)
        results['ARIMA'] = {'test_pred': tp, 'fc_pred': fp, 'fc_ci': fc, 'metrics': m}
    except Exception as e:
        st.warning(f"ARIMA failed for {sector_name}: {e}")

    try:
        tp, fp, fc, m = run_sarimax(train, test, fc_steps, sector_name)
        results['SARIMAX'] = {'test_pred': tp, 'fc_pred': fp, 'fc_ci': fc, 'metrics': m}
    except Exception as e:
        st.warning(f"SARIMAX failed for {sector_name}: {e}")

    try:
        tp, fp, fc, m = run_ets(train, test, fc_steps)
        results['ETS'] = {'test_pred': tp, 'fc_pred': fp, 'fc_ci': fc, 'metrics': m}
    except Exception as e:
        st.warning(f"ETS failed for {sector_name}: {e}")

    try:
        tp, fp, fc, m = run_hybrid1(train, test, fc_steps, sector_name)
        results['Hybrid1(SARIMAX+Ridge)'] = {'test_pred': tp, 'fc_pred': fp, 'fc_ci': fc, 'metrics': m}
    except Exception as e:
        st.warning(f"Hybrid1 failed for {sector_name}: {e}")

    try:
        tp, fp, fc, m = run_hybrid2(train, test, fc_steps)
        results['Hybrid2(ETS+Ridge)'] = {'test_pred': tp, 'fc_pred': fp, 'fc_ci': fc, 'metrics': m}
    except Exception as e:
        st.warning(f"Hybrid2 failed for {sector_name}: {e}")

    return results

# ─────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────
def plot_sector_forecast(ts, train, test, model_results, selected_models, sector_name, fc_steps, show_ci):
    cfg = SECTOR_CFG[sector_name]
    color = cfg['color']

    fig = go.Figure()
    full_ts = pd.concat([train, test])
    fc_start = full_ts.index[-1] + pd.DateOffset(months=1)
    fc_index = pd.date_range(fc_start, periods=fc_steps, freq='MS')

    # Background shading
    fig.add_vrect(x0=str(train.index[0])[:7], x1=str(train.index[-1])[:7],
                  fillcolor='rgba(100,149,237,0.05)', layer='below', line_width=0,
                  annotation_text="Train", annotation_position="top left",
                  annotation_font=dict(color='rgba(100,149,237,0.6)', size=10))
    fig.add_vrect(x0=str(test.index[0])[:7], x1=str(test.index[-1])[:7],
                  fillcolor='rgba(255,165,0,0.06)', layer='below', line_width=0,
                  annotation_text="Test", annotation_position="top left",
                  annotation_font=dict(color='rgba(255,165,0,0.6)', size=10))
    if len(fc_index) > 0:
        fig.add_vrect(x0=str(fc_index[0])[:7], x1=str(fc_index[-1])[:7],
                      fillcolor='rgba(34,197,94,0.05)', layer='below', line_width=0,
                      annotation_text="Forecast", annotation_position="top left",
                      annotation_font=dict(color='rgba(34,197,94,0.6)', size=10))

    # Actual line
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        name='Actual', mode='lines',
        line=dict(color='#e8edf5', width=2),
        hovertemplate='%{x|%Y-%m}<br>Actual: %{y:,.0f}<extra></extra>'
    ))

    # Model forecasts
    for mname in selected_models:
        if mname not in model_results:
            continue
        r = model_results[mname]
        mcolor = MODEL_COLORS.get(mname, '#aaa')

        # Test prediction
        fig.add_trace(go.Scatter(
            x=r['test_pred'].index, y=r['test_pred'].values,
            name=f'{mname} (test)', mode='lines',
            line=dict(color=mcolor, width=1.5, dash='dot'),
            showlegend=True,
            hovertemplate=f'%{{x|%Y-%m}}<br>{mname}: %{{y:,.0f}}<extra></extra>'
        ))

        # Forecast
        fc = r['fc_pred']
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values,
            name=f'{mname} (forecast)', mode='lines',
            line=dict(color=mcolor, width=2.5, dash='dash'),
            hovertemplate=f'%{{x|%Y-%m}}<br>Forecast: %{{y:,.0f}}<extra></extra>'
        ))

        # CI band
        if show_ci and 'fc_ci' in r:
            ci = r['fc_ci']
            fig.add_trace(go.Scatter(
                x=list(ci.index) + list(ci.index[::-1]),
                y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                fill='toself', fillcolor=f'rgba({int(mcolor[1:3],16)},{int(mcolor[3:5],16)},{int(mcolor[5:7],16)},0.10)',
                line=dict(color='rgba(0,0,0,0)'), name=f'{mname} CI 95%',
                showlegend=False, hoverinfo='skip'
            ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,15,25,0.6)',
        font=dict(family='Inter', color='#c5d8ee', size=11),
        legend=dict(bgcolor='rgba(10,15,30,0.8)', bordercolor='rgba(56,139,220,0.2)',
                    borderwidth=1, font=dict(size=10), orientation='h',
                    yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(56,139,220,0.1)', tickformat='%Y',
                   dtick='M12', showline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(56,139,220,0.08)',
                   title='1,000 Tons CO₂', tickformat=',.0f'),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )
    return fig

def plot_metrics_radar(metrics_dict):
    models  = list(metrics_dict.keys())
    metrics = ['MAE', 'RMSE', 'MAPE']
    # Normalize each metric (lower=better → invert to 0-1 scale)
    vals = {m: metrics_dict[m] for m in models}
    norm_data = {}
    for metric in metrics:
        raw = [vals[m][metric] for m in models]
        mn, mx = min(raw), max(raw)
        if mx == mn:
            norm_data[metric] = [1.0]*len(models)
        else:
            norm_data[metric] = [1 - (v-mn)/(mx-mn) for v in raw]  # invert

    colors_list = ['#2E86C1','#1abc9c','#e74c3c','#f1c40f','#e67e22']
    fig = go.Figure()
    for i, m in enumerate(models):
        r_vals = [norm_data[metric][i] for metric in metrics] + [norm_data[metrics[0]][i]]
        theta  = metrics + [metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=theta,
            fill='toself',
            fillcolor=f'rgba({int(colors_list[i%len(colors_list)][1:3],16)},{int(colors_list[i%len(colors_list)][3:5],16)},{int(colors_list[i%len(colors_list)][5:7],16)},0.15)',
            line=dict(color=colors_list[i%len(colors_list)], width=2),
            name=m
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(10,15,25,0.6)',
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                            gridcolor='rgba(56,139,220,0.15)'),
            angularaxis=dict(tickfont=dict(color='#c5d8ee', size=11),
                             gridcolor='rgba(56,139,220,0.15)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#c5d8ee'),
        legend=dict(bgcolor='rgba(10,15,30,0.8)', bordercolor='rgba(56,139,220,0.2)',
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=True
    )
    return fig

def plot_mape_bar(all_metrics):
    """all_metrics: {sector: {model: metrics}}"""
    rows = []
    for sector, mdict in all_metrics.items():
        for model, m in mdict.items():
            rows.append({'Sector': sector, 'Model': model, 'MAPE': m['MAPE']})
    df_bar = pd.DataFrame(rows)
    if df_bar.empty:
        return go.Figure()

    fig = px.bar(df_bar, x='Model', y='MAPE', color='Sector',
                 barmode='group',
                 color_discrete_map={'Power':'#2E86C1','Transport':'#E67E22','Industry':'#1E8449'})
    fig.add_hline(y=5, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                  annotation_text='5% threshold', annotation_font=dict(color='#aaa', size=10))
    fig.add_hline(y=10, line_dash='dot', line_color='rgba(255,100,100,0.4)',
                  annotation_text='10% threshold', annotation_font=dict(color='#f87171', size=10))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,15,25,0.6)',
        font=dict(family='Inter', color='#c5d8ee', size=11),
        xaxis=dict(showgrid=False, tickangle=-25),
        yaxis=dict(showgrid=True, gridcolor='rgba(56,139,220,0.1)', title='MAPE (%)'),
        legend=dict(bgcolor='rgba(10,15,30,0.8)', bordercolor='rgba(56,139,220,0.2)', borderwidth=1),
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
    )
    return fig

def plot_forecast_comparison(all_fc, sector_colors):
    """Overlay best model forecast for all 3 sectors"""
    fig = go.Figure()
    for sector, data in all_fc.items():
        if data is None: continue
        color = sector_colors[sector]
        fig.add_trace(go.Scatter(
            x=data['actual'].index, y=data['actual'].values,
            name=f'{sector} (actual)', mode='lines',
            line=dict(color=color, width=1.5),
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=data['fc'].index, y=data['fc'].values,
            name=f'{sector} (forecast)', mode='lines',
            line=dict(color=color, width=2.5, dash='dash'),
        ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,15,25,0.6)',
        font=dict(family='Inter', color='#c5d8ee', size=11),
        legend=dict(bgcolor='rgba(10,15,30,0.8)', bordercolor='rgba(56,139,220,0.2)',
                    borderwidth=1, font=dict(size=10), orientation='h',
                    yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(56,139,220,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(56,139,220,0.08)', title='1,000 Tons CO₂', tickformat=',.0f'),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )
    return fig

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 20px 0;'>
        <div style='font-size:2rem;'>🌿</div>
        <div style='font-size:0.9rem; font-weight:700; color:#7db3e0; letter-spacing:1px;'>CO₂ FORECAST</div>
        <div style='font-size:0.72rem; color:#3d5c7a; margin-top:2px;'>Academic Conference Edition</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader(
        "📁 อัปโหลดไฟล์ข้อมูล (EPPO Excel)",
        type=['xlsx', 'xls'],
        help="ไฟล์ Excel ของ EPPO ที่มี 3 sheets: eppo-power-dataset, eppo-transport-dataset, eppo-industry-dataset"
    )

    st.divider()
    st.markdown("<div style='font-size:0.8rem; color:#5d8aad; font-weight:600; letter-spacing:1px; margin-bottom:8px;'>⚙️ FORECAST SETTINGS</div>", unsafe_allow_html=True)

    fc_months = st.slider("จำนวนเดือนที่พยากรณ์", min_value=6, max_value=36, value=24, step=6,
                          help="พยากรณ์ล่วงหน้ากี่เดือนหลังจาก Test period")

    selected_models = st.multiselect(
        "เลือกโมเดลที่ต้องการแสดง",
        options=MODELS_LIST,
        default=['SARIMAX', 'Hybrid1(SARIMAX+Ridge)', 'Hybrid2(ETS+Ridge)'],
        help="เลือกได้หลายโมเดล"
    )

    show_ci = st.checkbox("แสดง Confidence Interval (95%)", value=True)

    st.divider()
    st.markdown("<div style='font-size:0.8rem; color:#5d8aad; font-weight:600; letter-spacing:1px; margin-bottom:8px;'>🏆 BEST MODELS</div>", unsafe_allow_html=True)
    for sector, model in BEST_MODELS.items():
        icon = SECTOR_CFG[sector]['icon']
        st.markdown(f"<div style='font-size:0.78rem; color:#c5d8ee; margin-bottom:4px;'>{icon} <b>{sector}</b>: <span style='color:#6ee7a6;'>{model}</span></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:0.72rem; color:#3d5c7a;'>Train: 2010–2023 | Test: 2024–2025<br>COVID dummy: Apr 2020 – Jun 2021</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🌿 CO₂ Emission Forecasting Dashboard</div>
    <div class='hero-subtitle'>Thailand Energy Sector — Power Generation · Transport · Industry</div>
    <div class='hero-badges'>
        <span class='badge'>SARIMAX + COVID Dummy</span>
        <span class='badge'>Hybrid Models</span>
        <span class='badge'>7 Models Comparison</span>
        <span class='badge badge-green'>Academic Edition</span>
        <span class='badge'>Data: EPPO 2010–2025</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
if uploaded is None:
    # Demo / Landing State
    st.markdown("""
    <div class='info-box'>
        <b>📋 วิธีใช้งาน:</b> อัปโหลดไฟล์ Excel ของ EPPO ที่มี 3 sheets (<code>eppo-power-dataset</code>, 
        <code>eppo-transport-dataset</code>, <code>eppo-industry-dataset</code>) จากนั้นระบบจะรันโมเดล 
        ARIMA, SARIMAX+COVID, ETS, Hybrid1, Hybrid2 อัตโนมัติ และแสดงผลพยากรณ์ทั้ง 3 sector บนหน้าเดียวกัน
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val, unit in [
        (col1, "Sectors", "3", "Power · Transport · Industry"),
        (col2, "Models", "5", "ARIMA · SARIMAX · ETS · Hybrid×2"),
        (col3, "Train Period", "168", "months (2010–2023)"),
        (col4, "Test Period", "24", "months (2024–2025)"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-unit'>{unit}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture diagram
    tab1, tab2 = st.tabs(["📐 Model Architecture", "📊 Methodology"])
    with tab1:
        st.markdown("""
        | โมเดล | ประเภท | จุดเด่น | Best For |
        |---|---|---|---|
        | **ARIMA** | Time Series | Baseline, non-seasonal | Quick benchmark |
        | **SARIMAX+COVID** | Time Series | Seasonal + COVID dummy | Industry |
        | **ETS** | Exponential Smoothing | Damped trend, robust | Stable trends |
        | **Hybrid1 (SARIMAX+Ridge)** | Hybrid | SARIMAX base + residual correction | Power |
        | **Hybrid2 (ETS+Ridge)** | Hybrid | ETS base + lag features | Transport |
        """)
    with tab2:
        st.markdown("""
        **Pipeline:**
        1. **Data Parse** — อ่านข้อมูล EPPO Excel 3 sheets, แปลง month/year → DatetimeIndex
        2. **Split** — Train: 2010–2023 | Test: 2024–2025 | Forecast: N months ahead
        3. **COVID Dummy** — Binary variable (Apr 2020 – Jun 2021) ใส่ใน SARIMAX exogenous
        4. **Auto-Order** — pmdarima auto_arima (BIC) เลือก (p,d,q)(P,D,Q,12) อัตโนมัติ
        5. **Hybrid** — SARIMAX/ETS เป็น base → Ridge regression แก้ residuals ด้วย lag features
        6. **Metrics** — MAE, RMSE, MAPE, R² บน Test period (2024–2025)
        """)
    st.stop()

# ─────────────────────────────────────────────
# Load & Parse Data
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    buf = io.BytesIO(file_bytes)
    power_df  = parse_eppo_sheet(buf, 'eppo-power-dataset',    ['oil','coal','gas','total'])
    buf.seek(0)
    transp_df = parse_eppo_sheet(buf, 'eppo-transport-dataset', ['oil','gas','total'])
    buf.seek(0)
    indust_df = parse_eppo_sheet(buf, 'eppo-industry-dataset',  ['oil','coal','gas','total'])
    return power_df, transp_df, indust_df

with st.spinner("📂 กำลังโหลดและประมวลผลข้อมูล..."):
    try:
        power_df, transp_df, indust_df = load_data(uploaded.read())
        SECTORS = {'Power': power_df, 'Transport': transp_df, 'Industry': indust_df}
    except Exception as e:
        st.error(f"❌ ไม่สามารถอ่านไฟล์ได้: {e}")
        st.stop()

# ─────────────────────────────────────────────
# Prepare Split
# ─────────────────────────────────────────────
sector_splits = {}
for name, df in SECTORS.items():
    ts    = df['total'].loc[TRAIN_START:]
    train = ts.loc[TRAIN_START:TRAIN_END]
    test  = ts.loc[TEST_START:TEST_END]
    if len(test) == 0:
        test = ts.iloc[-24:]
    sector_splits[name] = (ts, train, test)

# ─────────────────────────────────────────────
# Run Models (cached)
# ─────────────────────────────────────────────
all_results = {}
progress_bar = st.progress(0, text="⚙️ กำลังฝึกโมเดล...")

for i, (name, (ts, train, test)) in enumerate(sector_splits.items()):
    progress_bar.progress((i) / 3, text=f"⚙️ กำลังฝึกโมเดล [{name}]...")
    all_results[name] = run_all_models(
        train.values, train.index, test.values, test.index, fc_months, name
    )
    progress_bar.progress((i+1) / 3, text=f"✅ [{name}] เสร็จแล้ว")

progress_bar.empty()

# ─────────────────────────────────────────────
# TABS: Overview | Per Sector | Metrics | Export
# ─────────────────────────────────────────────
tab_overview, tab_power, tab_transport, tab_industry, tab_metrics, tab_export = st.tabs([
    "📊 Overview", "⚡ Power", "🚗 Transport", "🏭 Industry", "📈 Performance", "💾 Export"
])

# ══════════════════════════════════════════════
# TAB: OVERVIEW — All 3 Sectors on One Page
# ══════════════════════════════════════════════
with tab_overview:
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Summary metric row
    mcols = st.columns(6)
    metric_items = []
    for name in ['Power','Transport','Industry']:
        best_key = BEST_MODELS[name]
        if name in all_results and best_key in all_results[name]:
            m = all_results[name][best_key]['metrics']
            metric_items.append((name, best_key, m))

    for idx, (name, model, m) in enumerate(metric_items):
        cfg = SECTOR_CFG[name]
        grade = mape_grade(m['MAPE'])
        mcols[idx*2].markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{cfg['icon']} {name}</div>
            <div class='metric-value {grade}'>{m['MAPE']:.2f}%</div>
            <div class='metric-unit'>MAPE (Best Model)</div>
        </div>
        """, unsafe_allow_html=True)
        mcols[idx*2+1].markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>MAE</div>
            <div class='metric-value' style='font-size:1.1rem;'>{m['MAE']:,.0f}</div>
            <div class='metric-unit'>1,000 Tons</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # 3 Sector charts stacked
    for name in ['Power', 'Transport', 'Industry']:
        cfg = SECTOR_CFG[name]
        ts, train, test = sector_splits[name]
        results = all_results.get(name, {})

        st.markdown(f"""
        <div class='sector-header {cfg['css']}'>
            <span>{cfg['icon']}</span>
            <span>{name} Generation — CO₂ Emission Forecast</span>
            <span class='best-tag'>Best: {BEST_MODELS[name]}</span>
        </div>
        """, unsafe_allow_html=True)

        models_to_show = [m for m in selected_models if m in results] if selected_models else list(results.keys())[:3]
        if not models_to_show and results:
            models_to_show = list(results.keys())[:2]

        fig = plot_sector_forecast(ts, train, test, results, models_to_show, name, fc_months, show_ci)
        st.plotly_chart(fig, use_container_width=True)

        # Quick metrics for this sector
        if results:
            mc_list = st.columns(min(len(models_to_show), 5))
            for i, mname in enumerate(models_to_show):
                if mname in results and i < len(mc_list):
                    m = results[mname]['metrics']
                    is_best = (mname == BEST_MODELS[name])
                    grade = mape_grade(m['MAPE'])
                    mc_list[i].markdown(f"""
                    <div class='metric-card' style='{"border-color:rgba(34,197,94,0.4);" if is_best else ""}'>
                        <div class='metric-label'>{mname[:20]}{"⭐" if is_best else ""}</div>
                        <div class='metric-value {grade}' style='font-size:1.15rem;'>{m['MAPE']:.2f}%</div>
                        <div class='metric-unit'>MAPE</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Cross-sector comparison chart
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🌏 Cross-Sector Forecast Comparison (Best Models)")

    all_fc = {}
    for name in ['Power','Transport','Industry']:
        best_key = BEST_MODELS[name]
        ts, _, _ = sector_splits[name]
        if name in all_results and best_key in all_results[name]:
            all_fc[name] = {
                'actual': ts,
                'fc': all_results[name][best_key]['fc_pred']
            }
        else:
            all_fc[name] = None

    fig_cross = plot_forecast_comparison(all_fc, {s: SECTOR_CFG[s]['color'] for s in SECTOR_CFG})
    st.plotly_chart(fig_cross, use_container_width=True)


# ══════════════════════════════════════════════
# Helper: single sector tab content
# ══════════════════════════════════════════════
def render_sector_tab(name):
    cfg = SECTOR_CFG[name]
    ts, train, test = sector_splits[name]
    results = all_results.get(name, {})
    best_key = BEST_MODELS[name]

    st.markdown(f"""
    <div class='sector-header {cfg['css']}'>
        {cfg['icon']} {name} — Detailed Analysis
        <span class='best-tag'>Best Model: {best_key}</span>
    </div>
    """, unsafe_allow_html=True)

    # Main forecast chart
    st.markdown("##### 📈 Forecast Chart")
    models_to_show = selected_models if selected_models else list(results.keys())
    fig = plot_sector_forecast(ts, train, test, results, models_to_show, name, fc_months, show_ci)
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown("##### 📋 Model Performance Comparison")
    rows = []
    for mname, r in results.items():
        m = r['metrics']
        is_best = (mname == best_key)
        rows.append({
            'Model'  : f"⭐ {mname}" if is_best else mname,
            'MAE'    : f"{m['MAE']:,.1f}",
            'RMSE'   : f"{m['RMSE']:,.1f}",
            'MAPE %' : f"{m['MAPE']:.2f}%",
            'R²'     : f"{m['R²']:.4f}",
            'Order'  : m.get('order','—'),
            'Best'   : '✅' if is_best else '',
        })
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Forecast Table
    st.markdown("##### 🔢 Forecast Values (Best Model)")
    if best_key in results:
        fc_pred = results[best_key]['fc_pred']
        fc_ci   = results[best_key].get('fc_ci')
        fc_rows = []
        for i, (date, val) in enumerate(fc_pred.items()):
            row = {'Date': date.strftime('%Y-%m'), 'Forecast (1,000T)': f"{val:,.1f}"}
            if fc_ci is not None:
                row['CI Lower'] = f"{fc_ci.iloc[i,0]:,.1f}"
                row['CI Upper'] = f"{fc_ci.iloc[i,1]:,.1f}"
            fc_rows.append(row)
        df_fc = pd.DataFrame(fc_rows)
        st.dataframe(df_fc, use_container_width=True, hide_index=True, height=300)

    # Fuel Mix
    st.markdown("##### ⛽ Fuel Mix (2010–Present)")
    df = SECTORS[name]
    fuel_cols = [c for c in df.columns if c != 'total']
    fig_fuel = go.Figure()
    df_fuel = df.loc[TRAIN_START:]
    colors_fuel = ['#e74c3c','#3498db','#2ecc71','#f39c12']
    for j, fc_name in enumerate(fuel_cols):
        fig_fuel.add_trace(go.Scatter(
            x=df_fuel.index, y=df_fuel[fc_name].values,
            name=fc_name.capitalize(), mode='lines',
            line=dict(color=colors_fuel[j%len(colors_fuel)], width=2),
            fill='tonexty' if j > 0 else 'tozeroy',
            fillcolor=f'rgba({int(colors_fuel[j%len(colors_fuel)][1:3],16)},{int(colors_fuel[j%len(colors_fuel)][3:5],16)},{int(colors_fuel[j%len(colors_fuel)][5:7],16)},0.15)',
        ))
    fig_fuel.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,15,25,0.6)',
        font=dict(family='Inter', color='#c5d8ee', size=11),
        legend=dict(bgcolor='rgba(10,15,30,0.8)', bordercolor='rgba(56,139,220,0.2)', borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor='rgba(56,139,220,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(56,139,220,0.08)', title='1,000 Tons'),
        margin=dict(l=10, r=10, t=20, b=10), height=280,
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

with tab_power:
    render_sector_tab('Power')

with tab_transport:
    render_sector_tab('Transport')

with tab_industry:
    render_sector_tab('Industry')

# ══════════════════════════════════════════════
# TAB: METRICS
# ══════════════════════════════════════════════
with tab_metrics:
    st.markdown("#### 📈 Model Performance — All Sectors & Models")

    all_metrics_flat = {}
    for name, results in all_results.items():
        all_metrics_flat[name] = {mname: r['metrics'] for mname, r in results.items()}

    # MAPE Bar chart
    st.markdown("##### MAPE Comparison (%) — Lower is Better")
    fig_bar = plot_mape_bar(all_metrics_flat)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar per sector
    st.markdown("##### Model Radar Charts (Normalized Score — Higher is Better)")
    r_cols = st.columns(3)
    for i, name in enumerate(['Power','Transport','Industry']):
        cfg = SECTOR_CFG[name]
        with r_cols[i]:
            st.markdown(f"<div style='text-align:center;font-size:0.85rem;color:{cfg['color']};font-weight:600;margin-bottom:8px;'>{cfg['icon']} {name}</div>", unsafe_allow_html=True)
            if name in all_metrics_flat:
                fig_r = plot_metrics_radar(all_metrics_flat[name])
                st.plotly_chart(fig_r, use_container_width=True)

    # Heatmap of MAPE
    st.markdown("##### MAPE Heatmap")
    hm_data = {}
    all_model_names = sorted(set(m for v in all_metrics_flat.values() for m in v.keys()))
    for mname in all_model_names:
        row = {}
        for sname in ['Power','Transport','Industry']:
            row[sname] = all_metrics_flat.get(sname,{}).get(mname,{}).get('MAPE', None)
        hm_data[mname] = row
    df_hm = pd.DataFrame(hm_data).T
    fig_hm = px.imshow(
        df_hm.values.astype(float),
        x=df_hm.columns.tolist(),
        y=df_hm.index.tolist(),
        color_continuous_scale='RdYlGn_r',
        text_auto='.2f',
        labels=dict(color='MAPE %'),
    )
    fig_hm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#c5d8ee', size=11),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Summary table
    st.markdown("##### 📋 Full Metrics Summary")
    summary_rows = []
    for sname in ['Power','Transport','Industry']:
        for mname in all_model_names:
            m = all_metrics_flat.get(sname,{}).get(mname)
            if m:
                summary_rows.append({
                    'Sector': sname,
                    'Model': mname,
                    'MAE': f"{m['MAE']:,.1f}",
                    'RMSE': f"{m['RMSE']:,.1f}",
                    'MAPE %': f"{m['MAPE']:.2f}%",
                    'R²': f"{m['R²']:.4f}",
                    'Best': '⭐' if BEST_MODELS.get(sname) == mname else '',
                })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB: EXPORT
# ══════════════════════════════════════════════
with tab_export:
    st.markdown("#### 💾 Export Data")
    st.markdown("""
    <div class='info-box'>
        ดาวน์โหลดข้อมูลพยากรณ์ในรูปแบบ CSV สำหรับนำไปวิเคราะห์เพิ่มเติมหรือสร้างกราฟใน Excel / Canva
    </div>
    """, unsafe_allow_html=True)

    export_rows = []
    for name, results in all_results.items():
        ts, train, test = sector_splits[name]
        full_ts = ts
        for mname, r in results.items():
            for date, val in full_ts.items():
                is_test = date in r['test_pred'].index
                period  = 'Test' if is_test else 'Train'
                export_rows.append({
                    'Sector': name, 'Model': mname,
                    'Date': date.strftime('%Y-%m'),
                    'Year': date.year, 'Month': date.strftime('%b'),
                    'Period': period,
                    'Actual': round(val, 2),
                    'Predicted': round(r['test_pred'].get(date, float('nan')), 2) if is_test else '',
                    'Forecast': '', 'CI_Lower': '', 'CI_Upper': '',
                })
            fc = r['fc_pred']
            fc_ci = r.get('fc_ci')
            for i, (date, val) in enumerate(fc.items()):
                export_rows.append({
                    'Sector': name, 'Model': mname,
                    'Date': date.strftime('%Y-%m'),
                    'Year': date.year, 'Month': date.strftime('%b'),
                    'Period': 'Forecast',
                    'Actual': '',
                    'Predicted': '',
                    'Forecast': round(val, 2),
                    'CI_Lower': round(fc_ci.iloc[i,0], 2) if fc_ci is not None else '',
                    'CI_Upper': round(fc_ci.iloc[i,1], 2) if fc_ci is not None else '',
                })

    df_export = pd.DataFrame(export_rows)

    c1, c2 = st.columns(2)
    with c1:
        csv_all = df_export.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("⬇️ Download All Models (CSV)", csv_all,
                           "CO2_Forecast_AllModels.csv", "text/csv", use_container_width=True)

    with c2:
        df_best = df_export[df_export['Model'].isin(BEST_MODELS.values())]
        csv_best = df_best.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("⬇️ Download Best Models Only (CSV)", csv_best,
                           "CO2_Forecast_BestModels.csv", "text/csv", use_container_width=True)

    st.dataframe(df_export.head(50), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🌿 CO₂ Emission Forecasting Dashboard &nbsp;|&nbsp; 
    SARIMAX · ETS · Hybrid Models &nbsp;|&nbsp; 
    Data Source: EPPO Thailand &nbsp;|&nbsp; 
    Academic Conference Edition
</div>
""", unsafe_allow_html=True)
