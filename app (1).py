"""
CO₂ Emission Forecasting — 3 Sectors (Power, Transport, Industry)
Academic Conference Presentation App
7 Models: ARIMA, SARIMAX+COVID, ETS, Prophet, Hybrid1, Hybrid2, Hybrid3
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import pmdarima as pm
import io, base64, os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Emission Forecast — Thailand 3 Sectors",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Academic Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header banner */
.main-header {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 50%, #0d2137 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #00c896;
}
.main-header h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { color: #a8d8ea; margin: 0.3rem 0 0; font-size: 0.95rem; }

/* Sector cards */
.sector-card {
    background: linear-gradient(135deg, #f8fbff, #eef4fd);
    border: 1px solid #d0e4f7;
    border-left: 4px solid var(--sector-color, #2E86C1);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

/* Metric boxes */
.metric-row {
    display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1rem;
}
.metric-box {
    background: white;
    border: 1px solid #e2ecf9;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    flex: 1; min-width: 110px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-box .label { font-size: 0.72rem; color: #6b7a8d; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-box .value { font-size: 1.35rem; font-weight: 700; color: #1a3a5c; }
.metric-box .unit  { font-size: 0.7rem; color: #8a97a8; }

/* Section titles */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #0d2137;
    border-bottom: 2px solid #00c896;
    padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #0d2137; }
section[data-testid="stSidebar"] * { color: #d0e8f7 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #a8d8ea !important; font-weight: 600; }

/* Best model badge */
.best-badge {
    display: inline-block;
    background: #00c896; color: white;
    font-size: 0.72rem; font-weight: 700;
    padding: 2px 8px; border-radius: 20px;
    margin-left: 6px; vertical-align: middle;
}

/* Footer */
.footer {
    text-align: center; color: #8a97a8;
    font-size: 0.78rem; margin-top: 2rem;
    padding-top: 1rem; border-top: 1px solid #e2ecf9;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TRAIN_START = '2010-01'
TRAIN_END   = '2023-12'
TEST_START  = '2024-01'
TEST_END    = '2025-12'
COVID_START = '2020-04'
COVID_END   = '2021-06'

COLORS = {
    'actual'   : '#1A252F',
    'ARIMA'    : '#C0392B',
    'SARIMAX'  : '#148F77',
    'ETS'      : '#2874A6',
    'Prophet'  : '#8E44AD',
    'Hybrid1'  : '#D4AC0D',
    'Hybrid2'  : '#E67E22',
    'Hybrid3'  : '#17A589',
    'Power'    : '#2E86C1',
    'Transport': '#E67E22',
    'Industry' : '#1E8449',
}

SECTOR_BEST = {
    'Power'    : 'Hybrid1',
    'Transport': 'Hybrid2',
    'Industry' : 'SARIMAX',
}

MODEL_INFO = {
    'ARIMA'  : 'ARIMA — Baseline non-seasonal time series',
    'SARIMAX': 'SARIMAX+COVID — Seasonal with COVID-19 dummy variable',
    'ETS'    : 'ETS — Exponential Smoothing (Holt-Winters)',
    'Prophet': 'Prophet — Meta\'s decomposition-based forecasting',
    'Hybrid1': 'Hybrid 1 — SARIMAX + Ridge Regression',
    'Hybrid2': 'Hybrid 2 — ETS + Ridge Regression',
    'Hybrid3': 'Hybrid 3 — SARIMAX + Prophet Ensemble',
}

SARIMA_FALLBACK = {
    'Power'    : None,
    'Transport': None,
    'Industry' : ((1,1,1),(1,1,1,12)),
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
MONTH_MAP = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}

def parse_eppo_sheet(file_obj, sheet_name, col_names):
    raw = pd.read_excel(file_obj, sheet_name=sheet_name, header=None)
    records, yr = [], None
    for _, row in raw.iterrows():
        cell = str(row.iloc[0]).strip()
        try:
            v = int(float(cell))
            if 1990 <= v <= 2030:
                yr = v
            continue
        except: pass
        if yr and cell.lower() in MONTH_MAP:
            month = MONTH_MAP[cell.lower()]
            try:
                vals = [float(row.iloc[i]) for i in range(1, len(col_names)+1)]
                rec  = {'date': pd.Timestamp(year=yr, month=month, day=1)}
                rec.update(dict(zip(col_names, vals)))
                records.append(rec)
            except: pass
    df = pd.DataFrame(records).sort_values('date').set_index('date')
    return df

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def split(df, col='total'):
    ts    = df[col]
    train = ts.loc[TRAIN_START:TRAIN_END]
    test  = ts.loc[TEST_START:TEST_END]
    return ts, train, test

def make_exog(index):
    dummy = ((index >= COVID_START) & (index <= COVID_END)).astype(int)
    return pd.DataFrame({'covid': dummy}, index=index)

def clamp_ci(ci):
    c = ci.copy()
    c.iloc[:, 0] = c.iloc[:, 0].clip(lower=0)
    return c

def calc_metrics(actual, predicted, label=''):
    idx = actual.index.intersection(predicted.index)
    a, p = actual[idx].values, predicted[idx].values
    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mape = np.mean(np.abs((a - p) / a)) * 100
    return {'Model': label, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# ─────────────────────────────────────────────
# MODEL FITTING (cached per sector)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fit_all_models(sector_name, train_vals, test_vals, ts_vals, fc_periods):
    train = pd.Series(train_vals['values'], index=pd.DatetimeIndex(train_vals['index']))
    test  = pd.Series(test_vals['values'],  index=pd.DatetimeIndex(test_vals['index']))
    ts    = pd.Series(ts_vals['values'],    index=pd.DatetimeIndex(ts_vals['index']))
    FC_INDEX = pd.date_range('2026-01-01', periods=fc_periods, freq='MS')

    results   = {}
    forecasts = {}

    # ── ARIMA ────────────────────────────────
    ar = pm.auto_arima(train, d=1, seasonal=False,
                       max_p=3, max_q=3, stepwise=True,
                       information_criterion='bic', trace=False)
    arima_order = ar.order
    fit_a = ARIMA(train, order=arima_order).fit()
    tp_a  = fit_a.get_forecast(steps=len(test)).predicted_mean
    tp_a.index = test.index
    fit_af = ARIMA(ts.loc[TRAIN_START:TEST_END], order=arima_order).fit()
    fc_a   = fit_af.get_forecast(steps=fc_periods)
    fp_a   = fc_a.predicted_mean; fp_a.index = FC_INDEX
    fi_a   = clamp_ci(fc_a.conf_int()); fi_a.index = FC_INDEX
    results['ARIMA']   = calc_metrics(test, tp_a, f'ARIMA{arima_order}')
    forecasts['ARIMA'] = {'pred': fp_a, 'ci': fi_a, 'test_pred': tp_a, 'order': str(arima_order)}

    # ── SARIMAX+COVID ─────────────────────────
    if SARIMA_FALLBACK[sector_name]:
        s_order, s_seas = SARIMA_FALLBACK[sector_name]
    else:
        sa = pm.auto_arima(train, d=1, D=1, seasonal=True, m=12,
                           max_p=3, max_q=3, max_P=2, max_Q=2, max_D=1,
                           stepwise=True, information_criterion='bic', trace=False)
        s_order, s_seas = sa.order, sa.seasonal_order

    ex_tr = make_exog(train.index)
    ex_te = make_exog(test.index)
    ex_fu = make_exog(ts.loc[TRAIN_START:TEST_END].index)
    ex_fc = make_exog(FC_INDEX)

    fit_s = SARIMAX(train, exog=ex_tr, order=s_order, seasonal_order=s_seas).fit(disp=False)
    tp_s  = fit_s.get_forecast(steps=len(test), exog=ex_te).predicted_mean
    tp_s.index = test.index
    fit_sf = SARIMAX(ts.loc[TRAIN_START:TEST_END], exog=ex_fu, order=s_order, seasonal_order=s_seas).fit(disp=False)
    fc_s   = fit_sf.get_forecast(steps=fc_periods, exog=ex_fc)
    fp_s   = fc_s.predicted_mean; fp_s.index = FC_INDEX
    fi_s   = clamp_ci(fc_s.conf_int()); fi_s.index = FC_INDEX
    pvals  = pd.DataFrame({'coef': fit_s.params, 'p-value': fit_s.pvalues})
    pvals['sig'] = pvals['p-value'].apply(lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns')))

    results['SARIMAX']   = calc_metrics(test, tp_s, 'SARIMAX+COVID')
    forecasts['SARIMAX'] = {'pred': fp_s, 'ci': fi_s, 'test_pred': tp_s,
                             'order': f'{s_order}x{s_seas}', 'pvalues': pvals}

    # ── ETS ───────────────────────────────────
    fit_e = ExponentialSmoothing(train, trend='add', seasonal='add',
                                  seasonal_periods=12, damped_trend=True).fit(optimized=True)
    tp_e  = pd.Series(fit_e.forecast(len(test)), index=test.index)
    fit_ef = ExponentialSmoothing(ts.loc[TRAIN_START:TEST_END], trend='add', seasonal='add',
                                   seasonal_periods=12, damped_trend=True).fit(optimized=True)
    fp_e   = pd.Series(fit_ef.forecast(fc_periods), index=FC_INDEX)
    std_e  = np.std(fit_e.resid)
    fi_e   = pd.DataFrame({'lower': (fp_e - 1.96*std_e*np.sqrt(np.arange(1,fc_periods+1))).clip(0).values,
                            'upper': (fp_e + 1.96*std_e*np.sqrt(np.arange(1,fc_periods+1))).values}, index=FC_INDEX)
    results['ETS']   = calc_metrics(test, tp_e, 'ETS')
    forecasts['ETS'] = {'pred': fp_e, 'ci': fi_e, 'test_pred': tp_e}

    # ── Prophet ───────────────────────────────
    df_pr = pd.DataFrame({'ds': train.index, 'y': train.values})
    m_pr  = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='multiplicative')
    m_pr.fit(df_pr)
    fut_te  = pd.DataFrame({'ds': test.index})
    tp_pr   = pd.Series(m_pr.predict(fut_te)['yhat'].values, index=test.index)
    df_prf  = pd.DataFrame({'ds': ts.loc[TRAIN_START:TEST_END].index, 'y': ts.loc[TRAIN_START:TEST_END].values})
    m_prf   = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                      daily_seasonality=False, seasonality_mode='multiplicative')
    m_prf.fit(df_prf)
    fut_fc  = pd.DataFrame({'ds': FC_INDEX})
    pr_fc   = m_prf.predict(fut_fc)
    fp_pr   = pd.Series(pr_fc['yhat'].values, index=FC_INDEX)
    fi_pr   = pd.DataFrame({'lower': pr_fc['yhat_lower'].clip(0).values,
                             'upper': pr_fc['yhat_upper'].values}, index=FC_INDEX)
    results['Prophet']   = calc_metrics(test, tp_pr, 'Prophet')
    forecasts['Prophet'] = {'pred': fp_pr, 'ci': fi_pr, 'test_pred': tp_pr}

    # ── Hybrid 1: SARIMAX + Ridge ─────────────
    resid_train = train - fit_s.fittedvalues
    scaler1 = StandardScaler()
    lags = 3
    X_tr = np.column_stack([resid_train.shift(i).fillna(0).values for i in range(1, lags+1)])
    X_tr_sc = scaler1.fit_transform(X_tr)
    ridge1  = Ridge(alpha=1.0); ridge1.fit(X_tr_sc, resid_train.values)

    resid_te  = test - tp_s
    X_te = np.column_stack([resid_te.shift(i).fillna(0).values for i in range(1, lags+1)])
    X_te_sc   = scaler1.transform(X_te)
    tp_h1     = tp_s + pd.Series(ridge1.predict(X_te_sc), index=test.index)

    fc_resid_vals = np.zeros(fc_periods)
    for i in range(fc_periods):
        lag_vals = [fc_resid_vals[i-j] if i-j >= 0 else 0 for j in range(1, lags+1)]
        x_new    = scaler1.transform([lag_vals])
        fc_resid_vals[i] = ridge1.predict(x_new)[0]
    fp_h1 = fp_s + pd.Series(fc_resid_vals, index=FC_INDEX)
    ci_half = fi_s.iloc[:,1] - fi_s.iloc[:,0]
    fi_h1 = pd.DataFrame({'lower': (fp_h1 - ci_half/2).clip(0).values,
                           'upper': (fp_h1 + ci_half/2).values}, index=FC_INDEX)
    results['Hybrid1']   = calc_metrics(test, tp_h1, 'Hybrid1(SARIMAX+Ridge)')
    forecasts['Hybrid1'] = {'pred': fp_h1, 'ci': fi_h1, 'test_pred': tp_h1}

    # ── Hybrid 2: ETS + Ridge ─────────────────
    resid_ets = train - fit_e.fittedvalues
    scaler2 = StandardScaler()
    X_tr2 = np.column_stack([resid_ets.shift(i).fillna(0).values for i in range(1, lags+1)])
    X_tr2_sc = scaler2.fit_transform(X_tr2)
    ridge2 = Ridge(alpha=1.0); ridge2.fit(X_tr2_sc, resid_ets.values)

    resid_ets_te = test - tp_e
    X_te2 = np.column_stack([resid_ets_te.shift(i).fillna(0).values for i in range(1, lags+1)])
    X_te2_sc = scaler2.transform(X_te2)
    tp_h2 = tp_e + pd.Series(ridge2.predict(X_te2_sc), index=test.index)

    fc_resid2 = np.zeros(fc_periods)
    for i in range(fc_periods):
        lag_vals2 = [fc_resid2[i-j] if i-j >= 0 else 0 for j in range(1, lags+1)]
        x_new2 = scaler2.transform([lag_vals2])
        fc_resid2[i] = ridge2.predict(x_new2)[0]
    fp_h2 = fp_e + pd.Series(fc_resid2, index=FC_INDEX)
    fi_h2 = pd.DataFrame({'lower': (fp_h2 - ci_half/2).clip(0).values,
                           'upper': (fp_h2 + ci_half/2).values}, index=FC_INDEX)
    results['Hybrid2']   = calc_metrics(test, tp_h2, 'Hybrid2(ETS+Ridge)')
    forecasts['Hybrid2'] = {'pred': fp_h2, 'ci': fi_h2, 'test_pred': tp_h2}

    # ── Hybrid 3: SARIMAX + Prophet ensemble ──
    alpha = 0.5
    tp_h3 = alpha * tp_s + (1-alpha) * tp_pr
    fp_h3 = alpha * fp_s + (1-alpha) * fp_pr
    fi_h3 = pd.DataFrame({'lower': (alpha * fi_s.iloc[:,0].values + (1-alpha) * fi_pr.iloc[:,0].values).clip(0),
                           'upper':  alpha * fi_s.iloc[:,1].values + (1-alpha) * fi_pr.iloc[:,1].values}, index=FC_INDEX)
    results['Hybrid3']   = calc_metrics(test, tp_h3, 'Hybrid3(SARIMAX+Prophet)')
    forecasts['Hybrid3'] = {'pred': fp_h3, 'ci': fi_h3, 'test_pred': tp_h3}

    # Serialize for caching
    return _serialize_results(results, forecasts, FC_INDEX)

def _serialize_results(results, forecasts, FC_INDEX):
    """Convert pandas objects to JSON-safe dicts for st.cache_data"""
    ser_fc = {}
    for k, v in forecasts.items():
        ser_fc[k] = {
            'pred_index' : [str(i) for i in v['pred'].index],
            'pred_values': v['pred'].tolist(),
            'ci_lower'   : v['ci'].iloc[:,0].tolist(),
            'ci_upper'   : v['ci'].iloc[:,1].tolist(),
            'test_pred_index' : [str(i) for i in v['test_pred'].index],
            'test_pred_values': v['test_pred'].tolist(),
        }
        if 'pvalues' in v:
            ser_fc[k]['pvalues_index'] = v['pvalues'].index.tolist()
            ser_fc[k]['pvalues_coef']  = v['pvalues']['coef'].tolist()
            ser_fc[k]['pvalues_p']     = v['pvalues']['p-value'].tolist()
            ser_fc[k]['pvalues_sig']   = v['pvalues']['sig'].tolist()
    return results, ser_fc

def deserialize_forecasts(ser_fc):
    out = {}
    for k, v in ser_fc.items():
        pred = pd.Series(v['pred_values'],
                         index=pd.DatetimeIndex(v['pred_index']))
        ci   = pd.DataFrame({'lower': v['ci_lower'], 'upper': v['ci_upper']},
                            index=pd.DatetimeIndex(v['pred_index']))
        tp   = pd.Series(v['test_pred_values'],
                         index=pd.DatetimeIndex(v['test_pred_index']))
        out[k] = {'pred': pred, 'ci': ci, 'test_pred': tp}
        if 'pvalues_p' in v:
            out[k]['pvalues'] = pd.DataFrame({
                'coef'   : v['pvalues_coef'],
                'p-value': v['pvalues_p'],
                'sig'    : v['pvalues_sig'],
            }, index=v['pvalues_index'])
    return out

# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
def build_forecast_chart(ts, train, test, forecasts, selected_models, fc_periods, sector_name):
    FC_INDEX = pd.date_range('2026-01-01', periods=fc_periods, freq='MS')
    fig = go.Figure()

    # Background shading
    for start, end, color, label in [
        (train.index[0], train.index[-1], 'rgba(46,134,193,0.06)', 'Train 2010–2023'),
        (test.index[0],  test.index[-1],  'rgba(230,126,34,0.10)', 'Test 2024–2025'),
        (FC_INDEX[0],    FC_INDEX[-1],    'rgba(0,200,150,0.08)',  'Forecast'),
        (pd.Timestamp(COVID_START), pd.Timestamp(COVID_END), 'rgba(241,148,138,0.18)', 'COVID-19'),
    ]:
        fig.add_vrect(x0=str(start), x1=str(end), fillcolor=color,
                      layer='below', line_width=0,
                      annotation_text=label, annotation_position='top left',
                      annotation_font_size=9, annotation_font_color='#6b7a8d')

    # Actual line
    fig.add_trace(go.Scatter(
        x=ts.loc[TRAIN_START:TEST_END].index, y=ts.loc[TRAIN_START:TEST_END].values,
        name='Actual', line=dict(color=COLORS['actual'], width=2.5),
        mode='lines', hovertemplate='%{x|%Y-%m}<br>Actual: %{y:,.0f}<extra></extra>'
    ))

    # Models
    for model_key in selected_models:
        if model_key not in forecasts: continue
        fc  = forecasts[model_key]
        col = COLORS.get(model_key, '#888')

        # Test prediction
        fig.add_trace(go.Scatter(
            x=fc['test_pred'].index, y=fc['test_pred'].values,
            name=f'{model_key} (test)', line=dict(color=col, width=1.6, dash='dot'),
            opacity=0.85, hovertemplate=f'%{{x|%Y-%m}}<br>{model_key} test: %{{y:,.0f}}<extra></extra>'
        ))
        # Forecast
        fig.add_trace(go.Scatter(
            x=fc['pred'].index, y=fc['pred'].values,
            name=f'{model_key} forecast', line=dict(color=col, width=2.2),
            mode='lines+markers', marker=dict(size=4),
            hovertemplate=f'%{{x|%Y-%m}}<br>{model_key}: %{{y:,.0f}}<extra></extra>'
        ))
        # CI band
        fig.add_trace(go.Scatter(
            x=list(fc['ci'].index) + list(fc['ci'].index[::-1]),
            y=list(fc['ci'].iloc[:,1]) + list(fc['ci'].iloc[:,0][::-1]),
            fill='toself', fillcolor=col.replace(')', ',0.1)').replace('rgb', 'rgba') if col.startswith('rgb') else col + '22',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{model_key} 95% CI', showlegend=False,
            hoverinfo='skip'
        ))

    # Divider lines
    fig.add_vline(x=str(test.index[0]), line=dict(color='#aaa', dash='dash', width=1))
    fig.add_vline(x=str(FC_INDEX[0]),   line=dict(color='#aaa', dash='dash', width=1))

    fig.update_layout(
        title=dict(text=f'<b>{sector_name} Sector — CO₂ Emission Forecast</b>',
                   font=dict(size=16, color='#0d2137')),
        xaxis=dict(title='Date', showgrid=True, gridcolor='#f0f0f0', tickformat='%Y'),
        yaxis=dict(title='CO₂ Emission (1,000 Tons)', showgrid=True, gridcolor='#f0f0f0',
                   tickformat=',.0f'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=10)),
        height=420,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=70, b=50),
    )
    return fig

def build_metrics_radar(results_list):
    models = [r['Model'] for r in results_list]
    # Normalize each metric 0-1 (lower is better → invert)
    mape_vals = np.array([r['MAPE'] for r in results_list])
    mae_vals  = np.array([r['MAE']  for r in results_list])
    rmse_vals = np.array([r['RMSE'] for r in results_list])

    def norm_inv(arr):
        return 1 - (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    n_mape = norm_inv(mape_vals)
    n_mae  = norm_inv(mae_vals)
    n_rmse = norm_inv(rmse_vals)

    fig = go.Figure()
    cats = ['MAPE', 'MAE', 'RMSE', 'MAPE']  # close polygon

    model_keys = ['ARIMA','SARIMAX','ETS','Prophet','Hybrid1','Hybrid2','Hybrid3']
    for i, r in enumerate(results_list):
        key = [k for k in model_keys if k in r['Model']]
        key = key[0] if key else 'ARIMA'
        vals = [n_mape[i], n_mae[i], n_rmse[i], n_mape[i]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats, name=r['Model'],
                                       line=dict(color=COLORS.get(key,'#888'), width=1.8),
                                       fill='toself', opacity=0.4))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=8))),
        showlegend=True,
        height=340,
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(font=dict(size=9)),
        paper_bgcolor='white',
    )
    return fig

def build_mape_bar(results_all):
    """Bar chart comparing MAPE across all models and sectors"""
    rows = []
    for sector, results in results_all.items():
        for r in results:
            rows.append({'Sector': sector, 'Model': r['Model'], 'MAPE': r['MAPE']})
    df = pd.DataFrame(rows)

    fig = px.bar(df, x='Model', y='MAPE', color='Sector', barmode='group',
                 color_discrete_map=COLORS,
                 labels={'MAPE': 'MAPE (%)', 'Model': ''},
                 title='<b>Model MAPE Comparison — All Sectors</b>')
    fig.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(orientation='h', y=1.05),
                      yaxis=dict(gridcolor='#f0f0f0'),
                      margin=dict(l=50,r=20,t=60,b=100))
    fig.update_xaxes(tickangle=-35)
    return fig

def build_pvalue_chart(pvalues, sector_name):
    params = pvalues.index.tolist()
    pvals  = pvalues['p-value'].tolist()
    cols   = ['#C0392B' if p < 0.05 else '#AEB6BF' for p in pvals]
    log_p  = [-np.log10(p + 1e-10) for p in pvals]

    fig = go.Figure(go.Bar(
        y=params, x=log_p, orientation='h',
        marker_color=cols,
        text=[f'p={p:.3f} {s}' for p, s in zip(pvals, pvalues['sig'].tolist())],
        textposition='outside',
    ))
    fig.add_vline(x=-np.log10(0.05), line=dict(color='#E74C3C', dash='dash'),
                  annotation_text='p=0.05', annotation_font_size=9)
    fig.update_layout(
        title=f'<b>SARIMAX P-values — {sector_name}</b>',
        xaxis_title='-log₁₀(p-value) — higher = more significant',
        height=max(250, len(params)*35),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=120, r=150, t=50, b=30),
        yaxis=dict(autorange='reversed'),
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size: 2.5rem;'>🌿</div>
        <div style='font-size: 1.1rem; font-weight:700; color:#00c896;'>CO₂ Forecast</div>
        <div style='font-size: 0.75rem; color:#7ab5d0;'>Thailand Energy Sectors</div>
    </div>
    <hr style='border-color:#1e3a5c; margin: 0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Upload Data**")
    uploaded_file = st.file_uploader(
        "Upload EPPO Excel file",
        type=['xlsx'],
        help="File: Eppo_Out_CO2_from_Power__Transport__Industry_Dataset.xlsx"
    )

    st.markdown("---")
    st.markdown("**🔧 Forecast Settings**")

    fc_months = st.slider(
        "Forecast Months", min_value=6, max_value=36,
        value=24, step=6,
        help="Number of months to forecast beyond 2025-12"
    )

    st.markdown("**📊 Models to Display**")
    all_model_keys = list(MODEL_INFO.keys())
    selected_models = []
    for k, label in MODEL_INFO.items():
        short = label.split(' — ')[0]
        if st.checkbox(short, value=True, key=f'cb_{k}'):
            selected_models.append(k)

    st.markdown("---")
    st.markdown("**📌 Best Models**")
    for sec, best in SECTOR_BEST.items():
        col = COLORS[sec]
        st.markdown(f"<span style='color:{col};font-weight:700;'>●</span> **{sec}**: {best}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#5a8aaa; text-align:center;'>
    Train: 2010–2023<br>Test: 2024–2025<br>
    COVID dummy: Apr 2020 – Jun 2021<br><br>
    <b>7 Models</b> | <b>3 Sectors</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 CO₂ Emission Forecasting — Thailand Energy Sectors</h1>
    <p>Time Series Analysis with ARIMA · SARIMAX · ETS · Prophet · Hybrid Models &nbsp;|&nbsp;
       Power · Transport · Industry &nbsp;|&nbsp; 2010–2027</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING GATE
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Please upload the EPPO Excel file to begin analysis.")
    st.markdown("""
    ### Expected File Format
    The Excel file should contain three sheets:
    - **eppo-power-dataset** — Power generation CO₂ (oil, coal, gas, total)
    - **eppo-transport-dataset** — Transport CO₂ (oil, gas, total)
    - **eppo-industry-dataset** — Industry CO₂ (oil, coal, gas, total)

    Data range: Monthly, ~1990–2025 (analysis uses 2010–2025)
    """)
    st.stop()

# ─────────────────────────────────────────────
# PARSE DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    buf = io.BytesIO(file_bytes)
    power_df  = parse_eppo_sheet(buf, 'eppo-power-dataset',    ['oil','coal','gas','total'])
    buf.seek(0)
    transp_df = parse_eppo_sheet(buf, 'eppo-transport-dataset', ['oil','gas','total'])
    buf.seek(0)
    indust_df = parse_eppo_sheet(buf, 'eppo-industry-dataset',  ['oil','coal','gas','total'])
    return {'Power': power_df, 'Transport': transp_df, 'Industry': indust_df}

with st.spinner("📂 Loading EPPO data..."):
    SECTORS = load_data(uploaded_file.read())

# Quick data summary
col1, col2, col3 = st.columns(3)
for col, (name, df) in zip([col1, col2, col3], SECTORS.items()):
    ts = df['total']
    col.metric(f"{name} Sector", f"{ts.loc['2010':].mean():,.0f} KT avg",
               f"{ts.index[0].strftime('%Y-%m')} → {ts.index[-1].strftime('%Y-%m')}")

st.markdown("---")

# ─────────────────────────────────────────────
# FIT ALL MODELS  (with progress)
# ─────────────────────────────────────────────
ALL_RESULTS   = {}
ALL_FORECASTS = {}

progress_bar = st.progress(0, text="⏳ Fitting models...")
status_ph    = st.empty()

for i, (sector_name, df) in enumerate(SECTORS.items()):
    status_ph.info(f"🔄 Fitting models for **{sector_name}** sector...")
    ts, train, test = split(df)

    train_ser = {'index': [str(x) for x in train.index], 'values': train.tolist()}
    test_ser  = {'index': [str(x) for x in test.index],  'values': test.tolist()}
    ts_ser    = {'index': [str(x) for x in ts.index],    'values': ts.tolist()}

    raw_results, ser_fc = fit_all_models(sector_name, train_ser, test_ser, ts_ser, fc_months)
    ALL_RESULTS[sector_name]   = list(raw_results.values())
    ALL_FORECASTS[sector_name] = deserialize_forecasts(ser_fc)
    progress_bar.progress((i+1)/3)

progress_bar.empty()
status_ph.empty()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_overview, tab_forecast, tab_metrics, tab_pvalue, tab_data = st.tabs([
    "📊 Overview", "🔮 Forecast — All Sectors", "📈 Performance Metrics", "🔬 P-value Analysis", "📋 Data Table"
])

# ═══════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════
with tab_overview:
    st.markdown('<div class="section-title">📊 CO₂ Emission Overview — 3 Sectors (2010–2025)</div>', unsafe_allow_html=True)

    fig_ov = make_subplots(rows=1, cols=3,
                            subplot_titles=[f'<b>{s}</b>' for s in SECTORS.keys()],
                            shared_yaxes=False)
    for col_i, (name, df) in enumerate(SECTORS.items(), 1):
        ts = df['total'].loc['2010':]
        fuel_cols = [c for c in df.columns if c != 'total']
        fig_ov.add_trace(go.Scatter(x=ts.index, y=ts.values, name=f'{name} Total',
                                     line=dict(color=COLORS[name], width=2.5), showlegend=True), row=1, col=col_i)
        for fc in fuel_cols:
            fig_ov.add_trace(go.Scatter(x=df.loc['2010':].index, y=df.loc['2010':, fc].values,
                                         name=fc.capitalize(), line=dict(width=1.2, dash='dot'), opacity=0.6,
                                         showlegend=(col_i==1)), row=1, col=col_i)

    fig_ov.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                          legend=dict(orientation='h', y=-0.15),
                          margin=dict(l=50, r=20, t=60, b=50))
    fig_ov.update_xaxes(tickformat='%Y', showgrid=True, gridcolor='#f0f0f0')
    fig_ov.update_yaxes(tickformat=',.0f', showgrid=True, gridcolor='#f0f0f0', title_text='1,000 Tons')
    st.plotly_chart(fig_ov, use_container_width=True)

    # Key stats
    st.markdown('<div class="section-title">📌 Key Statistics (2010–2025)</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, df) in zip(cols, SECTORS.items()):
        ts = df['total'].loc['2010':'2025']
        trend = (ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0] * 100
        with col:
            st.markdown(f"""
            <div class="sector-card" style="--sector-color:{COLORS[name]}">
                <b style="color:{COLORS[name]}">⚡ {name} Sector</b>
                <div class="metric-row" style="margin-top:0.6rem">
                    <div class="metric-box"><div class="label">Mean</div><div class="value">{ts.mean():,.0f}</div><div class="unit">KT</div></div>
                    <div class="metric-box"><div class="label">Peak</div><div class="value">{ts.max():,.0f}</div><div class="unit">KT</div></div>
                    <div class="metric-box"><div class="label">Trend</div><div class="value">{trend:+.1f}%</div><div class="unit">total</div></div>
                </div>
                <div style="font-size:0.8rem;color:#566573">
                Best model: <b>{SECTOR_BEST[name]}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════
# TAB 2: FORECAST — ALL 3 SECTORS
# ═══════════════════════════
with tab_forecast:
    st.markdown('<div class="section-title">🔮 Forecast — All 3 Sectors on One Page</div>', unsafe_allow_html=True)

    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
    else:
        for sector_name, df in SECTORS.items():
            ts, train, test = split(df)
            scolor = COLORS[sector_name]
            best   = SECTOR_BEST[sector_name]
            best_res = next((r for r in ALL_RESULTS[sector_name] if best in r['Model']), {})

            # Sector header
            mape_val = best_res.get('MAPE', 0)
            mae_val  = best_res.get('MAE', 0)
            rmse_val = best_res.get('RMSE', 0)

            st.markdown(f"""
            <div class="sector-card" style="--sector-color:{scolor}">
                <b style="color:{scolor}; font-size:1.05rem;">
                    ⚡ {sector_name} Sector
                </b>
                <span class="best-badge">Best: {best}</span>
                <div class="metric-row" style="margin-top:0.6rem">
                    <div class="metric-box"><div class="label">MAPE</div><div class="value">{mape_val:.2f}%</div></div>
                    <div class="metric-box"><div class="label">MAE</div><div class="value">{mae_val:,.0f}</div></div>
                    <div class="metric-box"><div class="label">RMSE</div><div class="value">{rmse_val:,.0f}</div></div>
                    <div class="metric-box"><div class="label">FC Months</div><div class="value">{fc_months}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig = build_forecast_chart(ts, train, test, ALL_FORECASTS[sector_name],
                                        selected_models, fc_months, sector_name)
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table (best model)
            with st.expander(f"📋 {sector_name} Forecast Values ({best})"):
                fc_df = ALL_FORECASTS[sector_name][best]
                fc_table = pd.DataFrame({
                    'Date'     : [d.strftime('%Y-%m') for d in fc_df['pred'].index],
                    'Forecast' : fc_df['pred'].round(1).values,
                    'CI Lower' : fc_df['ci'].iloc[:,0].round(1).values,
                    'CI Upper' : fc_df['ci'].iloc[:,1].round(1).values,
                })
                st.dataframe(fc_table, hide_index=True, use_container_width=True)

            st.markdown("<hr style='border-color:#e2ecf9; margin: 0.5rem 0;'>", unsafe_allow_html=True)

# ═══════════════════════════
# TAB 3: PERFORMANCE METRICS
# ═══════════════════════════
with tab_metrics:
    st.markdown('<div class="section-title">📈 Model Performance — Test Period (2024–2025)</div>', unsafe_allow_html=True)

    # MAPE comparison bar
    st.plotly_chart(build_mape_bar(ALL_RESULTS), use_container_width=True)

    # Metrics table + radar per sector
    cols = st.columns(3)
    for col, (sector_name, results) in zip(cols, ALL_RESULTS.items()):
        with col:
            st.markdown(f"**{sector_name} Sector**")
            df_r = pd.DataFrame(results)[['Model','MAE','RMSE','MAPE']].round(2)
            best = SECTOR_BEST[sector_name]
            # Highlight best
            def highlight_best(row):
                return ['background-color: #d5f5e3; font-weight:bold'
                        if best in row['Model'] else '' for _ in row]
            st.dataframe(df_r.style.apply(highlight_best, axis=1), hide_index=True, use_container_width=True)
            st.plotly_chart(build_metrics_radar(results), use_container_width=True)

    # Summary best models
    st.markdown('<div class="section-title">🏆 Best Model Summary</div>', unsafe_allow_html=True)
    summary_rows = []
    for sector_name, results in ALL_RESULTS.items():
        best_key = SECTOR_BEST[sector_name]
        best_res = next((r for r in results if best_key in r['Model']), {})
        summary_rows.append({
            'Sector'     : sector_name,
            'Best Model' : best_res.get('Model', ''),
            'MAE'        : f"{best_res.get('MAE',0):,.1f}",
            'RMSE'       : f"{best_res.get('RMSE',0):,.1f}",
            'MAPE (%)'   : f"{best_res.get('MAPE',0):.2f}%",
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# ═══════════════════════════
# TAB 4: P-VALUE ANALYSIS
# ═══════════════════════════
with tab_pvalue:
    st.markdown('<div class="section-title">🔬 P-value Analysis — SARIMAX Coefficients</div>', unsafe_allow_html=True)
    st.markdown("""
    P-values indicate statistical significance of each coefficient:
    - **p < 0.001** → *** highly significant
    - **p < 0.01**  → ** significant
    - **p < 0.05**  → * significant
    - **p ≥ 0.05**  → ns (not significant)
    """)

    for sector_name in SECTORS:
        fc = ALL_FORECASTS[sector_name].get('SARIMAX', {})
        if 'pvalues' in fc:
            st.plotly_chart(build_pvalue_chart(fc['pvalues'], sector_name), use_container_width=True)
        else:
            st.info(f"{sector_name}: P-value data not available.")

# ═══════════════════════════
# TAB 5: DATA TABLE
# ═══════════════════════════
with tab_data:
    st.markdown('<div class="section-title">📋 Raw Data & Forecast Export</div>', unsafe_allow_html=True)

    sec_sel = st.selectbox("Select Sector", list(SECTORS.keys()))
    df_raw  = SECTORS[sec_sel]
    ts, train, test = split(df_raw)
    fc_best = ALL_FORECASTS[sec_sel][SECTOR_BEST[sec_sel]]
    FC_INDEX = pd.date_range('2026-01-01', periods=fc_months, freq='MS')

    # Build combined table
    rows = []
    for date, val in ts.items():
        is_test = date in test.index
        rows.append({
            'Date'  : date.strftime('%Y-%m'),
            'Period': 'Test' if is_test else 'Train',
            'Actual': round(val, 1),
            'Forecast': '',
            'CI_Lower': '',
            'CI_Upper': '',
        })
    for i, (date, val) in enumerate(fc_best['pred'].items()):
        rows.append({
            'Date'  : date.strftime('%Y-%m'),
            'Period': 'Forecast',
            'Actual': '',
            'Forecast': round(val, 1),
            'CI_Lower': round(fc_best['ci'].iloc[i,0], 1),
            'CI_Upper': round(fc_best['ci'].iloc[i,1], 1),
        })

    df_export = pd.DataFrame(rows)
    st.dataframe(df_export, use_container_width=True, height=400)

    # Download button
    csv_bytes = df_export.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label=f"⬇️ Download {sec_sel} Forecast CSV",
        data=csv_bytes,
        file_name=f"CO2_Forecast_{sec_sel}_{SECTOR_BEST[sec_sel]}.csv",
        mime='text/csv',
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CO₂ Emission Forecasting System &nbsp;|&nbsp;
    Data Source: EPPO Thailand &nbsp;|&nbsp;
    Models: ARIMA · SARIMAX+COVID · ETS · Prophet · Hybrid1 · Hybrid2 · Hybrid3 &nbsp;|&nbsp;
    Train 2010–2023 · Test 2024–2025 · Forecast 2026–2027
</div>
""", unsafe_allow_html=True)
