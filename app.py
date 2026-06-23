# -*- coding: utf-8 -*-
"""
app.py — CO2 Emission Forecast Dashboard
=========================================
เว็บแอปพยากรณ์ CO2 จาก 3 sectors (Power, Transport, Industry)
รองรับ 7 โมเดล: ARIMA, SARIMAX(+COVID), ETS, Prophet, Hybrid1(SARIMAX+Ridge),
Hybrid2(ETS+Ridge), Hybrid3(SARIMAX+Prophet)

วิธีรัน:  streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_utils import (
    list_sheets, guess_sheet_for_sector, parse_eppo_sheet,
    parse_csv_tidy, validate_sector_df,
)
from model_utils import (
    run_all_models, HAS_PROPHET, covid_window_in_range,
)

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO2 Emission Forecast Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Design tokens — สีหลักให้คงเส้นคงวากับชุดสีในงานวิเคราะห์ต้นฉบับ
# โทนรายงานวิชาการ: พื้นอ่อนสะอาด ตัวอักษรเข้มอ่านง่ายบนโปรเจคเตอร์
# ──────────────────────────────────────────────────────────────────────────
COLORS = {
    'bg':        '#F7F9F8',
    'surface':   '#FFFFFF',
    'ink':       '#13241D',
    'muted':     '#5B6B63',
    'line':      '#DDE6E1',
    'accent':    '#0E6E4E',   # เขียวหลัก — CO2 / สิ่งแวดล้อม
    'accent2':   '#0B4F8A',   # น้ำเงิน — ตัดกับเขียว
    'warn':      '#C0392B',

    'actual':    '#13241D',
    'arima':     '#C0392B',
    'sarimax':   '#0E6E4E',
    'ets':       '#0B4F8A',
    'prophet':   '#7D3C98',
    'hybrid1':   '#D4AC0D',
    'hybrid2':   '#D35400',
    'hybrid3':   '#17A589',

    'power':     '#0B4F8A',
    'transport': '#D35400',
    'industry':  '#0E6E4E',

    'test_bg':   'rgba(212,172,13,0.08)',
    'fc_bg':     'rgba(14,110,78,0.07)',
}

MODEL_COLOR_KEY = {
    'ARIMA': 'arima', 'SARIMAX': 'sarimax', 'ETS': 'ets', 'Prophet': 'prophet',
    'Hybrid1': 'hybrid1', 'Hybrid2': 'hybrid2', 'Hybrid3': 'hybrid3',
}
MODEL_DISPLAY_NAME = {
    'ARIMA': 'ARIMA', 'SARIMAX': 'SARIMAX (+COVID)', 'ETS': 'ETS (Holt-Winters)',
    'Prophet': 'Prophet', 'Hybrid1': 'Hybrid 1 (SARIMAX+Ridge)',
    'Hybrid2': 'Hybrid 2 (ETS+Ridge)', 'Hybrid3': 'Hybrid 3 (SARIMAX+Prophet)',
}
ALL_MODEL_KEYS = ['ARIMA', 'SARIMAX', 'ETS', 'Prophet', 'Hybrid1', 'Hybrid2', 'Hybrid3']

SECTOR_UNIT = "พันตัน CO₂ (1,000 Tons)"


def inject_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans Thai', -apple-system, sans-serif;
        }}
        .stApp {{
            background-color: {COLORS['bg']};
        }}
        .block-container {{
            padding-top: 1.6rem;
            max-width: 1320px;
        }}
        h1, h2, h3 {{
            color: {COLORS['ink']};
            font-weight: 700;
        }}
        .app-title {{
            font-size: 2.05rem;
            font-weight: 700;
            color: {COLORS['ink']};
            margin-bottom: 0.1rem;
            letter-spacing: -0.01em;
        }}
        .app-subtitle {{
            font-size: 1.0rem;
            color: {COLORS['muted']};
            margin-bottom: 1.1rem;
        }}
        .sector-band {{
            border-radius: 14px;
            padding: 1.35rem 1.5rem 0.4rem 1.5rem;
            margin: 1.6rem 0 1.0rem 0;
            border: 1px solid {COLORS['line']};
            background: {COLORS['surface']};
            border-left: 6px solid var(--band-color, {COLORS['accent']});
        }}
        .sector-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {COLORS['ink']};
            margin-bottom: 0.15rem;
        }}
        .sector-caption {{
            font-size: 0.88rem;
            color: {COLORS['muted']};
            margin-bottom: 0.6rem;
        }}
        div[data-testid="stMetric"] {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 12px;
            padding: 0.75rem 1rem 0.55rem 1rem;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {COLORS['muted']};
            font-weight: 500;
        }}
        div[data-testid="stMetricValue"] {{
            color: {COLORS['ink']};
            font-family: 'IBM Plex Mono', monospace;
        }}
        .badge {{
            display: inline-block;
            font-size: 0.74rem;
            font-weight: 600;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            margin-right: 0.35rem;
        }}
        .badge-best {{ background: rgba(14,110,78,0.12); color: {COLORS['accent']}; }}
        .badge-info {{ background: rgba(11,79,138,0.10); color: {COLORS['accent2']}; }}
        .footer-note {{
            color: {COLORS['muted']};
            font-size: 0.82rem;
            text-align: center;
            margin-top: 2.4rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS['line']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: #FBFCFB;
        }}
    </style>
    """, unsafe_allow_html=True)


def sector_band_open(name, sector_color, caption):
    st.markdown(
        f"""<div class="sector-band" style="--band-color:{sector_color};">
        <div class="sector-title">{name}</div>
        <div class="sector-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def sector_band_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Cached helpers
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_list_sheets(file_bytes):
    return list_sheets(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_parse_eppo(file_bytes, sheet_name):
    df, fuel_cols, total_col = parse_eppo_sheet(io.BytesIO(file_bytes), sheet_name)
    return df, fuel_cols, total_col


@st.cache_data(show_spinner=False)
def cached_parse_csv(file_bytes):
    return parse_csv_tidy(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_run_models(data_hash, df_records, total_col, fuel_cols, test_months, max_horizon,
                       selected_models_tuple, order_mode, manual_arima_order, manual_sarima_order,
                       manual_sarima_seasonal, use_covid):
    """
    cache key อิงจาก data_hash + พารามิเตอร์ทั้งหมด — เปลี่ยนพารามิเตอร์ใดก็ตาม cache จะ miss และรันใหม่
    df_records: list of (date_iso, {col: val, ...}) เพื่อให้ hashable/cacheable
    """
    dates = [pd.Timestamp(d) for d, _ in df_records]
    cols = list(df_records[0][1].keys())
    data = {c: [rec[1][c] for rec in df_records] for c in cols}
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name='date'))

    selected_models = list(selected_models_tuple)
    out = run_all_models(
        df, total_col, list(fuel_cols), test_months, max_horizon, selected_models,
        order_mode=order_mode, manual_arima_order=manual_arima_order,
        manual_sarima_order=manual_sarima_order, manual_sarima_seasonal=manual_sarima_seasonal,
        use_covid=use_covid, status_cb=None,
    )
    return out


def df_to_records(df):
    return [(d.isoformat(), row.to_dict()) for d, row in df.iterrows()]


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — การตั้งค่าทั้งหมด
# ──────────────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("### 📂 ข้อมูลนำเข้า")
    file_format = st.sidebar.radio(
        "รูปแบบไฟล์",
        ["Excel (EPPO multi-sheet)", "CSV (tidy format)"],
        help="EPPO multi-sheet = ไฟล์ต้นฉบับที่มี 3 ชีตแยกตาม sector (เหมือนไฟล์ที่ใช้ใน notebook)\n"
             "CSV tidy = ไฟล์เดียวมีคอลัมน์ date, sector, total (และเชื้อเพลิงอื่นๆ ถ้ามี)",
    )
    uploaded = st.sidebar.file_uploader(
        "อัปโหลดไฟล์ข้อมูล CO₂",
        type=["xlsx", "xls", "csv"],
        help="ไฟล์ Excel ต้นฉบับ: Eppo_Out_CO2_from_Power_Transport_Industry_Dataset.xlsx",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ การพยากรณ์")
    max_horizon = st.sidebar.slider(
        "จำนวนเดือนที่ต้องการพยากรณ์ล่วงหน้า", min_value=1, max_value=36, value=24, step=1,
        help="โมเดลจะถูก fit ด้วย horizon สูงสุด 36 เดือนครั้งเดียว แล้วเลื่อนแสดงผลตามจำนวนที่เลือกได้ทันที",
    )
    test_months = st.sidebar.slider(
        "จำนวนเดือนสำหรับ Test set (วัดความแม่นยำ)", min_value=6, max_value=36, value=24, step=1,
        help="ใช้ข้อมูลจริงช่วงท้ายสุด N เดือนเป็น Test set เพื่อคำนวณ MAE/RMSE/MAPE "
             "(เดือนก่อนหน้านั้นทั้งหมดเป็น Train)",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 โมเดลที่ใช้เปรียบเทียบ")
    available_models = list(ALL_MODEL_KEYS)
    if not HAS_PROPHET:
        available_models = [m for m in available_models if m not in ('Prophet', 'Hybrid3')]
        st.sidebar.caption("⚠️ ไม่พบไลบรารี `prophet` ในระบบ — ปิดใช้งาน Prophet และ Hybrid3 อัตโนมัติ")
    selected_models = st.sidebar.multiselect(
        "เลือกโมเดล", options=available_models,
        default=available_models,
        format_func=lambda k: MODEL_DISPLAY_NAME.get(k, k),
    )

    use_covid = st.sidebar.checkbox(
        "ใส่ COVID dummy ใน SARIMAX (เม.ย.2020–มิ.ย.2021)", value=True,
        help="ใส่ตัวแปรหุ่น (exogenous dummy) ช่วงโควิดในโมเดล SARIMAX/Hybrid1 "
             "ระบบจะปิดอัตโนมัติถ้าข้อมูลที่อัปโหลดไม่คาบเกี่ยวช่วงเวลานี้เลย",
    )

    with st.sidebar.expander("⚙️ ตั้งค่าขั้นสูง — Order ของ ARIMA/SARIMA"):
        order_mode = st.radio(
            "วิธีเลือก order",
            ["fixed", "auto"],
            format_func=lambda x: "ใช้ค่าคงที่ (เร็ว, แนะนำสำหรับขึ้นพรีเซนต์สด)"
            if x == "fixed" else "ค้นหาอัตโนมัติด้วย BIC (ช้ากว่า, แม่นยำกว่า)",
        )
        col1, col2 = st.columns(2)
        with col1:
            ar_p = st.number_input("ARIMA p", 0, 5, 1)
            ar_d = st.number_input("ARIMA d", 0, 2, 1)
            ar_q = st.number_input("ARIMA q", 0, 5, 1)
        with col2:
            sp = st.number_input("SARIMA P (seasonal)", 0, 3, 1)
            sd = st.number_input("SARIMA D (seasonal)", 0, 2, 1)
            sq = st.number_input("SARIMA Q (seasonal)", 0, 3, 1)
        manual_arima_order = (int(ar_p), int(ar_d), int(ar_q))
        manual_sarima_order = (1, 1, 1)
        manual_sarima_seasonal = (int(sp), int(sd), int(sq), 12)
        st.caption(
            "หมายเหตุ: เวอร์ชันนี้ไม่ใช้ไลบรารี `pmdarima` (เพื่อความเสถียรตอน deploy บนคลาวด์) "
            "การค้นหาอัตโนมัติใช้ BIC grid-search ด้วย statsmodels ล้วนๆ แทน — ผลลัพธ์เทียบเคียงได้ "
            "แต่ใช้เวลานานกว่า auto_arima เล็กน้อย"
        )

    return {
        'file_format': file_format, 'uploaded': uploaded, 'max_horizon': max_horizon,
        'test_months': test_months, 'selected_models': selected_models, 'use_covid': use_covid,
        'order_mode': order_mode, 'manual_arima_order': manual_arima_order,
        'manual_sarima_order': manual_sarima_order, 'manual_sarima_seasonal': manual_sarima_seasonal,
    }


# ──────────────────────────────────────────────────────────────────────────
# โหลด + ตรวจสอบข้อมูล (ขั้นตอนยืนยันก่อนรันโมเดล เพื่อให้ตรงกับไฟล์จริงของผู้ใช้)
# ──────────────────────────────────────────────────────────────────────────

def load_sectors_from_excel(file_bytes):
    sheets = cached_list_sheets(file_bytes)
    guesses = guess_sheet_for_sector(sheets)

    st.markdown("#### 🔎 ตรวจสอบการจับคู่ชีตข้อมูลกับแต่ละ Sector")
    st.caption("ระบบจับคู่ชีตให้อัตโนมัติจากชื่อชีต — กรุณาตรวจสอบ/แก้ไขให้ตรงกับไฟล์จริงก่อนรันโมเดล")
    cols = st.columns(3)
    chosen_sheets = {}
    sector_names = ['Power', 'Transport', 'Industry']
    options = ["(ไม่ใช้)"] + sheets
    for col, sector in zip(cols, sector_names):
        default_sheet = guesses.get(sector)
        default_idx = options.index(default_sheet) if default_sheet in options else 0
        with col:
            chosen = st.selectbox(f"ชีตของ {sector}", options, index=default_idx, key=f"sheet_{sector}")
            chosen_sheets[sector] = None if chosen == "(ไม่ใช้)" else chosen

    sectors = {}
    parse_errors = {}
    for sector, sheet in chosen_sheets.items():
        if sheet is None:
            continue
        try:
            df, fuel_cols, total_col = cached_parse_eppo(file_bytes, sheet)
            sectors[sector] = (df, fuel_cols, total_col)
        except Exception as e:
            parse_errors[sector] = str(e)

    for sector, err in parse_errors.items():
        st.error(f"❌ {sector}: {err}")

    return sectors


def load_sectors_from_csv(file_bytes):
    raw_sectors = cached_parse_csv(file_bytes)
    st.markdown("#### 🔎 ตรวจสอบการจับคู่ค่า sector ในไฟล์ CSV")
    sector_names = ['Power', 'Transport', 'Industry']
    options = ["(ไม่ใช้)"] + list(raw_sectors.keys())
    cols = st.columns(3)
    sectors = {}
    for col, sector in zip(cols, sector_names):
        # เดาแบบง่าย: จับคู่ตาม keyword ในชื่อ key ที่มีอยู่จริง
        guess = next((k for k in raw_sectors if sector.lower() in str(k).lower()), None)
        default_idx = options.index(guess) if guess in options else 0
        with col:
            chosen = st.selectbox(f"ค่า sector ของ {sector}", options, index=default_idx, key=f"csvsec_{sector}")
        if chosen != "(ไม่ใช้)":
            df = raw_sectors[chosen]
            total_col = 'total' if 'total' in df.columns else df.columns[-1]
            fuel_cols = [c for c in df.columns if c != total_col]
            sectors[sector] = (df, fuel_cols, total_col)
    return sectors


def render_data_preview(sectors):
    st.markdown("#### 📋 พรีวิวข้อมูลที่อ่านได้ (ตรวจสอบให้ตรงกับไฟล์จริงก่อนพยากรณ์)")
    tabs = st.tabs(list(sectors.keys()))
    all_ok = True
    for tab, (sector, (df, fuel_cols, total_col)) in zip(tabs, sectors.items()):
        with tab:
            issues = validate_sector_df(df)
            c1, c2, c3 = st.columns(3)
            c1.metric("ช่วงข้อมูล", f"{df.index.min():%Y-%m} → {df.index.max():%Y-%m}")
            c2.metric("จำนวนเดือน", f"{len(df)}")
            c3.metric("คอลัมน์เชื้อเพลิง", ", ".join(fuel_cols) if fuel_cols else "—")
            if issues:
                all_ok = False
                for msg in issues:
                    st.warning(f"⚠️ {msg}")
            st.dataframe(df.tail(12), use_container_width=True, height=260)
    return all_ok


# ──────────────────────────────────────────────────────────────────────────
# กราฟ — Plotly แบบ interactive (ซูม/hover ได้ เหมาะกับโชว์สดบนเวที)
# ──────────────────────────────────────────────────────────────────────────

def build_forecast_chart(sector_name, ts_full, train, test, forecasts, horizon, best_key, sector_color):
    fig = go.Figure()

    # ── พื้นหลังแบ่งช่วง Train / Test / Forecast ──────────────────────────
    fc_index_full = None
    for v in forecasts.values():
        fc_index_full = v['pred'].index
        break
    if fc_index_full is None:
        return fig
    fc_index = fc_index_full[:horizon]

    fig.add_vrect(x0=test.index[0], x1=test.index[-1], fillcolor=COLORS['test_bg'],
                  line_width=0, annotation_text="Test", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=COLORS['muted'])
    fig.add_vrect(x0=fc_index[0], x1=fc_index[-1], fillcolor=COLORS['fc_bg'],
                  line_width=0, annotation_text="Forecast", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=COLORS['muted'])

    # ── เส้นข้อมูลจริง ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ts_full.index, y=ts_full.values, name='ค่าจริง (Actual)',
        mode='lines', line=dict(color=COLORS['actual'], width=2.2),
    ))

    # ── เส้นโมเดลต่างๆ (test prediction + forecast ต่อกัน) ───────────────
    for mkey, data in forecasts.items():
        color = COLORS.get(MODEL_COLOR_KEY.get(mkey, ''), '#888')
        is_best = (mkey == best_key)
        test_pred = data['test_pred']
        fc_pred = data['pred'].iloc[:horizon]
        combined_x = list(test_pred.index) + list(fc_pred.index)
        combined_y = list(test_pred.values) + list(fc_pred.values)
        fig.add_trace(go.Scatter(
            x=combined_x, y=combined_y,
            name=f"{MODEL_DISPLAY_NAME.get(mkey, mkey)}" + (" ⭐" if is_best else ""),
            mode='lines', line=dict(color=color, width=3.0 if is_best else 1.6,
                                     dash='solid' if is_best else 'dot'),
            opacity=1.0 if is_best else 0.75,
        ))

        if is_best:
            ci = data['ci'].iloc[:horizon]
            fig.add_trace(go.Scatter(
                x=list(fc_pred.index) + list(fc_pred.index[::-1]),
                y=list(ci.iloc[:, 1].values) + list(ci.iloc[:, 0].values[::-1]),
                fill='toself', fillcolor=color.replace(')', ',0.13)').replace('rgb', 'rgba')
                if color.startswith('rgb') else _hex_to_rgba(color, 0.13),
                line=dict(width=0), name=f'95% CI ({MODEL_DISPLAY_NAME.get(mkey, mkey)})',
                showlegend=True, hoverinfo='skip',
            ))

    fig.update_layout(
        title=None,
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['surface'],
        font=dict(family="IBM Plex Sans Thai, sans-serif", color=COLORS['ink'], size=13),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor=COLORS['line'], title=SECTOR_UNIT),
        hovermode='x unified',
    )
    return fig


def _hex_to_rgba(hexcolor, alpha):
    hexcolor = hexcolor.lstrip('#')
    if len(hexcolor) != 6:
        return f'rgba(100,100,100,{alpha})'
    r, g, b = tuple(int(hexcolor[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def build_metrics_table(results_dict, best_key_label):
    rows = []
    for label, m in results_dict.items():
        rows.append({
            'โมเดล': label, 'MAE': round(m['MAE'], 2), 'RMSE': round(m['RMSE'], 2),
            'MAPE (%)': round(m['MAPE'], 2),
        })
    table = pd.DataFrame(rows).sort_values('MAPE (%)').reset_index(drop=True)
    return table


def build_combined_chart(sector_data, horizon):
    """กราฟรวม CO2 ทั้ง 3 sector (ใช้โมเดลที่ดีที่สุดต่อ sector) เพื่อดูภาพรวมทั้งประเทศ/ทั้งระบบ"""
    fig = go.Figure()
    total_hist = None
    total_fc = None
    for sector, (ts_full, fc_index, fc_series, color) in sector_data.items():
        if total_hist is None:
            total_hist = ts_full.copy()
            total_fc = fc_series.iloc[:horizon].copy()
        else:
            total_hist = total_hist.add(ts_full, fill_value=0)
            total_fc = total_fc.add(fc_series.iloc[:horizon], fill_value=0)

    if total_hist is None:
        return fig

    fig.add_trace(go.Scatter(x=total_hist.index, y=total_hist.values, name='รวม 3 Sector (Actual)',
                              mode='lines', line=dict(color=COLORS['ink'], width=2.4)))
    fc_idx = total_fc.index
    fig.add_trace(go.Scatter(
        x=[total_hist.index[-1]] + list(fc_idx), y=[total_hist.values[-1]] + list(total_fc.values),
        name='รวม 3 Sector (Forecast, โมเดลที่ดีที่สุดต่อ sector)',
        mode='lines', line=dict(color=COLORS['accent'], width=2.6, dash='dash'),
    ))
    fig.add_vrect(x0=fc_idx[0], x1=fc_idx[-1], fillcolor=COLORS['fc_bg'], line_width=0,
                  annotation_text="Forecast", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=COLORS['muted'])
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=COLORS['surface'], paper_bgcolor=COLORS['surface'],
        font=dict(family="IBM Plex Sans Thai, sans-serif", color=COLORS['ink'], size=13),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=COLORS['line'], title=SECTOR_UNIT),
        hovermode='x unified',
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    st.markdown('<div class="app-title">🌿 CO₂ Emission Forecast Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'การพยากรณ์การปลดปล่อย CO₂ ราย Sector — Power · Transport · Industry '
        '&nbsp;|&nbsp; เปรียบเทียบ 7 โมเดลพยากรณ์อนุกรมเวลา</div>',
        unsafe_allow_html=True,
    )

    cfg = render_sidebar()

    if cfg['uploaded'] is None:
        st.info(
            "👈 กรุณาอัปโหลดไฟล์ข้อมูล CO₂ จากแถบด้านซ้าย เพื่อเริ่มการพยากรณ์\n\n"
            "**รูปแบบไฟล์ที่รองรับ:**\n"
            "- **Excel (EPPO multi-sheet):** ไฟล์ต้นฉบับที่มี 3 ชีต (เช่น `eppo-power-dataset`, "
            "`eppo-transport-dataset`, `eppo-industry-dataset`) แต่ละชีตมีแถวปีตามด้วยแถวเดือน "
            "Jan–Dec และคอลัมน์ค่าเชื้อเพลิง (oil/coal/gas) + total\n"
            "- **CSV (tidy):** คอลัมน์ `date` (หรือ `year`+`month`), `sector`, `total` "
            "และคอลัมน์เชื้อเพลิงเพิ่มเติมได้ (ไม่บังคับ)"
        )
        st.stop()

    file_bytes = cfg['uploaded'].getvalue()

    with st.spinner("กำลังอ่านไฟล์ข้อมูล..."):
        if cfg['file_format'].startswith("Excel"):
            try:
                sectors = load_sectors_from_excel(file_bytes)
            except Exception as e:
                st.error(f"ไม่สามารถอ่านไฟล์ Excel ได้: {e}")
                st.stop()
        else:
            try:
                sectors = load_sectors_from_csv(file_bytes)
            except Exception as e:
                st.error(f"ไม่สามารถอ่านไฟล์ CSV ได้: {e}")
                st.stop()

    if not sectors:
        st.warning("ยังไม่มี sector ที่จับคู่ข้อมูลสำเร็จ กรุณาตรวจสอบการเลือกชีต/คอลัมน์ด้านบน")
        st.stop()

    data_ok = render_data_preview(sectors)
    if not data_ok:
        st.warning(
            "⚠️ พบข้อสังเกตบางอย่างในข้อมูล (ดูคำเตือนด้านบนของแต่ละ sector) "
            "ระบบจะพยายามพยากรณ์ต่อไปได้ แต่ผลลัพธ์อาจไม่แม่นยำเท่าข้อมูลที่สมบูรณ์"
        )

    if not cfg['selected_models']:
        st.warning("กรุณาเลือกอย่างน้อย 1 โมเดลจากแถบด้านซ้าย")
        st.stop()

    run = st.button("🚀 รันการพยากรณ์ทั้ง 3 Sector", type="primary", use_container_width=False)
    if 'has_run' not in st.session_state:
        st.session_state['has_run'] = False
    if run:
        st.session_state['has_run'] = True

    if not st.session_state['has_run']:
        st.stop()

    st.markdown("---")

    sector_colors = {'Power': COLORS['power'], 'Transport': COLORS['transport'], 'Industry': COLORS['industry']}
    sector_captions = {
        'Power': 'ภาคการผลิตไฟฟ้า (Power Generation)',
        'Transport': 'ภาคขนส่ง (Transport)',
        'Industry': 'ภาคอุตสาหกรรม (Industry)',
    }

    all_outputs = {}
    summary_rows = []
    combined_chart_data = {}

    progress = st.progress(0.0, text="กำลังเตรียมโมเดล...")
    n_sectors = len(sectors)

    for i, (sector, (df, fuel_cols, total_col)) in enumerate(sectors.items()):
        progress.progress(i / n_sectors, text=f"กำลังประมวลผล sector: {sector} ...")
        data_hash = f"{sector}_{df['total' if 'total' in df.columns else total_col].sum():.4f}_{len(df)}"
        records = df_to_records(df)
        with st.spinner(f"กำลัง fit โมเดลสำหรับ {sector} (ครั้งแรกอาจใช้เวลาสักครู่ ผลถัดไปจะถูกแคชไว้)..."):
            # หมายเหตุ: fit โมเดลด้วย horizon สูงสุดคงที่ (36 เดือน) เสมอ ไม่ผูกกับค่า slider ที่เลือก
            # เพื่อให้การเลื่อน "จำนวนเดือนที่พยากรณ์" ทำได้ทันทีโดยไม่ต้อง refit โมเดลใหม่ทุกครั้ง
            out = cached_run_models(
                data_hash, tuple(records), total_col, tuple(fuel_cols),
                cfg['test_months'], 36, tuple(cfg['selected_models']),
                cfg['order_mode'], cfg['manual_arima_order'], cfg['manual_sarima_order'],
                cfg['manual_sarima_seasonal'], cfg['use_covid'],
            )
        all_outputs[sector] = out
    progress.progress(1.0, text="เสร็จสิ้น")
    progress.empty()

    # ── ส่วนสรุปภาพรวม KPI ───────────────────────────────────────────────
    st.markdown("### 🏆 สรุปภาพรวม — โมเดลที่ดีที่สุดต่อ Sector")
    kpi_cols = st.columns(len(all_outputs))
    best_keys = {}
    for col, (sector, out) in zip(kpi_cols, all_outputs.items()):
        if not out['results']:
            continue
        results_df = pd.DataFrame([
            {'label': lbl, **m} for lbl, m in out['results'].items()
        ])
        best_row = results_df.loc[results_df['MAPE'].idxmin()]
        # map label กลับเป็น model key
        best_key = next((k for k, v in out['forecasts'].items() if v['label'] == best_row['label']), None)
        best_keys[sector] = best_key
        with col:
            st.metric(
                f"{sector} — โมเดลที่ดีที่สุด",
                MODEL_DISPLAY_NAME.get(best_key, best_row['label']),
                delta=f"MAPE {best_row['MAPE']:.2f}%",
                delta_color="off",
            )

    # ── ส่วนแต่ละ Sector (ทั้ง 3 อยู่ในหน้าเดียวกัน) ───────────────────────
    for sector, (df, fuel_cols, total_col) in sectors.items():
        out = all_outputs[sector]
        sector_band_open(sector, sector_colors[sector], sector_captions[sector])

        if not out['forecasts']:
            st.warning("ไม่มีผลลัพธ์โมเดลสำหรับ sector นี้ (ตรวจสอบการเลือกโมเดล/ข้อมูล)")
            sector_band_close()
            continue

        best_key = best_keys.get(sector)
        fig = build_forecast_chart(
            sector, out['ts_full'], out['train'], out['test'], out['forecasts'],
            cfg['max_horizon'], best_key, sector_colors[sector],
        )
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{sector}")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**📐 ผลเปรียบเทียบความแม่นยำ (Test set)**")
            mtable = build_metrics_table(out['results'], best_key)
            st.dataframe(
                mtable.style.highlight_min(subset=['MAPE (%)'], color=COLORS['fc_bg']),
                use_container_width=True, hide_index=True, height=min(38 * (len(mtable) + 1), 280),
            )
            best_label = next((v['label'] for v in out['forecasts'].values()
                                if v['label'] == mtable.iloc[0]['โมเดล']), mtable.iloc[0]['โมเดล'])
            st.markdown(
                f'<span class="badge badge-best">⭐ ดีที่สุด: {best_label}</span>'
                f'<span class="badge badge-info">Test: {cfg["test_months"]} เดือนล่าสุด</span>',
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(f"**📋 ตารางพยากรณ์ {cfg['max_horizon']} เดือนข้างหน้า (โมเดลที่ดีที่สุด)**")
            if best_key and best_key in out['forecasts']:
                fc_show = out['forecasts'][best_key]['pred'].iloc[:cfg['max_horizon']]
                ci_show = out['forecasts'][best_key]['ci'].iloc[:cfg['max_horizon']]
                fc_table = pd.DataFrame({
                    'เดือน': fc_show.index.strftime('%Y-%m'),
                    'พยากรณ์': fc_show.values.round(1),
                    'ขอบล่าง (95% CI)': ci_show.iloc[:, 0].values.round(1),
                    'ขอบบน (95% CI)': ci_show.iloc[:, 1].values.round(1),
                })
                st.dataframe(fc_table, use_container_width=True, hide_index=True, height=280)
                csv_bytes = fc_table.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    f"⬇️ ดาวน์โหลดตารางพยากรณ์ {sector} (CSV)", data=csv_bytes,
                    file_name=f"forecast_{sector}.csv", mime="text/csv", key=f"dl_{sector}",
                )

        if best_key in ('Hybrid1', 'Hybrid2') and 'ridge_coef' in out['forecasts'].get(best_key, {}):
            with st.expander(f"🔑 สัมประสิทธิ์ Ridge ของ {sector} (ผลของสัดส่วนเชื้อเพลิงต่อ residual)"):
                coef = out['forecasts'][best_key]['ridge_coef']
                coef_df = pd.DataFrame({'เชื้อเพลิง': list(coef.keys()), 'ค่าสัมประสิทธิ์': list(coef.values())})
                st.dataframe(coef_df, hide_index=True, use_container_width=True)

        sector_band_close()

        if best_key and best_key in out['forecasts']:
            combined_chart_data[sector] = (
                out['ts_full'], out['fc_index'], out['forecasts'][best_key]['pred'], sector_colors[sector],
            )

    # ── ส่วนรวม 3 Sector ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 ภาพรวม CO₂ รวม 3 Sector (ใช้โมเดลที่ดีที่สุดต่อ Sector)")
    if combined_chart_data:
        combined_fig = build_combined_chart(combined_chart_data, cfg['max_horizon'])
        st.plotly_chart(combined_fig, use_container_width=True, key="combined_chart")

        total_fc_value = sum(
            data[2].iloc[:cfg['max_horizon']].sum() for data in combined_chart_data.values()
        )
        total_hist_last12 = sum(
            data[0].iloc[-12:].sum() for data in combined_chart_data.values()
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("รวม CO₂ พยากรณ์ทั้งหมด", f"{total_fc_value:,.0f} พันตัน",
                  help=f"รวม {cfg['max_horizon']} เดือนข้างหน้า ทั้ง 3 sector")
        c2.metric("ค่าเฉลี่ย CO₂ 12 เดือนล่าสุด (รวม 3 sector)", f"{total_hist_last12:,.0f} พันตัน")
        avg_monthly_fc = total_fc_value / cfg['max_horizon']
        c3.metric("ค่าเฉลี่ยพยากรณ์ต่อเดือน", f"{avg_monthly_fc:,.0f} พันตัน")

    # ── Ranking ภาพรวมทุกโมเดลทุก sector ───────────────────────────────
    st.markdown("### 📊 Overall Ranking — เปรียบเทียบโมเดลข้ามทุก Sector")
    rank_rows = []
    for sector, out in all_outputs.items():
        if not out['results']:
            continue
        res_df = pd.DataFrame([{'label': lbl, **m} for lbl, m in out['results'].items()])
        res_df = res_df.sort_values('MAPE').reset_index(drop=True)
        res_df['rank'] = res_df.index + 1
        key_map = {v['label']: k for k, v in out['forecasts'].items()}
        for _, row in res_df.iterrows():
            rank_rows.append({
                'โมเดล': key_map.get(row['label'], row['label']), 'Sector': sector,
                'MAPE (%)': round(row['MAPE'], 2), 'Rank': int(row['rank']),
            })
    if rank_rows:
        rank_df = pd.DataFrame(rank_rows)
        pivot_mape = rank_df.pivot_table(index='โมเดล', columns='Sector', values='MAPE (%)', aggfunc='first')
        pivot_rank = rank_df.pivot_table(index='โมเดล', columns='Sector', values='Rank', aggfunc='first')
        pivot_mape['ค่าเฉลี่ย MAPE (%)'] = pivot_mape.mean(axis=1).round(2)
        pivot_rank['ค่าเฉลี่ย Rank'] = pivot_rank.mean(axis=1).round(2)
        pivot_mape = pivot_mape.loc[pivot_rank.sort_values('ค่าเฉลี่ย Rank').index]
        pivot_mape.index = [MODEL_DISPLAY_NAME.get(k, k) for k in pivot_mape.index]
        st.dataframe(pivot_mape.round(2), use_container_width=True)
        st.caption("เรียงจากโมเดลที่มี Rank เฉลี่ยดีที่สุด (อันดับ 1 = ดีที่สุดในแต่ละ sector) ไปยังแย่ที่สุด")

    st.markdown(
        '<div class="footer-note">'
        'CO₂ Emission Forecast Dashboard — จัดทำเพื่อการนำเสนอผลงานวิชาการ<br>'
        f'Train/Test split: ใช้ {cfg["test_months"]} เดือนล่าสุดของข้อมูลจริงเป็น Test set '
        '(แบบ dynamic ตามข้อมูลที่อัปโหลด) | Forecast horizon สูงสุด 36 เดือน คำนวณครั้งเดียวแล้วเลื่อนดูได้'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
