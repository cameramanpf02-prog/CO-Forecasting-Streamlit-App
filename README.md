# 🌿 CO₂ Emission Forecasting Dashboard
### Academic Conference Edition — Thailand Energy 3 Sectors

---

## 📋 Overview

ระบบพยากรณ์การปล่อย CO₂ จาก 3 ภาคพลังงานของประเทศไทย ได้แก่  
**Power Generation · Transport · Industry**

ใช้โมเดล Time Series และ Hybrid Models เปรียบเทียบกัน 5 โมเดล:
- **ARIMA** — Baseline non-seasonal
- **SARIMAX+COVID** — Seasonal + COVID-19 dummy variable
- **ETS** — Exponential Smoothing with damped trend
- **Hybrid1 (SARIMAX+Ridge)** — SARIMAX base + Ridge residual correction
- **Hybrid2 (ETS+Ridge)** — ETS base + Ridge residual correction

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# หรือ
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

เปิดบราวเซอร์ที่ `http://localhost:8501`

---

## 📁 File Structure

```
co2_forecast_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 📊 Input Data Format

ต้องการไฟล์ Excel ของ EPPO ที่มี **3 sheets**:

| Sheet Name | Columns | Description |
|---|---|---|
| `eppo-power-dataset` | oil, coal, gas, total | Power Generation CO₂ |
| `eppo-transport-dataset` | oil, gas, total | Transport CO₂ |
| `eppo-industry-dataset` | oil, coal, gas, total | Industry CO₂ |

**Format:** แต่ละ Sheet มีปีและเดือน (Jan–Dec) ในรูปแบบ EPPO standard

---

## ⚙️ Features

| Feature | Description |
|---|---|
| 📈 Multi-model Forecast | พยากรณ์ด้วย 5 โมเดลพร้อมกัน |
| 🗓️ Adjustable Horizon | เลือกจำนวนเดือนที่พยากรณ์ได้ (6–36 เดือน) |
| 📊 Performance Metrics | MAE, RMSE, MAPE, R² บน Test set |
| 🎯 Best Model Highlight | ไฮไลต์โมเดลที่ดีที่สุดต่อ Sector |
| 📉 Radar Chart | เปรียบเทียบโมเดลด้วย Radar/Spider chart |
| 🌡️ MAPE Heatmap | Heatmap เปรียบเทียบทุกโมเดล/ทุก Sector |
| 📋 Confidence Interval | แสดง CI 95% บน Forecast |
| 💾 CSV Export | ดาวน์โหลดผลลัพธ์ทั้งหมด |

---

## 🏆 Best Models (from research)

| Sector | Best Model | MAPE |
|---|---|---|
| Power | Hybrid1 (SARIMAX+Ridge) | ~3–5% |
| Transport | Hybrid2 (ETS+Ridge) | ~3–6% |
| Industry | SARIMAX+COVID | ~4–7% |

---

## 📐 Methodology

```
Data: EPPO Thailand Monthly CO₂ (2010–2025)
Train: 2010–2023 (168 months)
Test:  2024–2025 (24 months) — มีค่าจริง ใช้วัดความแม่น
Forecast: N months ahead (configurable)

COVID Dummy: Apr 2020 – Jun 2021
Auto-order: pmdarima auto_arima (BIC criterion)
Hybrid: Base model + Ridge regression on lag features of residuals
```

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
statsmodels>=0.14.0
pmdarima>=2.0.4
scikit-learn>=1.3.0
plotly>=5.18.0
openpyxl>=3.1.0
```

---

*Developed for Academic Conference Presentation*  
*Data Source: Energy Policy and Planning Office (EPPO), Thailand*
