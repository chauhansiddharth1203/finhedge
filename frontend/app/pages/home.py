"""
home.py - FinHedge home dashboard.
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
div[data-testid="metric-container"] {
    background: #111827; border: 1px solid #1e293b; border-radius: 8px;
    padding: 1rem 1.25rem; box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
div[data-testid="metric-container"] label {
    color: #64748b !important; font-size: 0.7rem !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important; font-size: 1.4rem !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 2px solid #1e293b; gap: 0; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-weight: 500; font-size: 0.875rem;
    padding: 0.6rem 1.2rem; border-radius: 0; border-bottom: 2px solid transparent; margin-bottom: -2px; }
.stTabs [aria-selected="true"] { color: #10b981 !important; border-bottom: 2px solid #10b981 !important; background: transparent !important; }
.stButton > button[kind="primary"] { background: #10b981 !important; border: none !important;
    border-radius: 6px !important; font-weight: 600 !important; color: #fff !important; }
.stButton > button[kind="primary"]:hover { background: #059669 !important; }
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }
section[data-testid="stSidebar"] { border-right: 1px solid #1e293b !important; }
div[data-testid="stAlert"] { border-radius: 6px !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Header
st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;border-bottom:1px solid #1e293b;margin-bottom:1.5rem;">
    <div style="font-size:1.75rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.03em;">
        FinHedge <span style="color:#10b981;">AI</span>
    </div>
    <div style="font-size:0.875rem;color:#64748b;margin-top:0.25rem;">
        Stock Price Prediction &amp; Deep Hedging Platform - DA5402 MLOps - IIT Madras
    </div>
</div>
""", unsafe_allow_html=True)

# System status
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}

def check_ready():
    try:
        r = requests.get(f"{BACKEND_URL}/ready", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

backend_ok, health_data = check_backend()
ready_data  = check_ready()
mlflow_ok   = ready_data.get("checks", {}).get("mlflow", False)
models_ok   = ready_data.get("checks", {}).get("models_loaded", False)
uptime      = health_data.get("uptime_seconds", 0)

st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;'>System Status</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
status_cards = [
    (c1, "Backend API", "#10b981" if backend_ok else "#f43f5e", "Online" if backend_ok else "Offline"),
    (c2, "MLflow",      "#10b981" if mlflow_ok  else "#f59e0b", "Connected" if mlflow_ok else "Unavailable"),
    (c3, "Models",      "#10b981" if models_ok  else "#f59e0b", "Ready" if models_ok else "Not Trained"),
]
for col, label, color, val in status_cards:
    with col:
        st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {color};
        border-radius:8px;padding:1rem 1.25rem;">
        <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">{label}</div>
        <div style="font-size:1.2rem;font-weight:600;color:{color};margin-top:0.4rem;">● {val}</div>
        </div>""", unsafe_allow_html=True)

with c4:
    hrs  = int(uptime // 3600)
    mins = int((uptime % 3600) // 60)
    secs = int(uptime % 60)
    up   = f"{hrs}h {mins}m {secs}s" if hrs else (f"{mins}m {secs}s" if mins else f"{secs}s")
    st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid #3b82f6;
    border-radius:8px;padding:1rem 1.25rem;">
    <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">Uptime</div>
    <div style="font-size:1.2rem;font-weight:600;color:#f1f5f9;margin-top:0.4rem;">{up if uptime else "—"}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# Quick start
st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;'>Getting Started</div>", unsafe_allow_html=True)
steps = [
    ("#3b82f6","01","Ingest Data",   "Pipeline - Run Ingestion to download 2 years of stock data."),
    ("#8b5cf6","02","Train Model",   "Pipeline - Train Model to fit LSTM and XGBoost models."),
    ("#10b981","03","Run Prediction","Prediction - select ticker and model to view forecast."),
    ("#f59e0b","04","Hedge Position","Hedging - enter your portfolio to get CVaR-optimal hedge."),
]
cols = st.columns(4)
for col, (color, num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-radius:10px;
        padding:1.25rem;height:140px;">
        <div style="font-size:0.65rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:0.1em;">Step {num}</div>
        <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin:0.4rem 0 0.5rem 0;">{title}</div>
        <div style="font-size:0.8rem;color:#64748b;line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# Platform overview
st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;'>Platform Overview</div>", unsafe_allow_html=True)
ca, cb, cc = st.columns(3)
overview = [
    (ca, "#10b981", "AI Models",    "LSTM (64-32 units) - price regression<br>XGBoost - directional classification<br>Deep CVaR Hedger (MLP policy network)<br>MLflow experiment tracking"),
    (cb, "#3b82f6", "MLOps Stack",  "Apache Airflow - scheduled pipelines<br>DVC - data and model versioning<br>Prometheus + Grafana - monitoring<br>Docker Compose - 6-service orchestration"),
    (cc, "#f59e0b", "Risk Metrics", "RMSE, MAE, MAPE, R2 Score<br>Direction Accuracy<br>Annualised Sharpe Ratio<br>CVaR (95%) and Max Drawdown"),
]
for col, color, title, body in overview:
    with col:
        st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;">
        <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:{color};margin-bottom:0.75rem;">{title}</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:2.">{body}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:1rem 1.5rem;
display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <div>
        <div style="font-size:0.8rem;font-weight:600;color:#f1f5f9;">Siddharth Chauhan</div>
        <div style="font-size:0.7rem;color:#64748b;">Roll No: CH21B103 - IIT Madras</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.7rem;color:#64748b;">ch21b103@smail.iitm.ac.in - +91 9518241509</div>
        <div style="font-size:0.7rem;color:#334155;">DA5402 MLOps Course Project - FinHedge AI v1.0</div>
    </div>
</div>
""", unsafe_allow_html=True)
