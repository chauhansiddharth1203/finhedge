"""
main.py – FinHedge Streamlit home dashboard.

This file is the entry point for the Streamlit multi-page app.
All pages are in the pages/ directory.
"""

import os
import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinHedge AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
    }
    .status-ok   { color: #28a745; font-weight: bold; }
    .status-warn { color: #ffc107; font-weight: bold; }
    .status-err  { color: #dc3545; font-weight: bold; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 FinHedge AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Stock Price Prediction & Deep Hedging Platform</p>',
    unsafe_allow_html=True,
)

# ── System status ──────────────────────────────────────────────────────────
st.subheader("System Status")

col1, col2, col3, col4 = st.columns(4)

def check_backend() -> tuple[bool, dict]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}

def check_ready() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/ready", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

backend_ok, health_data = check_backend()
ready_data = check_ready()

with col1:
    status = "🟢 Online" if backend_ok else "🔴 Offline"
    st.metric("Backend API", status)

with col2:
    mlflow_ok = ready_data.get("checks", {}).get("mlflow", False)
    st.metric("MLflow", "🟢 Connected" if mlflow_ok else "🟡 Unavailable")

with col3:
    models_ok = ready_data.get("checks", {}).get("models_loaded", False)
    st.metric("Models", "🟢 Loaded" if models_ok else "🟡 Not trained yet")

with col4:
    uptime = health_data.get("uptime_seconds", 0)
    st.metric("Uptime", f"{uptime:.0f}s" if uptime else "—")

st.divider()

# ── Quick start guide ──────────────────────────────────────────────────────
st.subheader("Quick Start")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("**Step 1: Pipeline**\n\nGo to **Pipeline** page → enter ticker → click **Run Ingestion** to download stock data.")
with c2:
    st.info("**Step 2: Train**\n\nStill on **Pipeline** page → click **Train Model** to train LSTM + XGBoost.")
with c3:
    st.success("**Step 3: Predict**\n\nGo to **Prediction** page → select ticker & model → view next-day price forecast.")
with c4:
    st.warning("**Step 4: Hedge**\n\nGo to **Hedging** page → enter position details → get optimal hedge recommendation.")

st.divider()

# ── Feature overview ───────────────────────────────────────────────────────
st.subheader("Platform Features")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
    **🧠 AI Models**
    - LSTM (64→32 units) for price regression
    - XGBoost for directional classification
    - CVaR Deep Hedger (MLP + CVaR loss)
    - Trained & tracked via MLflow
    """)
with col_b:
    st.markdown("""
    **⚙️ MLOps Stack**
    - Apache Airflow data pipeline
    - DVC for version control & CI
    - MLflow experiment tracking
    - Prometheus + Grafana monitoring
    """)
with col_c:
    st.markdown("""
    **📊 Financial Metrics**
    - RMSE, MAE, MAPE
    - Direction Accuracy
    - Annualised Sharpe Ratio
    - CVaR (95%) & Max Drawdown
    """)

st.divider()
st.caption("FinHedge AI v1.0.0 | DA5402 – MLOps Course Project | IIT Madras")
