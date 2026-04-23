"""
main.py – FinHedge home dashboard.
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FinHedge AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — professional dark financial terminal ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid #1e293b;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.6rem 1.2rem;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #10b981 !important;
    border-bottom: 2px solid #10b981 !important;
    background: transparent !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: #10b981 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #059669 !important;
    box-shadow: 0 0 12px rgba(16,185,129,0.3) !important;
}

/* Dividers */
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid #1e293b !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* Info/success/warning boxes */
div[data-testid="stAlert"] { border-radius: 6px !important; border-width: 1px !important; }

/* Dataframes */
div[data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.5rem;">
    <div style="font-size: 1.75rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.03em;">
        FinHedge <span style="color: #10b981;">AI</span>
    </div>
    <div style="font-size: 0.875rem; color: #64748b; margin-top: 0.25rem;">
        Stock Price Prediction &amp; Deep Hedging Platform &nbsp;·&nbsp; DA5402 MLOps &nbsp;·&nbsp; IIT Madras
    </div>
</div>
""", unsafe_allow_html=True)

# ── System Status ──────────────────────────────────────────────────────────
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
ready_data = check_ready()
mlflow_ok  = ready_data.get("checks", {}).get("mlflow", False)
models_ok  = ready_data.get("checks", {}).get("models_loaded", False)
uptime     = health_data.get("uptime_seconds", 0)

st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;'>System Status</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    color = "#10b981" if backend_ok else "#f43f5e"
    label = "Online" if backend_ok else "Offline"
    st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {color};
    border-radius:8px;padding:1rem 1.25rem;">
    <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">Backend API</div>
    <div style="font-size:1.2rem;font-weight:600;color:{color};margin-top:0.4rem;">● {label}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    color = "#10b981" if mlflow_ok else "#f59e0b"
    label = "Connected" if mlflow_ok else "Unavailable"
    st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {color};
    border-radius:8px;padding:1rem 1.25rem;">
    <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">MLflow</div>
    <div style="font-size:1.2rem;font-weight:600;color:{color};margin-top:0.4rem;">● {label}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    color = "#10b981" if models_ok else "#f59e0b"
    label = "Models Ready" if models_ok else "Not Trained"
    st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {color};
    border-radius:8px;padding:1rem 1.25rem;">
    <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">Models</div>
    <div style="font-size:1.2rem;font-weight:600;color:{color};margin-top:0.4rem;">● {label}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    hrs  = int(uptime // 3600)
    mins = int((uptime % 3600) // 60)
    secs = int(uptime % 60)
    uptime_str = f"{hrs}h {mins}m {secs}s" if hrs > 0 else (f"{mins}m {secs}s" if mins > 0 else f"{secs}s")
    st.markdown(f"""<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid #3b82f6;
    border-radius:8px;padding:1rem 1.25rem;">
    <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">Uptime</div>
    <div style="font-size:1.2rem;font-weight:600;color:#f1f5f9;margin-top:0.4rem;">{uptime_str if uptime else "—"}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Quick Start ────────────────────────────────────────────────────────────
st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;'>Getting Started</div>", unsafe_allow_html=True)

steps = [
    ("#3b82f6", "01", "Ingest Data",   "Pipeline page → Run Ingestion to download 2 years of stock data."),
    ("#8b5cf6", "02", "Train Model",   "Pipeline page → Train Model to fit LSTM + XGBoost models."),
    ("#10b981", "03", "Run Prediction","Prediction page → select ticker and model → view forecast."),
    ("#f59e0b", "04", "Hedge Position","Hedging page → enter your portfolio → get CVaR-optimal hedge."),
]
cols = st.columns(4)
for col, (color, num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;height:140px;">
            <div style="font-size:0.65rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:0.1em;">Step {num}</div>
            <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin:0.4rem 0 0.5rem 0;">{title}</div>
            <div style="font-size:0.8rem;color:#64748b;line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Platform Overview ──────────────────────────────────────────────────────
st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;'>Platform Overview</div>", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;">
        <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#10b981;margin-bottom:0.75rem;">AI Models</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:2;">
            LSTM (64→32 units) — price regression<br>
            XGBoost — directional classification<br>
            Deep CVaR Hedger (MLP policy network)<br>
            MLflow experiment tracking
        </div>
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;">
        <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#3b82f6;margin-bottom:0.75rem;">MLOps Stack</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:2;">
            Apache Airflow — scheduled pipelines<br>
            DVC — data & model versioning<br>
            Prometheus + Grafana — monitoring<br>
            Docker Compose — 6-service orchestration
        </div>
    </div>""", unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;">
        <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#f59e0b;margin-bottom:0.75rem;">Risk Metrics</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:2;">
            RMSE, MAE, MAPE, R² Score<br>
            Direction Accuracy<br>
            Annualised Sharpe Ratio<br>
            CVaR (95%) & Max Drawdown
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:1rem 1.5rem;
margin-top:1rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <div>
        <div style="font-size:0.75rem;font-weight:600;color:#f1f5f9;">Siddharth Chauhan</div>
        <div style="font-size:0.7rem;color:#64748b;">Roll No: CH21B103 &nbsp;·&nbsp; IIT Madras</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.7rem;color:#64748b;">ch21b103@smail.iitm.ac.in &nbsp;·&nbsp; +91 9518241509</div>
        <div style="font-size:0.7rem;color:#334155;">DA5402 MLOps Course Project &nbsp;·&nbsp; FinHedge AI v1.0</div>
    </div>
</div>
""", unsafe_allow_html=True)
