"""
05_About.py – Project report and documentation page.
"""
import streamlit as st

st.set_page_config(page_title="About | FinHedge", page_icon="📋", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
hr { border-color: #1e293b !important; }
section[data-testid="stSidebar"] { border-right: 1px solid #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 1.25rem 0;border-bottom:1px solid #1e293b;margin-bottom:1.5rem;">
    <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">About FinHedge</div>
    <div style="font-size:0.8rem;color:#64748b;margin-top:0.2rem;">
        Project Report · DA5402 MLOps · IIT Madras
    </div>
</div>
""", unsafe_allow_html=True)

# ── Author card ────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid #10b981;
border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
    <div style="font-size:1rem;font-weight:700;color:#f1f5f9;">Siddharth Chauhan</div>
    <div style="font-size:0.85rem;color:#64748b;margin-top:0.3rem;line-height:1.8;">
        Roll No: <span style="color:#94a3b8;font-weight:500;">CH21B103</span> &nbsp;·&nbsp;
        IIT Madras<br>
        Email: <span style="color:#94a3b8;font-weight:500;">ch21b103@smail.iitm.ac.in</span> &nbsp;·&nbsp;
        Phone: <span style="color:#94a3b8;font-weight:500;">+91 9518241509</span><br>
        Course: <span style="color:#94a3b8;font-weight:500;">DA5402 — MLOps</span> &nbsp;·&nbsp;
        April 2026
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Project", "ML Models", "MLOps Stack", "How to Use"])

# ── Tab 1: Overview ────────────────────────────────────────────────────────
with tab1:
    st.markdown("### What is FinHedge?")
    st.markdown("""
    FinHedge AI is an end-to-end MLOps application for **stock price prediction** and
    **risk-optimal hedging**. It combines three AI models into a production-ready system:

    - A **PyTorch LSTM** that forecasts next-day closing prices from 60-day lookback windows
    - An **XGBoost classifier** that predicts market direction (UP / FLAT / DOWN)
    - A **Deep CVaR Hedger** (MLP policy network) that recommends the optimal hedge ratio
      to minimise tail risk at 95% confidence
    """)

    st.markdown("### System Architecture")
    st.markdown("""
    ```
    Yahoo Finance Data
          │
          ▼
    Ingestion → Feature Engineering → Preprocessing
                                            │
                        ┌───────────────────┼──────────────────┐
                        ▼                   ▼                  ▼
                  LSTM Model          XGBoost Model      Deep Hedger
                  (price reg.)        (direction)        (CVaR policy)
                        └───────────────────┴──────────────────┘
                                            │
                                     MLflow Registry
                                            │
                                     FastAPI Backend (:8000)
                                     ┌──────┴──────┐
                                     ▼             ▼
                               Streamlit UI   Prometheus → Grafana
                               (:8501)        (:9090)      (:3001)
    ```
    """)

    col1, col2, col3 = st.columns(3)
    services = [
        ("Backend API", "FastAPI + PyTorch + XGBoost", ":8000", "#10b981"),
        ("Frontend UI", "Streamlit — 5 pages", ":8501", "#3b82f6"),
        ("MLflow", "Experiment tracking + Model registry", ":5000", "#8b5cf6"),
        ("Airflow", "Scheduled pipeline orchestration", ":8081", "#f59e0b"),
        ("Prometheus", "Metrics collection (15 metrics)", ":9090", "#ef4444"),
        ("Grafana", "10-panel monitoring dashboard", ":3001", "#06b6d4"),
    ]
    cols = st.columns(3)
    for i, (name, desc, port, color) in enumerate(services):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {color};
            border-radius:8px;padding:0.9rem 1rem;margin-bottom:0.75rem;">
                <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">{name}</div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem;">{desc}</div>
                <div style="font-size:0.7rem;font-weight:600;color:{color};margin-top:0.3rem;font-family:monospace;">localhost{port}</div>
            </div>""", unsafe_allow_html=True)

# ── Tab 2: ML Models ───────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;margin-bottom:1rem;">
            <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#10b981;margin-bottom:0.75rem;">LSTM Price Predictor</div>
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
                <b style="color:#f1f5f9;">Architecture:</b> 2-layer LSTM (64→32 units) + Linear<br>
                <b style="color:#f1f5f9;">Input:</b> 60-day × 22 features<br>
                <b style="color:#f1f5f9;">Output:</b> Next-day closing price<br>
                <b style="color:#f1f5f9;">Loss:</b> MSE with early stopping<br>
                <b style="color:#f1f5f9;">Features:</b> RSI, MACD, Bollinger Bands, ATR, OBV, volatility
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;">
            <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#f59e0b;margin-bottom:0.75rem;">Deep CVaR Hedger</div>
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
                <b style="color:#f1f5f9;">Architecture:</b> MLP (input → 64 → 32 → 1)<br>
                <b style="color:#f1f5f9;">Training:</b> Simulated GBM price paths<br>
                <b style="color:#f1f5f9;">Loss:</b> CVaR at 95% confidence<br>
                <b style="color:#f1f5f9;">Output:</b> Hedge ratio ∈ [−1, +1]<br>
                <b style="color:#f1f5f9;">Reference:</b> Black-Scholes delta hedge
            </div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;margin-bottom:1rem;">
            <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#3b82f6;margin-bottom:0.75rem;">XGBoost Classifier</div>
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
                <b style="color:#f1f5f9;">Task:</b> Direction classification<br>
                <b style="color:#f1f5f9;">Classes:</b> UP / FLAT / DOWN<br>
                <b style="color:#f1f5f9;">Input:</b> Flattened 60-day window<br>
                <b style="color:#f1f5f9;">Also:</b> Price regression head<br>
                <b style="color:#f1f5f9;">Role:</b> Baseline vs LSTM
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:1.25rem;">
            <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#8b5cf6;margin-bottom:0.75rem;">Evaluation Metrics</div>
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
                <b style="color:#f1f5f9;">RMSE</b> — price prediction error<br>
                <b style="color:#f1f5f9;">MAE</b> — mean absolute error<br>
                <b style="color:#f1f5f9;">Direction Acc.</b> — UP/DOWN correct %<br>
                <b style="color:#f1f5f9;">Sharpe Ratio</b> — risk-adjusted return<br>
                <b style="color:#f1f5f9;">CVaR 95%</b> — worst-case loss estimate
            </div>
        </div>""", unsafe_allow_html=True)

# ── Tab 3: MLOps Stack ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### Pipeline Flow")
    st.markdown("""
    ```
    DVC Pipeline:   ingest → preprocess → train → evaluate
    Airflow DAGs:   Daily (data ingestion)  +  Weekly (model retraining)
    MLflow:         Every training run → params, metrics, artifacts, model registry
    Prometheus:     Scrapes /metrics every 15s → 15 custom metrics
    Grafana:        10-panel live dashboard
    GitHub Actions: lint → pytest → dvc dag → docker build
    ```
    """)

    cols = st.columns(2)
    stack = [
        ("DVC", "dvc.yaml", "4-stage reproducible pipeline. Run: dvc repro", "#10b981"),
        ("MLflow", "trainer.py", "Logs params, metrics, artifacts + model registry", "#3b82f6"),
        ("Airflow", "dags/", "Daily ingestion + weekly retraining DAGs", "#f59e0b"),
        ("Prometheus", "prometheus.yml", "15 metrics: latency, errors, drift, RMSE", "#ef4444"),
        ("Grafana", "finhedge.json", "Pre-provisioned 10-panel dashboard", "#06b6d4"),
        ("GitHub Actions", "ci.yml", "lint + tests + DVC + Docker on every push", "#8b5cf6"),
    ]
    for i, (tool, file, desc, color) in enumerate(stack):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {color};
            border-radius:8px;padding:0.9rem 1rem;margin-bottom:0.75rem;">
                <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">{tool}</div>
                <div style="font-size:0.7rem;font-family:monospace;color:{color};margin:0.2rem 0;">{file}</div>
                <div style="font-size:0.78rem;color:#64748b;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── Tab 4: How to Use ──────────────────────────────────────────────────────
with tab4:
    st.markdown("### Quick Start")
    steps = [
        ("1", "Start the app", "docker compose up -d", "Open http://localhost:8501 in browser"),
        ("2", "Ingest data", "Pipeline page → Run Ingestion", "Downloads 2 years of stock data"),
        ("3", "Train model", "Pipeline page → Train Model", "Trains LSTM + XGBoost (3–5 mins)"),
        ("4", "Evaluate", "Pipeline page → Evaluate Model", "Computes RMSE, Sharpe, Direction Accuracy"),
        ("5", "Predict", "Prediction page → Run Prediction", "Forecasts next-day price with confidence interval"),
        ("6", "Hedge", "Hedging page → Get Hedge Recommendation", "CVaR-optimal hedge ratio for your portfolio"),
    ]
    for num, title, action, result in steps:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.75rem;
        background:#111827;border:1px solid #1e293b;border-radius:8px;padding:0.9rem 1rem;">
            <div style="background:#10b981;color:#fff;font-weight:700;font-size:0.8rem;
            border-radius:50%;width:26px;height:26px;display:flex;align-items:center;
            justify-content:center;flex-shrink:0;">{num}</div>
            <div>
                <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">{title}</div>
                <div style="font-size:0.78rem;color:#10b981;font-family:monospace;margin:0.2rem 0;">{action}</div>
                <div style="font-size:0.75rem;color:#64748b;">{result}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Service URLs")
    services_urls = [
        ("FinHedge UI", "http://localhost:8501", "—"),
        ("FastAPI Docs", "http://localhost:8000/docs", "—"),
        ("MLflow", "http://localhost:5000", "—"),
        ("Airflow", "http://localhost:8081", "admin / admin"),
        ("Grafana", "http://localhost:3001", "admin / finhedge123"),
        ("Prometheus", "http://localhost:9090", "—"),
    ]
    rows = "".join([
        f"<tr><td style='padding:0.5rem 1rem;color:#f1f5f9;font-size:0.82rem;'>{s}</td>"
        f"<td style='padding:0.5rem 1rem;font-family:monospace;font-size:0.78rem;color:#10b981;'>{u}</td>"
        f"<td style='padding:0.5rem 1rem;color:#64748b;font-size:0.78rem;'>{l}</td></tr>"
        for s, u, l in services_urls
    ])
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:#111827;
    border:1px solid #1e293b;border-radius:8px;overflow:hidden;">
        <thead><tr style="border-bottom:1px solid #1e293b;">
            <th style="padding:0.6rem 1rem;text-align:left;font-size:0.65rem;font-weight:600;
            text-transform:uppercase;letter-spacing:0.08em;color:#475569;">Service</th>
            <th style="padding:0.6rem 1rem;text-align:left;font-size:0.65rem;font-weight:600;
            text-transform:uppercase;letter-spacing:0.08em;color:#475569;">URL</th>
            <th style="padding:0.6rem 1rem;text-align:left;font-size:0.65rem;font-weight:600;
            text-transform:uppercase;letter-spacing:0.08em;color:#475569;">Login</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)
