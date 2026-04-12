"""
04_Monitoring.py – Live monitoring page.

Shows:
  - Prometheus metrics (scraped via HTTP)
  - Data drift status
  - Model performance over time
  - Grafana iframe embed
  - Alert log
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

BACKEND_URL  = os.getenv("BACKEND_URL",  "http://localhost:8000")
GRAFANA_URL  = os.getenv("GRAFANA_URL",  "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

st.set_page_config(page_title="Monitoring | FinHedge", page_icon="📡", layout="wide")
st.markdown("## 📡 Live Monitoring")
st.caption("Real-time model performance, data drift, and infrastructure metrics.")

# ── Auto-refresh ───────────────────────────────────────────────────────────
refresh_interval = st.sidebar.selectbox("Auto-refresh", [0, 10, 30, 60], index=0,
                                        format_func=lambda x: "Off" if x == 0 else f"{x}s")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()

if refresh_interval > 0:
    st.sidebar.success(f"Auto-refreshing every {refresh_interval}s")
    time.sleep(refresh_interval)
    st.rerun()

# ── Prometheus metric fetcher ──────────────────────────────────────────────
def prom_query(query: str) -> float | None:
    """Query Prometheus instant value."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        data = r.json()
        result = data.get("data", {}).get("result", [])
        if result:
            return float(result[0]["value"][1])
    except Exception:
        pass
    return None

def prom_range(query: str, duration: str = "1h") -> pd.DataFrame:
    """Query Prometheus range and return DataFrame."""
    try:
        now   = int(time.time())
        step  = 30
        start = now - {"1h": 3600, "6h": 21600, "1d": 86400}.get(duration, 3600)
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": query, "start": start, "end": now, "step": step},
            timeout=5,
        )
        result = r.json().get("data", {}).get("result", [])
        rows   = []
        for series in result:
            for ts, val in series["values"]:
                rows.append({"timestamp": pd.to_datetime(ts, unit="s"), "value": float(val),
                             "labels": str(series["metric"])})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ── Layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🌊 Drift Detection", "📺 Grafana Dashboard"])

with tab1:
    st.subheader("Live System Metrics")

    # ── Metric cards ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total_preds = prom_query(f'finhedge_prediction_requests_total{{ticker="{ticker}"}}')
    total_errors = prom_query(f'finhedge_prediction_errors_total{{ticker="{ticker}"}}')
    rmse = prom_query(f'finhedge_model_rmse{{ticker="{ticker}",model="lstm"}}')
    sharpe = prom_query(f'finhedge_model_sharpe{{ticker="{ticker}",model="lstm"}}')

    with col1:
        st.metric("Total Predictions", f"{int(total_preds) if total_preds else 0:,}")
    with col2:
        err_rate = (total_errors / total_preds * 100) if total_preds and total_errors else 0
        colour   = "normal" if err_rate < 5 else "inverse"
        st.metric("Error Rate", f"{err_rate:.2f}%", delta=None,
                  help="Alert threshold: 5%")
    with col3:
        st.metric("Model RMSE", f"{rmse:.4f}" if rmse else "—")
    with col4:
        st.metric("Sharpe Ratio", f"{sharpe:.4f}" if sharpe else "—")

    st.divider()

    # ── Prediction latency histogram ──────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Prediction Rate (last 1h)")
        df_rate = prom_range(
            f'rate(finhedge_prediction_requests_total{{ticker="{ticker}"}}[5m])', "1h"
        )
        if not df_rate.empty:
            fig = go.Figure(go.Scatter(
                x=df_rate["timestamp"], y=df_rate["value"],
                mode="lines", fill="tozeroy",
                line=dict(color="#667eea"),
            ))
            fig.update_layout(height=250, plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_title="Time", yaxis_title="Req/s")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction traffic yet. Run some predictions to see data.")

    with col_r:
        st.subheader("Inference Latency (p50, p90, p99)")
        p50 = prom_query(
            f'histogram_quantile(0.50, rate(finhedge_prediction_latency_seconds_bucket{{ticker="{ticker}"}}[5m]))'
        )
        p90 = prom_query(
            f'histogram_quantile(0.90, rate(finhedge_prediction_latency_seconds_bucket{{ticker="{ticker}"}}[5m]))'
        )
        p99 = prom_query(
            f'histogram_quantile(0.99, rate(finhedge_prediction_latency_seconds_bucket{{ticker="{ticker}"}}[5m]))'
        )
        perc_data = {
            "Percentile": ["p50", "p90", "p99"],
            "Latency (s)": [p50 or 0, p90 or 0, p99 or 0],
        }
        fig_lat = px.bar(
            perc_data, x="Percentile", y="Latency (s)",
            color="Percentile", color_discrete_sequence=["#28a745", "#ffc107", "#dc3545"],
        )
        fig_lat.update_layout(height=250, plot_bgcolor="white", paper_bgcolor="white",
                               showlegend=False)
        fig_lat.add_hline(y=0.2, line_dash="dash", line_color="red",
                          annotation_text="200ms SLA")
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Hedge action distribution ─────────────────────────────────────────
    st.subheader("Hedge Actions Distribution")
    actions = {"HEDGE_SHORT": 0, "HEDGE_LONG": 0, "HOLD": 0}
    for act in actions:
        v = prom_query(f'finhedge_hedge_requests_total{{ticker="{ticker}",action="{act}"}}')
        if v:
            actions[act] = int(v)
    if sum(actions.values()) > 0:
        fig_pie = px.pie(
            values=list(actions.values()),
            names=list(actions.keys()),
            color_discrete_map={
                "HEDGE_SHORT": "#dc3545",
                "HEDGE_LONG":  "#28a745",
                "HOLD":        "#6c757d",
            },
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No hedge requests recorded yet.")

with tab2:
    st.subheader("Data Drift Detection")
    st.caption("KL-divergence of current feature distributions vs. EDA baseline.")

    # Fetch drift scores from Prometheus
    drift_alert = prom_query(f'finhedge_drift_alert{{ticker="{ticker}"}}')
    if drift_alert is not None:
        if drift_alert == 1:
            st.error("🚨 **DRIFT ALERT**: Significant data drift detected! Consider retraining.")
        else:
            st.success("✅ No significant data drift detected.")

    # Simulated drift scores for display (replace with live Prometheus data)
    feature_names = [
        "rsi_14", "macd", "bb_width", "hist_vol_20", "vol_ma_ratio",
        "return_1d", "ret_mean_5", "ret_std_20", "atr_14", "roc_10",
    ]
    drift_scores = []
    for feat in feature_names:
        score = prom_query(
            f'finhedge_data_drift_score{{ticker="{ticker}",feature="{feat}"}}'
        )
        drift_scores.append(score if score is not None else 0.0)

    df_drift = pd.DataFrame({"Feature": feature_names, "KL Divergence": drift_scores})
    df_drift["Status"] = df_drift["KL Divergence"].apply(
        lambda x: "🔴 Drift" if x > 0.05 else "🟢 OK"
    )
    df_drift = df_drift.sort_values("KL Divergence", ascending=False)

    fig_drift = px.bar(
        df_drift, x="KL Divergence", y="Feature", orientation="h",
        color="KL Divergence",
        color_continuous_scale=["#28a745", "#ffc107", "#dc3545"],
    )
    fig_drift.add_vline(x=0.05, line_dash="dash", line_color="red",
                        annotation_text="Drift Threshold (0.05)")
    fig_drift.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_drift, use_container_width=True)

    st.dataframe(
        df_drift,
        use_container_width=True, hide_index=True,
    )

with tab3:
    st.subheader("Grafana Dashboard")
    st.caption(f"Live dashboards at [{GRAFANA_URL}]({GRAFANA_URL}) (login: admin / finhedge123)")

    # Embed Grafana panel
    grafana_panel_url = (
        f"{GRAFANA_URL}/d/finhedge/finhedge-ai-monitoring"
        f"?orgId=1&refresh=10s&kiosk"
    )
    st.markdown(
        f'<iframe src="{grafana_panel_url}" width="100%" height="600" '
        f'frameborder="0"></iframe>',
        unsafe_allow_html=True,
    )
    st.info(
        "If the Grafana iframe is blank, open it directly: "
        f"[Grafana Dashboard ↗]({GRAFANA_URL})"
    )
