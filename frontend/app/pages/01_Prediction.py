"""
01_Prediction.py – Stock price prediction page.

Allows users to:
  - Select any ticker + model type
  - View next-day price forecast with confidence interval
  - See historical actual vs predicted chart
  - View model performance metrics
"""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Prediction | FinHedge", page_icon="🔮", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .predict-header { font-size:2rem; font-weight:700; color:#1a1a2e; }
    .direction-up   { color:#28a745; font-size:1.4rem; font-weight:bold; }
    .direction-down { color:#dc3545; font-size:1.4rem; font-weight:bold; }
    .direction-flat { color:#6c757d; font-size:1.4rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="predict-header">🔮 Price Prediction</p>', unsafe_allow_html=True)
st.caption("Forecast next-day stock closing price using LSTM or XGBoost models.")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Prediction Settings")

POPULAR_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "JPM", "GS"]
ticker_choice = st.sidebar.selectbox("Ticker (or type below)", POPULAR_TICKERS)
ticker_input  = st.sidebar.text_input("Custom ticker", value="", placeholder="e.g. RELIANCE.NS")
ticker        = ticker_input.strip().upper() if ticker_input.strip() else ticker_choice

model_type = st.sidebar.selectbox("Model", ["lstm", "xgboost"])
horizon    = st.sidebar.slider("Forecast horizon (days)", 1, 5, 1)

run_btn = st.sidebar.button("🚀 Run Prediction", type="primary", use_container_width=True)

# ── Main panel ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching data and running {model_type.upper()} model for {ticker} …"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/predict",
                json={"ticker": ticker, "model_type": model_type, "horizon": horizon},
                timeout=60,
            )
            if resp.status_code != 200:
                st.error(f"API Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
                st.stop()

            data = resp.json()

            # ── Key metrics row ───────────────────────────────────────────
            st.subheader(f"Results for **{ticker}**")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Current Price", f"${data['current_price']:.2f}")
            with col2:
                pred = data["predictions"][0]["predicted_price"]
                delta = pred - data["current_price"]
                st.metric("Predicted Price", f"${pred:.2f}", delta=f"{delta:+.2f}")
            with col3:
                direction = data["direction"]
                colour = {"UP": "🟢", "DOWN": "🔴", "FLAT": "🟡"}[direction]
                st.metric("Direction", f"{colour} {direction}")
            with col4:
                st.metric("Confidence", f"{data['direction_prob']:.1%}")
            with col5:
                st.metric("Hist. Volatility", f"{data['volatility_1y']:.1%}")

            st.divider()

            # ── Prediction chart ──────────────────────────────────────────
            col_chart, col_metrics = st.columns([3, 1])

            with col_chart:
                st.subheader("Price Chart & Forecast")
                # Fetch recent price history for context via yfinance
                try:
                    import yfinance as yf
                    hist = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)

                    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                        shared_xaxes=True, vertical_spacing=0.05)

                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=hist.index, open=hist["Open"], high=hist["High"],
                        low=hist["Low"],  close=hist["Close"],
                        name="OHLC", increasing_line_color="#28a745",
                        decreasing_line_color="#dc3545",
                    ), row=1, col=1)

                    # Prediction point
                    pred_date = data["predictions"][0]["date"]
                    fig.add_trace(go.Scatter(
                        x=[pred_date], y=[pred],
                        mode="markers+text",
                        marker=dict(size=14, color="#667eea", symbol="star"),
                        text=[f"${pred:.2f}"], textposition="top center",
                        name="Forecast",
                    ), row=1, col=1)

                    # Confidence band
                    lb = data["predictions"][0]["lower_bound"]
                    ub = data["predictions"][0]["upper_bound"]
                    fig.add_trace(go.Scatter(
                        x=[pred_date, pred_date], y=[lb, ub],
                        mode="lines",
                        line=dict(color="#667eea", dash="dash"),
                        name="90% CI",
                    ), row=1, col=1)

                    # Volume
                    colours = ["#28a745" if c >= o else "#dc3545"
                               for c, o in zip(hist["Close"], hist["Open"])]
                    fig.add_trace(go.Bar(
                        x=hist.index, y=hist["Volume"],
                        marker_color=colours, name="Volume", opacity=0.7,
                    ), row=2, col=1)

                    fig.update_layout(
                        height=500, xaxis_rangeslider_visible=False,
                        plot_bgcolor="white", paper_bgcolor="white",
                        legend=dict(orientation="h", y=1.02),
                    )
                    fig.update_yaxes(title_text="Price ($)", row=1)
                    fig.update_yaxes(title_text="Volume",   row=2)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not render chart: {e}")
                    st.json(data["predictions"])

            with col_metrics:
                st.subheader("Model Metrics")
                m = data.get("metrics", {})
                metric_map = {
                    "rmse":         ("RMSE",          "↓ lower is better"),
                    "mae":          ("MAE",            "↓ lower is better"),
                    "mape":         ("MAPE (%)",       "↓ lower is better"),
                    "r2":           ("R² Score",       "↑ higher is better"),
                    "direction_acc":("Direction Acc.", "↑ higher is better"),
                    "sharpe":       ("Sharpe Ratio",   "↑ higher is better"),
                }
                for key, (label, hint) in metric_map.items():
                    if key in m:
                        val = m[key]
                        fmt = f"{val:.4f}" if key != "mape" else f"{val:.2f}%"
                        st.metric(label, fmt, help=hint)

                st.caption(f"Model: **{model_type.upper()}**  v{data.get('model_version',1)}")
                st.caption(f"Run ID: `{data.get('run_id','—')[:8]}`")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend. Is it running?")
        except Exception as exc:
            st.exception(exc)

else:
    # Landing state
    st.info(
        "👈 Select a ticker and model from the sidebar, then click **Run Prediction**."
    )
    st.markdown("""
    ### How it works
    1. Latest market data is fetched from Yahoo Finance.
    2. 20+ technical indicators are computed (RSI, MACD, Bollinger Bands, ATR, OBV …).
    3. The LSTM processes a 60-day lookback window to predict the next closing price.
    4. XGBoost classifies the direction (UP / FLAT / DOWN) using the same features.
    5. A 90% confidence interval is derived from historical volatility.
    """)
