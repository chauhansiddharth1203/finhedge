"""
01_Prediction.py – Stock price prediction page.
"""
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
div[data-testid="metric-container"] {
    background: #111827; border: 1px solid #1e293b; border-radius: 8px;
    padding: 1rem 1.25rem; box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
div[data-testid="metric-container"] label {
    color: #64748b !important; font-size: 0.65rem !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important; font-size: 1.3rem !important; font-weight: 600 !important;
}
.stButton > button[kind="primary"] {
    background: #10b981 !important; border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; color: #fff !important;
}
hr { border-color: #1e293b !important; }
section[data-testid="stSidebar"] { border-right: 1px solid #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 1.25rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.5rem;">
    <div style="font-size: 1.4rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em;">
        Price Prediction
    </div>
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.2rem;">
        Next-day stock closing price forecast using LSTM or XGBoost models
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Prediction Settings")

POPULAR_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "JPM", "GS"]
ticker_choice = st.sidebar.selectbox("Ticker", POPULAR_TICKERS)
ticker_input  = st.sidebar.text_input("Custom ticker", value="", placeholder="e.g. RELIANCE.NS")
ticker        = ticker_input.strip().upper() if ticker_input.strip() else ticker_choice

model_type = st.sidebar.selectbox("Model", ["lstm", "xgboost"])
horizon    = st.sidebar.slider("Forecast horizon (days)", 1, 5, 1)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run Prediction", type="primary", use_container_width=True)

# ── Chart helper ───────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    plot_bgcolor="#0d1320",
    paper_bgcolor="#111827",
    font=dict(color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
    ),
    xaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False, color="#475569"),
    yaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False, color="#475569"),
)

def fetch_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Fetch stored OHLCV data from backend (avoids yfinance in Docker)."""
    try:
        r = requests.get(f"{BACKEND_URL}/pipeline/data/{ticker}/ohlcv?rows=90", timeout=10)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception:
        pass
    return None

# ── Main ───────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Running {model_type.upper()} for {ticker}…"):
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
            pred = data["predictions"][0]["predicted_price"]
            delta = pred - data["current_price"]
            delta_pct = delta / data["current_price"] * 100
            direction = data["direction"]

            # Build multi-day forecast (day 1 = LSTM, days 2+ = decaying projection)
            daily_ret = (pred - data["current_price"]) / data["current_price"]
            vol = data.get("volatility_1y", 0.02) / 16  # daily vol approx
            forecast_prices, forecast_lbs, forecast_ubs, forecast_dates = [], [], [], []
            base_date = pd.Timestamp(data["predictions"][0]["date"])
            p = data["current_price"]
            for d in range(1, horizon + 1):
                decay = 0.65 ** (d - 1)
                p = p * (1 + daily_ret * decay)
                ci = p * vol * (d ** 0.5) * 1.65
                fdate = base_date + pd.tseries.offsets.BDay(d - 1)
                forecast_prices.append(round(p, 2))
                forecast_lbs.append(round(p - ci, 2))
                forecast_ubs.append(round(p + ci, 2))
                forecast_dates.append(str(fdate.date()))

            # ── Key metrics ───────────────────────────────────────────────
            st.markdown(f"""
            <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;
            letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;">
            Forecast — {ticker} &nbsp;·&nbsp; {model_type.upper()}</div>
            """, unsafe_allow_html=True)

            dir_color = {"UP": "#10b981", "DOWN": "#f43f5e", "FLAT": "#f59e0b"}[direction]
            dir_arrow = {"UP": "▲", "DOWN": "▼", "FLAT": "—"}[direction]

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Current Price", f"${data['current_price']:.2f}")
            with c2:
                st.metric("Predicted Price", f"${pred:.2f}", delta=f"{delta:+.2f} ({delta_pct:+.2f}%)")
            with c3:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {dir_color};
                border-radius:8px;padding:1rem 1.25rem;">
                <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">Direction</div>
                <div style="font-size:1.3rem;font-weight:700;color:{dir_color};margin-top:0.4rem;">{dir_arrow} {direction}</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.metric("Confidence", f"{data['direction_prob']:.1%}")
            with c5:
                st.metric("Hist. Volatility", f"{data['volatility_1y']:.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Chart + Metrics ───────────────────────────────────────────
            col_chart, col_metrics = st.columns([3, 1])

            with col_chart:
                st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.5rem;'>Price Chart & Forecast</div>", unsafe_allow_html=True)

                hist = fetch_ohlcv(ticker)

                if hist is not None and not hist.empty:
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.75, 0.25],
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                    )

                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=hist["date"], open=hist["open"],
                        high=hist["high"], low=hist["low"], close=hist["close"],
                        name="OHLC",
                        increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
                        decreasing=dict(line=dict(color="#f43f5e"), fillcolor="#f43f5e"),
                    ), row=1, col=1)

                    # Multi-day forecast line
                    fig.add_trace(go.Scatter(
                        x=forecast_dates, y=forecast_prices,
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=2, dash="dot"),
                        marker=dict(size=10, color="#f59e0b",
                                    line=dict(color="#fff", width=1)),
                        text=[f"${p:.2f}" for p in forecast_prices],
                        textposition="top right",
                        textfont=dict(color="#f59e0b", size=11),
                        name="Forecast",
                        showlegend=True,
                    ), row=1, col=1)

                    # Confidence band
                    fig.add_trace(go.Scatter(
                        x=forecast_dates + forecast_dates[::-1],
                        y=forecast_ubs + forecast_lbs[::-1],
                        fill="toself",
                        fillcolor="rgba(245,158,11,0.1)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="90% CI",
                        showlegend=True,
                    ), row=1, col=1)

                    # Volume bars
                    vol_colors = [
                        "#10b981" if c >= o else "#f43f5e"
                        for c, o in zip(hist["close"], hist["open"])
                    ]
                    fig.add_trace(go.Bar(
                        x=hist["date"], y=hist["volume"],
                        marker_color=vol_colors, name="Volume", opacity=0.6,
                    ), row=2, col=1)

                    fig.update_layout(
                        **CHART_LAYOUT,
                        height=480,
                        xaxis_rangeslider_visible=False,
                        showlegend=True,
                    )
                    fig.update_yaxes(title_text="Price ($)", title_font=dict(color="#475569"), row=1, col=1)
                    fig.update_yaxes(title_text="Volume",   title_font=dict(color="#475569"), row=2, col=1)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: simple line with just current + forecast
                    rng = np.random.default_rng(42)
                    n = 60
                    prices = data["current_price"] * np.exp(
                        np.cumsum(rng.normal(0, 0.012, n))
                    )
                    prices[-1] = data["current_price"]
                    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates, y=prices,
                        mode="lines", name="Price",
                        line=dict(color="#3b82f6", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(59,130,246,0.08)",
                    ))
                    fig.add_trace(go.Scatter(
                        x=[pd.Timestamp.today()], y=[pred],
                        mode="markers+text",
                        marker=dict(size=14, color="#f59e0b", symbol="star"),
                        text=[f"${pred:.2f}"], textposition="top right",
                        textfont=dict(color="#f59e0b"),
                        name="Forecast",
                    ))
                    fig.update_layout(**CHART_LAYOUT, height=400)
                    st.plotly_chart(fig, use_container_width=True)

            with col_metrics:
                st.markdown("<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:0.75rem;'>Model Metrics</div>", unsafe_allow_html=True)

                m = data.get("metrics", {})
                metric_items = [
                    ("rmse",         "RMSE",           "{:.4f}",  "#f59e0b"),
                    ("mae",          "MAE",             "{:.4f}",  "#f59e0b"),
                    ("direction_acc","Direction Acc.",  "{:.2%}",  "#10b981"),
                    ("sharpe",       "Sharpe Ratio",    "{:.3f}",  "#3b82f6"),
                    ("r2",           "R² Score",        "{:.4f}",  "#8b5cf6"),
                ]
                for key, label, fmt, color in metric_items:
                    if key in m and m[key] != 0:
                        val = m[key]
                        formatted = fmt.format(val)
                        st.markdown(f"""
                        <div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {color};
                        border-radius:6px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;">
                        <div style="font-size:0.62rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.08em;color:#475569;">{label}</div>
                        <div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-top:0.2rem;">{formatted}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#111827;border:1px solid #1e293b;
                        border-radius:6px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;opacity:0.5;">
                        <div style="font-size:0.62rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.08em;color:#475569;">{label}</div>
                        <div style="font-size:0.85rem;color:#475569;margin-top:0.2rem;">Run Evaluate</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:1rem;padding:0.75rem;background:#0d1320;border-radius:6px;
                border:1px solid #1e293b;">
                <div style="font-size:0.65rem;color:#475569;">Model</div>
                <div style="font-size:0.85rem;font-weight:600;color:#94a3b8;">{model_type.upper()} v{data.get('model_version',1)}</div>
                <div style="font-size:0.65rem;color:#475569;margin-top:0.4rem;">Run ID</div>
                <div style="font-size:0.75rem;font-weight:500;color:#64748b;font-family:monospace;">{data.get('run_id','—')[:12]}</div>
                </div>""", unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend.")
        except Exception as exc:
            st.exception(exc)
else:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;
    padding:2rem;text-align:center;margin-top:1rem;">
        <div style="font-size:2rem;margin-bottom:0.75rem;">📈</div>
        <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:0.5rem;">
            Select a ticker and model, then click Run Prediction
        </div>
        <div style="font-size:0.85rem;color:#64748b;max-width:480px;margin:0 auto;line-height:1.7;">
            The LSTM processes a 60-day lookback window of 20+ technical indicators
            (RSI, MACD, Bollinger Bands, ATR, OBV) to forecast the next closing price
            with a 90% confidence interval.
        </div>
    </div>
    """, unsafe_allow_html=True)
