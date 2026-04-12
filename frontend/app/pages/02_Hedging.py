"""
02_Hedging.py – Deep hedging recommendation page.

Users enter their portfolio state and predicted return,
and receive an optimal hedge recommendation from the CVaR policy.
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Hedging | FinHedge", page_icon="🛡️", layout="wide")

st.markdown("## 🛡️ Deep Hedging Recommendation")
st.caption("CVaR-optimal hedge ratio powered by a deep MLP policy network.")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Portfolio State")

ticker         = st.sidebar.text_input("Ticker", value="AAPL").upper()
current_price  = st.sidebar.number_input("Current Price ($)", value=180.0, min_value=0.01, step=0.5)
position_size  = st.sidebar.number_input("Shares Held", value=100, min_value=1, step=10)
predicted_ret  = st.sidebar.slider(
    "Predicted 1-day Return", min_value=-0.10, max_value=0.10, value=0.01, step=0.001,
    format="%.3f",
)
time_fraction  = st.sidebar.slider(
    "Hedge Horizon Elapsed (%)", min_value=0, max_value=100, value=0, step=5,
)

hedge_btn = st.sidebar.button("🛡️ Get Hedge Recommendation", type="primary", use_container_width=True)

# ── Predict first button helper ────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("💡 **Tip**: Run Prediction first to get the predicted return, then paste it here.")

# ── Main ───────────────────────────────────────────────────────────────────
if hedge_btn:
    with st.spinner("Computing optimal hedge …"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/hedge",
                json={
                    "ticker":           ticker,
                    "current_price":    current_price,
                    "position_size":    float(position_size),
                    "predicted_return": predicted_ret,
                    "time_fraction":    time_fraction / 100.0,
                },
                timeout=30,
            )

            if resp.status_code != 200:
                st.error(f"API Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
                st.stop()

            rec = resp.json()

            # ── Action banner ─────────────────────────────────────────────
            action = rec["action"]
            banner_colour = {
                "HEDGE_SHORT": "#dc3545",
                "HEDGE_LONG":  "#28a745",
                "HOLD":        "#6c757d",
            }[action]
            action_icon = {
                "HEDGE_SHORT": "🔻 HEDGE SHORT",
                "HEDGE_LONG":  "🔺 HEDGE LONG",
                "HOLD":        "⏸ HOLD — No Action",
            }[action]

            st.markdown(
                f'<div style="background:{banner_colour};color:white;padding:1rem;'
                f'border-radius:10px;font-size:1.5rem;font-weight:700;text-align:center;">'
                f'{action_icon}</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            # ── Metric row ────────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Hedge Ratio",    f"{rec['hedge_ratio']:+.4f}")
            with c2:
                st.metric("Hedge Quantity", f"{rec['hedge_quantity']:.2f} shares")
            with c3:
                st.metric("95% CVaR Est.", f"${rec['cvar_95']:,.2f}")
            with c4:
                st.metric("Trans. Cost",   f"${rec['cost_estimate']:.4f}")
            with c5:
                st.metric("ΔS Delta Ref.", f"{rec['delta_hedge_ref']:.4f}")

            st.divider()

            col_left, col_right = st.columns([2, 1])

            with col_left:
                # ── Gauge chart for hedge ratio ───────────────────────────
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=rec["hedge_ratio"],
                    delta={"reference": 0, "valueformat": ".3f"},
                    gauge={
                        "axis":  {"range": [-1, 1], "tickwidth": 1},
                        "bar":   {"color": banner_colour},
                        "steps": [
                            {"range": [-1, -0.05], "color": "#ffcccc"},
                            {"range": [-0.05, 0.05], "color": "#f0f0f0"},
                            {"range": [0.05,  1],   "color": "#ccffcc"},
                        ],
                        "threshold": {
                            "line":  {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": rec["delta_hedge_ref"],
                        },
                    },
                    title={"text": "Hedge Ratio (−1=full short  0=no hedge  +1=full long)"},
                    number={"suffix": "  (deep model)", "valueformat": ".4f"},
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── P&L Simulation ────────────────────────────────────────
                st.subheader("Simulated Hedged vs Unhedged P&L")
                moves = np.linspace(-0.10, 0.10, 200)
                pos_value     = current_price * position_size
                unhedged_pnl  = moves * pos_value
                hedge_pnl     = -rec["hedge_ratio"] * rec["hedge_quantity"] * current_price * moves
                hedged_pnl    = unhedged_pnl + hedge_pnl - rec["cost_estimate"]

                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=moves * 100, y=unhedged_pnl,
                    name="Unhedged", line=dict(color="#dc3545", width=2),
                ))
                fig_pnl.add_trace(go.Scatter(
                    x=moves * 100, y=hedged_pnl,
                    name="Hedged", line=dict(color="#28a745", width=2, dash="dash"),
                ))
                fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_pnl.add_vline(x=predicted_ret * 100, line_dash="dash",
                                  line_color="#667eea", annotation_text="Predicted")
                fig_pnl.update_layout(
                    xaxis_title="Stock Move (%)",
                    yaxis_title="P&L ($)",
                    height=350,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

            with col_right:
                st.subheader("Rationale")
                st.info(rec["rationale"])

                st.subheader("Position Summary")
                pos_value = current_price * position_size
                st.markdown(f"""
| Field | Value |
|-------|-------|
| Ticker | **{ticker}** |
| Position | {position_size} shares |
| Position Value | **${pos_value:,.2f}** |
| Predicted Return | **{predicted_ret:+.2%}** |
| CVaR 95% | **${rec['cvar_95']:,.2f}** |
| Action | **{action}** |
| Hedge Qty | **{rec['hedge_quantity']:.2f}** |
| Delta Reference | **{rec['delta_hedge_ref']:.4f}** |
                """)

                st.subheader("Risk Reduction")
                if rec["hedge_ratio"] != 0:
                    reduction = abs(rec["hedge_ratio"]) * 100
                    st.progress(min(100, int(reduction)),
                                text=f"Estimated risk reduction: {reduction:.1f}%")
                else:
                    st.success("No hedge needed — position is at low risk.")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is it running?")
        except Exception as exc:
            st.exception(exc)

else:
    st.info("👈 Fill in your portfolio details in the sidebar and click **Get Hedge Recommendation**.")
    st.markdown("""
    ### How the hedger works
    - A **deep MLP policy network** maps your portfolio state → optimal hedge ratio.
    - Trained on **simulated GBM paths** to minimise **CVaR at 95%** confidence.
    - Accounts for **proportional transaction costs** (0.02% per trade).
    - Compared against the classical **Black-Scholes delta hedge** as a reference.

    ### Interpretation
    | Action | Meaning |
    |--------|---------|
    | 🔻 HEDGE SHORT | Short `hedge_quantity` shares to offset downside risk |
    | 🔺 HEDGE LONG  | Buy additional shares to capture predicted upside |
    | ⏸ HOLD         | Current risk is acceptable — no action needed |
    """)
