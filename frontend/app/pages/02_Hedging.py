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
                # ── Gauge ─────────────────────────────────────────────────
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=rec["hedge_ratio"],
                    delta={
                        "reference": rec["delta_hedge_ref"],
                        "valueformat": ".3f",
                        "increasing": {"color": "#10b981"},
                        "decreasing": {"color": "#f43f5e"},
                        "prefix": "vs BS delta: ",
                    },
                    gauge={
                        "axis": {
                            "range": [-1, 1],
                            "tickwidth": 1,
                            "tickcolor": "#475569",
                            "tickvals": [-1, -0.5, 0, 0.5, 1],
                            "ticktext": ["-1<br>Full Short", "-0.5", "0<br>No Hedge", "+0.5", "+1<br>Full Long"],
                            "tickfont": {"color": "#94a3b8", "size": 10},
                        },
                        "bar":   {"color": banner_colour, "thickness": 0.25},
                        "bgcolor": "#0d1320",
                        "bordercolor": "#1e293b",
                        "steps": [
                            {"range": [-1,    -0.33], "color": "rgba(244,63,94,0.25)"},
                            {"range": [-0.33,  0.33], "color": "rgba(100,116,139,0.15)"},
                            {"range": [ 0.33,  1   ], "color": "rgba(16,185,129,0.25)"},
                        ],
                        "threshold": {
                            "line":      {"color": "#f59e0b", "width": 3},
                            "thickness": 0.85,
                            "value":     rec["delta_hedge_ref"],
                        },
                    },
                    title={
                        "text": f"CVaR Hedge Ratio — {ticker}<br><span style='font-size:0.75em;color:#64748b'>Yellow marker = Black-Scholes delta reference</span>",
                        "font": {"color": "#f1f5f9", "size": 14},
                    },
                    number={
                        "font":        {"color": banner_colour, "size": 36},
                        "valueformat": ".4f",
                    },
                ))
                fig_gauge.update_layout(
                    height=310,
                    paper_bgcolor="#111827",
                    font=dict(color="#94a3b8"),
                    margin=dict(l=20, r=20, t=60, b=10),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── P&L Chart ─────────────────────────────────────────────
                moves        = np.linspace(-0.12, 0.12, 400)
                pos_value    = current_price * position_size
                unhedged_pnl = moves * pos_value
                hedge_pnl    = -rec["hedge_ratio"] * rec["hedge_quantity"] * current_price * moves
                hedged_pnl   = unhedged_pnl + hedge_pnl - rec["cost_estimate"]
                pred_x       = predicted_ret * 100
                pred_u       = float(np.interp(predicted_ret, moves, unhedged_pnl))
                pred_h       = float(np.interp(predicted_ret, moves, hedged_pnl))

                fig_pnl = go.Figure()

                # Loss zone shading
                fig_pnl.add_hrect(
                    y0=min(min(unhedged_pnl), min(hedged_pnl)), y1=0,
                    fillcolor="rgba(244,63,94,0.05)", line_width=0,
                    annotation_text="Loss Zone", annotation_position="bottom left",
                    annotation_font_color="#f43f5e", annotation_font_size=10,
                )
                # Profit zone shading
                fig_pnl.add_hrect(
                    y0=0, y1=max(max(unhedged_pnl), max(hedged_pnl)),
                    fillcolor="rgba(16,185,129,0.05)", line_width=0,
                    annotation_text="Profit Zone", annotation_position="top left",
                    annotation_font_color="#10b981", annotation_font_size=10,
                )

                # Unhedged line
                fig_pnl.add_trace(go.Scatter(
                    x=moves * 100, y=unhedged_pnl,
                    fill="tozeroy",
                    fillcolor="rgba(244,63,94,0.10)",
                    line=dict(color="#f43f5e", width=2.5),
                    name="Unhedged Position",
                    hovertemplate="Move: %{x:.1f}%<br>P&L: $%{y:,.0f}<extra>Unhedged</extra>",
                ))

                # Hedged line
                fig_pnl.add_trace(go.Scatter(
                    x=moves * 100, y=hedged_pnl,
                    fill="tozeroy",
                    fillcolor="rgba(16,185,129,0.10)",
                    line=dict(color="#10b981", width=2.5),
                    name="Hedged (CVaR-Optimal)",
                    hovertemplate="Move: %{x:.1f}%<br>P&L: $%{y:,.0f}<extra>Hedged</extra>",
                ))

                # Breakeven line
                fig_pnl.add_hline(
                    y=0, line_color="#334155", line_width=1.5, line_dash="solid",
                )

                # Predicted return vertical
                fig_pnl.add_vline(
                    x=pred_x, line_dash="dash", line_color="#f59e0b", line_width=2,
                )
                fig_pnl.add_annotation(
                    x=pred_x, y=max(pred_u, pred_h),
                    text=f"Predicted<br>{pred_x:+.1f}%",
                    showarrow=True, arrowhead=2, arrowcolor="#f59e0b",
                    font=dict(color="#f59e0b", size=11),
                    bgcolor="#111827", bordercolor="#f59e0b", borderwidth=1,
                )

                # Mark predicted P&L dots
                fig_pnl.add_trace(go.Scatter(
                    x=[pred_x], y=[pred_u],
                    mode="markers",
                    marker=dict(size=10, color="#f43f5e", symbol="circle",
                                line=dict(color="#fff", width=2)),
                    name=f"Unhedged P&L at forecast: ${pred_u:,.0f}",
                    showlegend=True,
                ))
                fig_pnl.add_trace(go.Scatter(
                    x=[pred_x], y=[pred_h],
                    mode="markers",
                    marker=dict(size=10, color="#10b981", symbol="circle",
                                line=dict(color="#fff", width=2)),
                    name=f"Hedged P&L at forecast: ${pred_h:,.0f}",
                    showlegend=True,
                ))

                fig_pnl.update_layout(
                    title=dict(
                        text=f"<b>P&L Simulation — {ticker}</b>  |  Position: {position_size} shares @ ${current_price:.2f}",
                        font=dict(color="#f1f5f9", size=13),
                        x=0,
                    ),
                    plot_bgcolor="#0d1320",
                    paper_bgcolor="#111827",
                    font=dict(color="#94a3b8", size=12),
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis=dict(
                        title=dict(text="Stock Price Move (%)", font=dict(color="#64748b", size=11)),
                        gridcolor="#1e293b", zeroline=False,
                        tickcolor="#334155", tickfont=dict(color="#64748b"),
                        ticksuffix="%",
                    ),
                    yaxis=dict(
                        title=dict(text="Profit / Loss (USD)", font=dict(color="#64748b", size=11)),
                        gridcolor="#1e293b", zeroline=False,
                        tickcolor="#334155", tickfont=dict(color="#64748b"),
                        tickprefix="$", tickformat=",.0f",
                    ),
                    legend=dict(
                        orientation="v", x=1.01, y=1,
                        bgcolor="#111827", bordercolor="#1e293b", borderwidth=1,
                        font=dict(color="#94a3b8", size=10),
                    ),
                    hovermode="x unified",
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
