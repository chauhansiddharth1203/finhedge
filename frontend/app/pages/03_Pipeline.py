"""
03_Pipeline.py – ML Pipeline management console.

Shows:
  - DVC pipeline stage statuses
  - Airflow DAG run history (via Airflow REST API)
  - MLflow experiment runs
  - Controls to trigger pipeline stages and training
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

BACKEND_URL      = os.getenv("BACKEND_URL",  "http://localhost:8000")
AIRFLOW_API_URL  = os.getenv("AIRFLOW_URL",  "http://localhost:8080")   # container-to-container
AIRFLOW_LINK_URL = "http://localhost:8081"   # browser-facing URL
MLFLOW_URL       = os.getenv("MLFLOW_URL",   "http://localhost:5000")

AIRFLOW_AUTH = ("admin", "admin")

st.markdown("## ⚙️ ML Pipeline Console")
st.caption("Monitor and control the end-to-end FinHedge pipeline.")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Pipeline Controls")
ticker     = st.sidebar.text_input("Ticker", value="AAPL").upper()
period     = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
model_type = st.sidebar.selectbox("Model", ["lstm", "xgboost"])

st.sidebar.divider()
run_ingest  = st.sidebar.button("📥 Run Ingestion",   use_container_width=True)
run_preproc = st.sidebar.button("🔧 Run Preprocess",  use_container_width=True)
run_train   = st.sidebar.button("🧠 Train Model",     type="primary", use_container_width=True)
run_eval    = st.sidebar.button("📊 Evaluate Model",  use_container_width=True)

st.sidebar.divider()
epochs     = st.sidebar.slider("Epochs",     10, 200, 50, step=10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
lr         = st.sidebar.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)

# ── Stage trigger helper ───────────────────────────────────────────────────
def trigger_stage(stage: str) -> None:
    with st.spinner(f"Running **{stage}** for {ticker} …"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/pipeline/trigger",
                json={"ticker": ticker, "stage": stage, "model_type": model_type, "period": period},
                timeout=10,
            )
            if resp.status_code == 200:
                job = resp.json()
                st.success(f"Stage **{stage}** queued. Job ID: `{job['job_id']}`")
                # Poll for up to 30s
                for _ in range(30):
                    time.sleep(1)
                    jr = requests.get(f"{BACKEND_URL}/pipeline/jobs/{job['job_id']}", timeout=5)
                    if jr.status_code == 200:
                        jdata = jr.json()
                        if jdata["status"] in ("success", "failed"):
                            if jdata["status"] == "success":
                                st.success(f"✅ Stage **{stage}** completed in {jdata.get('duration_s', '?'):.1f}s")
                            else:
                                st.error(f"❌ Stage **{stage}** failed: {jdata.get('error', '?')}")
                            break
            else:
                st.error(f"Error: {resp.json().get('detail', resp.text)}")
        except Exception as e:
            st.error(f"Could not trigger stage: {e}")

if run_ingest:
    trigger_stage("ingest")
if run_preproc:
    trigger_stage("preprocess")
if run_eval:
    trigger_stage("evaluate")

if run_train:
    with st.spinner(f"Training **{model_type.upper()}** for {ticker} (this may take a few minutes) …"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/pipeline/train",
                json={"ticker": ticker, "model_type": model_type,
                      "epochs": epochs, "batch_size": batch_size,
                      "lr": lr, "period": period},
                timeout=600,
            )
            if resp.status_code == 200:
                result = resp.json()
                st.success(
                    f"✅ Training complete! Run ID: `{result['run_id'][:8]}`  "
                    f"RMSE: `{result['metrics'].get('rmse', 0):.4f}`  "
                    f"Sharpe: `{result['metrics'].get('sharpe', 0):.4f}`  "
                    f"Duration: `{result['duration_s']:.1f}s`"
                )
            else:
                st.error(f"Training failed: {resp.json().get('detail', resp.text)}")
        except Exception as e:
            st.error(f"Training error: {e}")

st.divider()

# ── Pipeline Status ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Stage Status", "🌊 Airflow DAGs", "🧪 MLflow Runs"])

with tab1:
    st.subheader(f"Pipeline Stages — {ticker}")
    try:
        r = requests.get(f"{BACKEND_URL}/pipeline/status/{ticker}", timeout=10)
        if r.status_code == 200:
            status = r.json()
            stages = status.get("stages", [])

            STATUS_ICON = {
                "success": "✅",
                "running": "⏳",
                "failed":  "❌",
                "idle":    "⬜",
            }

            # Pipeline flow diagram
            stage_names = [s["stage"] for s in stages]
            stage_statuses = [s["status"] for s in stages]

            cols = st.columns(len(stages))
            for i, (col, stage) in enumerate(zip(cols, stages)):
                with col:
                    icon = STATUS_ICON.get(stage["status"], "⬜")
                    colour = {
                        "success": "#28a745",
                        "running": "#ffc107",
                        "failed":  "#dc3545",
                        "idle":    "#6c757d",
                    }.get(stage["status"], "#6c757d")

                    st.markdown(
                        f'<div style="border:2px solid {colour};border-radius:10px;'
                        f'padding:1rem;text-align:center;">'
                        f'<div style="font-size:2rem;">{icon}</div>'
                        f'<div style="font-weight:bold;">{stage["stage"].upper()}</div>'
                        f'<div style="color:{colour};font-size:0.85rem;">{stage["status"]}</div>'
                        f'<div style="font-size:0.75rem;color:#888;">{stage.get("ended_at","—")[:19] if stage.get("ended_at") else "—"}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if i < len(stages) - 1:
                        pass  # arrow would go between cols

            if status.get("last_run_at"):
                st.caption(f"Last run: {status['last_run_at']}")
        else:
            st.warning(f"Could not fetch status: {r.status_code}")
    except Exception as e:
        st.warning(f"Backend unreachable: {e}")

with tab2:
    st.subheader("Airflow DAG Runs")
    st.markdown(f"[Open Airflow UI →]({AIRFLOW_LINK_URL})  (login: admin / admin)")
    try:
        dag_ids = ["finhedge_data_ingestion", "finhedge_model_retraining"]
        for dag_id in dag_ids:
            r = requests.get(
                f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns?limit=5&order_by=-execution_date",
                auth=AIRFLOW_AUTH, timeout=5,
            )
            if r.status_code == 200:
                runs = r.json().get("dag_runs", [])
                if runs:
                    df = pd.DataFrame([{
                        "Run ID":    run["dag_run_id"],
                        "State":     run["state"],
                        "Start":     run.get("start_date", "—")[:19],
                        "End":       run.get("end_date",   "—")[:19],
                    } for run in runs])

                    def colour_state(val):
                        c = {"success": "background-color:#d4edda",
                             "running": "background-color:#fff3cd",
                             "failed":  "background-color:#f8d7da"}.get(val, "")
                        return c

                    st.markdown(f"**{dag_id}**")
                    st.dataframe(
                        df.style.applymap(colour_state, subset=["State"]),
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.info(f"Airflow DAG `{dag_id}` not found or Airflow is offline.")
    except Exception:
        st.info("Airflow DAG history unavailable - but Airflow is running.")
        st.markdown(f"[Open Airflow UI →]({AIRFLOW_LINK_URL})  (admin / admin)")

with tab3:
    st.subheader("MLflow Experiment Runs")
    try:
        r = requests.get(f"{BACKEND_URL}/pipeline/runs?limit=20", timeout=10)
        if r.status_code == 200:
            runs = r.json()
            if runs:
                rows = []
                for run in runs:
                    rows.append({
                        "Run ID":   run["run_id"][:8],
                        "Name":     run.get("run_name", "—"),
                        "Status":   run["status"],
                        "RMSE":     run["metrics"].get("rmse", "—"),
                        "Sharpe":   run["metrics"].get("sharpe", "—"),
                        "Dir Acc":  run["metrics"].get("direction_acc", "—"),
                        "Model":    run["params"].get("model_type", "—"),
                        "Ticker":   run["params"].get("ticker", "—"),
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown(
                    f"[Open MLflow UI ↗]({MLFLOW_URL})",
                    unsafe_allow_html=False,
                )
            else:
                st.info("No runs found. Train a model to see experiment history.")
        else:
            st.warning("Could not fetch MLflow runs from backend.")
    except Exception as e:
        st.warning(f"MLflow not reachable: {e}")
