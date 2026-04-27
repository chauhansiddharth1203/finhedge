# FinHedge AI — User Manual

**Author:** Siddharth Chauhan | Roll No: CH21B103
**Course:** DA5402 — MLOps | IIT Madras
**Contact:** ch21b103@smail.iitm.ac.in | +91 9518241509

---

## What is FinHedge?

FinHedge is an AI-powered tool that helps you:
1. **Predict** whether a stock price will go **up, down, or stay flat** tomorrow.
2. **Recommend** a hedging action to protect your portfolio if a loss is predicted.

You don't need to write any code to use it.

---

## Getting Started

### Step 1: Start the Application

Open a Terminal and run:
```
cd finhedge
docker-compose up
```

Wait about 60 seconds for all services to start. Then open your web browser and go to:

**http://localhost:8501**

You will see the FinHedge home dashboard.

---

## The 5 Pages

### 🏠 Home (Dashboard)
This page shows you whether all the system components are working:
- **Backend API**: The AI engine. Should show 🟢 Online.
- **MLflow**: The experiment tracker. Should show 🟢 Connected.
- **Models**: Shows whether trained models are available.
- **Quick Start Guide**: Follow the numbered steps to get predictions.

---

### 🔮 Prediction Page

**What it does**: Predicts tomorrow's closing price for any stock.

**How to use it**:
1. In the left sidebar, select a stock ticker from the dropdown (e.g. `AAPL` for Apple).
   - Or type a custom ticker in the box (e.g. `RELIANCE.NS` for Reliance Industries).
2. Select the model: **LSTM** (recommended) or **XGBoost**.
3. Click the blue **🚀 Run Prediction** button.

**What you'll see**:
- **Current Price**: Today's closing price.
- **Predicted Price**: Tomorrow's forecast, with how much it will change.
- **Direction**: 🟢 UP / 🔴 DOWN / 🟡 FLAT — the expected movement.
- **Confidence**: How certain the model is (higher = more confident).
- **Candlestick Chart**: Shows recent price history with the forecast marked as a ⭐.
- **Model Metrics**: RMSE, Direction Accuracy, Sharpe Ratio — lower RMSE and higher accuracy = better model.

> 💡 **Tip**: Copy the predicted return (e.g. `-0.02` for a 2% drop) and paste it into the Hedging page.

---

### 🛡️ Hedging Page

**What it does**: Tells you how to protect your investment if the prediction is bearish (price expected to fall).

**How to use it**:
1. Enter the stock **Ticker** (same as what you used in Prediction).
2. Enter the **Current Price** (shown on the Prediction page).
3. Enter how many **Shares** you own.
4. Enter the **Predicted 1-day Return** from the Prediction page (e.g. `-0.02` for a 2% drop).
5. Click **🛡️ Get Hedge Recommendation**.

**What you'll see**:
- **Action Banner** (top):
  - 🔻 **HEDGE SHORT** → You should sell some shares short to protect against a fall.
  - 🔺 **HEDGE LONG** → Price is expected to rise; optionally buy more shares.
  - ⏸ **HOLD** → No action needed; risk is acceptable.
- **Hedge Ratio**: How much of your position to hedge (−1 = hedge everything, 0 = no hedge).
- **Hedge Quantity**: The exact number of shares to short/buy.
- **95% CVaR**: The worst-case expected loss at 95% confidence (in dollars).
- **P&L Chart**: Shows how your portfolio performs under different market moves, with and without the hedge.
- **Rationale**: A plain-English explanation of the recommendation.

---

### ⚙️ Pipeline Page

**What it does**: Lets you download data, train models, and monitor the pipeline.

**How to use it** (follow this order the first time):

1. Enter a **Ticker** in the sidebar (e.g. `AAPL`).
2. Click **📥 Run Ingestion** → Downloads 2 years of stock data.
3. Click **🔧 Run Preprocess** → Computes technical indicators and prepares data for training.
4. Click **🧠 Train Model** → Trains the LSTM model (takes 2–5 minutes).
5. Click **📊 Evaluate Model** → Computes performance metrics.

**The three tabs**:
- **Stage Status**: Shows which pipeline stages are complete (✅) or not yet run (⬜).
- **Airflow DAGs**: Shows the automated daily/weekly pipeline run history.
- **MLflow Runs**: Shows all past training experiments with their metrics. Click a run to see details.

---

### 📡 Monitoring Page

**What it does**: Shows live system health, model performance, and data quality.

**What you'll see**:
- **Total Predictions**: How many predictions have been made.
- **Error Rate**: Should stay below 5%. Red if something is wrong.
- **Model RMSE**: Prediction error. Lower is better.
- **Drift Alert**: 🟢 if data patterns are normal, 🔴 if the market has changed significantly (may need retraining).

**Grafana Dashboard tab**: Opens the full monitoring dashboard with charts for prediction rate, latency, drift scores, and hedge actions.

> 💡 **Auto-refresh**: Select a refresh interval (10s, 30s, 60s) in the sidebar to see live updates.

---

## Glossary

| Term | Simple Meaning |
|------|---------------|
| **LSTM** | A type of AI that learns patterns over time (good for sequences like stock prices) |
| **XGBoost** | A fast AI for classifying direction (UP/DOWN/FLAT) |
| **Hedge** | A trade that offsets potential losses in another trade |
| **CVaR** | "Conditional Value at Risk" — the average loss in the worst 5% of scenarios |
| **RMSE** | "Root Mean Square Error" — average prediction error in dollars |
| **Sharpe Ratio** | Risk-adjusted return (>1 is good, >2 is very good) |
| **Direction Accuracy** | % of times the model correctly predicts up vs down |
| **Data Drift** | When market conditions change so much that the model needs retraining |
| **MLflow** | The system that records all training experiments |
| **DVC** | The system that versions your data and model files |
| **Airflow** | The system that automatically runs the data pipeline on a schedule |
| **Prometheus** | Collects live metrics from the application |
| **Grafana** | Displays those metrics in visual dashboards |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect to backend" | Make sure `docker-compose up` is running. Wait 30 more seconds. |
| "Model not found" error | Go to the Pipeline page and click **Train Model** first. |
| Prediction page shows error | The ticker may be invalid or delisted. Try `AAPL`, `MSFT`, or `GOOGL`. |
| Airflow tab shows "offline" | Normal if Airflow is not needed right now — use the Pipeline controls instead. |
| Grafana iframe is blank | Click the **"Open Grafana ↗"** link to open it directly. Login: admin / finhedge123 |
| Training is very slow | Normal for the first run. Subsequent runs use cached data and are faster. |

---

## Service URLs

| Service | URL | Login |
|---------|-----|-------|
| FinHedge UI | http://localhost:8501 | — |
| API Documentation | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8081 | admin / admin |
| Grafana | http://localhost:3001 | admin / finhedge123 |
| Prometheus | http://localhost:9090 | — |
