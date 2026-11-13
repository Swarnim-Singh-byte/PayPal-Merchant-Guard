# 🛡️ PayPal Merchant Guard
**Deep Learning Continuous Assessment (CA) Project | 10 Marks**  
*College Deep Learning & Hackathon Extension Suite*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary
**PayPal Merchant Guard** is an end-to-end intelligent risk mitigation, merchant retention, and cash-flow management engine built for a digital merchant on PayPal. It addresses three linked operational problems using specialized Deep Learning architectures:

1. **Transaction Fraud Detection (Module A)**: Deep Multi-Layer Perceptron (MLP) identifying unauthorized payment drainages and ledger discrepancies under severe class imbalance ($0.129\%$ fraud rate).
2. **Merchant Churn Prevention (Module B)**: Tabular MLP estimating merchant dormancy and abandonment risk based on velocity decay and recency in early transaction windows.
3. **Cash-Flow Liquidity Forecasting (Module C)**: Recurrent Gated Neural Network (GRU) projecting payment volume for the upcoming 24 hours to optimize settlement sweeps and working capital reserves.
4. **Bonus Extension (Unsupervised Anomaly Detection)**: Deep Autoencoder trained exclusively on benign transactions to detect zero-day fraud attacks via reconstruction error.

---

## 📊 Rubric & Marks Mapping (10 Marks Total)

| Section | Rubric Focus | Marks | Implementation & Notebook Section |
| :--- | :--- | :---: | :--- |
| **1. Problem Definition & Dataset** | Linked merchant problems, PaySim overview, class balance, simulated note | **1** | `Section 1` in `paypal_merchant_guard.ipynb` |
| **2. Data Preprocessing** | Outliers (IQR), balance error features, standardization vs normalization, leakage prevention | **1** | `Section 2` in `paypal_merchant_guard.ipynb` |
| **3. Neural Network Implementation** | Fraud MLP, Churn MLP (derived label check), Forecaster (GRU vs LSTM), Autoencoder | **3** | `Section 3` in `paypal_merchant_guard.ipynb` |
| **4. Training & Optimization** | Class weights, Adam, EarlyStopping, ReduceLROnPlateau, 2 settings tabulated | **1** | `Section 4` in `paypal_merchant_guard.ipynb` |
| **5. Evaluation** | Precision, Recall, $F_1$, ROC-AUC, PR-AUC, threshold tuning vs Logistic Regression & Naive baseline | **2** | `Section 5` in `paypal_merchant_guard.ipynb` |
| **6. Graphs & Visualization** | Epoch learning curves, confusion matrix heatmaps, PR curves, forecast vs actual | **1** | `Section 6` in `paypal_merchant_guard.ipynb` |
| **7. Interpretation & Conclusion** | Results table, overfitting analysis, why MLP beats baseline, limitations & future work | **1** | `Section 7` in `paypal_merchant_guard.ipynb` |

---

## 🏆 Model Performance Benchmark

| Module | Model Architecture | Primary Metric 1 | Primary Metric 2 | Baseline Comparison |
| :--- | :--- | :---: | :---: | :--- |
| **Module A — Fraud Detection** | Baseline: Logistic Regression | $F_1$: `0.5790` | ROC-AUC: `0.9891` | Fails on multi-variable balance mismatch interactions |
| **Module A — Fraud Detection** | **Proposed: Deep MLP (64-32)** | **$F_1$: `0.8873`** | **ROC-AUC: `0.9976`** | **+30.8% $F_1$ gain**; optimal threshold ($\tau^* = 0.505$) eliminates false alarms |
| **Module A — Bonus** | Unsupervised Autoencoder | ROC-AUC: `0.7528` | PR-AUC: `0.3459` | Zero-day fraud detection via reconstruction error without fraud labels |
| **Module B — Merchant Churn** | Baseline: Logistic Regression | $F_1$: `0.9539` | ROC-AUC: `0.9939` | Standard linear weighting of recency and payment counts |
| **Module B — Merchant Churn** | **Proposed: Churn MLP (32-16)** | **$F_1$: `0.9980`** | **ROC-AUC: `0.9943`** | Captures non-linear interactions between activity days and volume decay |
| **Module C — Volume Forecast** | Baseline: Naive Persistence | MAE: `\$187.96M` | RMSE: `\$349.28M` | Simple persistence lag $y_{t+1} = y_t$; lags sudden volume shifts |
| **Module C — Volume Forecast** | **Proposed: GRU Forecaster** | **MAE: `\$147.67M`** | **RMSE: `\$275.88M`** | **21.4% MAE reduction** over baseline; captures diurnal payment cycles |
| **Module C — Volume Forecast** | Comparison: LSTM Forecaster | MAE: `\$234.12M` | RMSE: `\$294.18M` | Long-term gating; slightly higher parameter count than GRU |

---

## 📂 Project Directory Structure

```
PAYSIM/
├── app.py                             # Phase 2: Interactive Streamlit Web Dashboard
├── paypal_merchant_guard.ipynb        # Phase 1: Primary Evaluated Notebook (Runs top-to-bottom)
├── requirements.txt                   # Complete dependencies list
├── .gitignore                         # Git exclusion rules (protects credentials & large data)
├── LICENSE                            # MIT License
├── README.md                          # Documentation and instructions
├── data/
│   ├── PS_20174392719_1491204439457_log.csv  # Full PaySim Dataset (6.36M rows)
│   ├── sample_transactions.csv        # Pre-computed curated batch for Streamlit testing
│   ├── merchant_cohort_sample.csv     # Scored merchant behavior cohort
│   └── recent_hourly_volumes.csv      # Recent 96 hours of volume data
├── models/
│   ├── fraud_mlp.keras                # Saved Fraud Deep MLP
│   ├── churn_mlp.keras                # Saved Churn MLP
│   ├── forecaster.keras               # Saved Forecaster GRU
│   ├── autoencoder_fraud.keras        # Saved Anomaly Autoencoder
│   ├── fraud_scaler.joblib            # Fitted StandardScaler for fraud features
│   ├── churn_scaler.joblib            # Fitted StandardScaler for churn features
│   ├── forecast_scaler.joblib         # Fitted MinMaxScaler for volume sequences
│   └── fraud_features.joblib          # Feature column ordering contract
└── scripts/
    ├── download_data.py               # Automated PaySim downloader
    ├── build_notebook.py              # Notebook generation utility
    ├── train_and_evaluate.py          # Standalone training & pipeline verification
    └── prepare_app_assets.py          # Pre-computing fast data caches for Streamlit
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
Clone this repository and create a virtual environment with Python 3.11:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Preparation
If you do not have the PaySim CSV in `data/`, run the automated downloader:
```bash
python scripts/download_data.py
```
This fetches the official PaySim dataset (6.36M rows) and extracts it directly to `data/PS_20174392719_1491204439457_log.csv`.

---

### 3. Running Phase 1 (Jupyter Notebook)
To inspect or re-execute the notebook:

```bash
# Launch Jupyter Lab / Notebook
jupyter notebook paypal_merchant_guard.ipynb
```
*Note*: The notebook is pre-executed and can also be opened directly in **Google Colab** or VS Code.

---

### 4. Running Phase 2 (Streamlit Dashboard)
Launch the interactive command center:

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app includes three functional tabs:
1. **🚨 Fraud Alerts & Scoring**: Single-transaction inspector with ledger error diagnostics and batch CSV scoring.
2. **📉 Merchant Churn Risk**: Real-time merchant dormancy ranking with interactive filters and retention recommendations.
3. **📈 Cash-Flow Forecasting**: Multi-step autoregressive 24-hour liquidity projection with uncertainty envelopes.

---

## 🛠️ Phase 3 Hackathon Extension: Live PayPal Sandbox & LLM Explanations

To connect this pipeline to live authorization flows:
1. Create a PayPal Developer Sandbox account at [developer.paypal.com](https://developer.paypal.com).
2. Create a `.env` file in the root directory (never commit this file):
   ```ini
   PAYPAL_CLIENT_ID="your_sandbox_client_id_here"
   PAYPAL_CLIENT_SECRET="your_sandbox_client_secret_here"
   PAYPAL_ENV="sandbox"
   ```
3. Use the PayPal Orders V2 REST API (`https://api-m.sandbox.paypal.com/v2/checkout/orders`) to intercept transaction authorization webhooks.
4. The transaction payload is converted into the feature vector (`errorBalanceOrig`, `errorBalanceDest`, `amount`), scored by `models/fraud_mlp.keras`, and flagged before capture if probability exceeds threshold $\tau^* = 0.50$.

---

## 📄 License
This project is released under the [MIT License](LICENSE).
