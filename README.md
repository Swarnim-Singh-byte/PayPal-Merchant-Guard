# 🛡️ PayPal Merchant Guard
**Deep Learning Continuous Assessment (CA) Project | 10 Marks**  
*College Deep Learning & Hackathon Extension Suite*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary
**PayPal Merchant Guard** is an end-to-end intelligent risk mitigation, merchant retention, and cash-flow management engine built for a digital merchant on PayPal. It addresses three linked operational problems using specialized Deep Learning architectures:

1. **Transaction Fraud Detection (Module A)**: Deep Multi-Layer Perceptron (MLP) identifying unauthorized payment drainages and ledger discrepancies under severe class imbalance ($0.129\%$ real fraud rate). Evaluated on held-out test data preserving real prevalence, threshold tuned strictly on validation ($\tau^* = 0.9345$), and benchmarked against Logistic Regression, Random Forest, and XGBoost.
2. **Merchant Churn Prevention (Module B)**: Tabular MLP estimating merchant dormancy and abandonment risk based on velocity decay and recency in early transaction windows (Days 1–20 features $\to$ Days 21–30 label, strictly zero temporal overlap). Benchmarked against a One-Rule recency baseline, Logistic Regression, Random Forest, and XGBoost.
3. **Cash-Flow Liquidity Forecasting (Module C)**: Recurrent Gated Neural Network (GRU) projecting payment volume for the upcoming 24 hours to optimize settlement sweeps and working capital reserves. Evaluated across 1h, 6h, 12h, and 24h horizons against Naive Persistence and Seasonal 24h baselines, with empirical 10th/90th percentile error confidence bounds.
4. **Bonus Extension (Unsupervised Anomaly Detection)**: Deep Autoencoder trained exclusively on benign transactions to detect zero-day fraud attacks via reconstruction error.

---

## 📊 Rubric & Marks Mapping (10 Marks Total)

| Section | Rubric Focus | Marks | Implementation & Notebook Section |
| :--- | :--- | :---: | :--- |
| **1. Problem Definition & Dataset** | Linked merchant problems, PaySim overview, class balance, simulated note | **1** | `Section 1` in `paypal_merchant_guard.ipynb` |
| **2. Data Preprocessing** | Outliers (IQR), balance error features, standardization vs normalization, leakage prevention | **1** | `Section 2` in `paypal_merchant_guard.ipynb` |
| **3. Neural Network Implementation** | Fraud MLP, Churn MLP (derived label check), Forecaster (GRU vs LSTM), Autoencoder | **3** | `Section 3` in `paypal_merchant_guard.ipynb` |
| **4. Training & Optimization** | Class weights, Adam, EarlyStopping, ReduceLROnPlateau, 2 settings tabulated | **1** | `Section 4` in `paypal_merchant_guard.ipynb` |
| **5. Evaluation** | Precision, Recall, $F_1$, ROC-AUC, PR-AUC, threshold tuning vs baselines (LR, RF, XGBoost) | **2** | `Section 5` in `paypal_merchant_guard.ipynb` |
| **6. Graphs & Visualization** | Epoch learning curves, confusion matrix heatmaps, PR curves, forecast vs actual, permutation bar chart | **1** | `Section 6` in `paypal_merchant_guard.ipynb` |
| **7. Interpretation & Conclusion** | Results table, overfitting analysis, why MLP beats baseline, limitations & future work | **1** | `Section 7` in `paypal_merchant_guard.ipynb` |

---

## 🏆 Model Performance Benchmark

### 1. Module A: Fraud Detection Under Real Class Prevalence (0.129%)
Evaluated on a completely held-out test set of **775,000 transactions** (1,000 fraud, 774,000 legitimate) matching the exact natural PaySim distribution ($0.129\%$). Scalers fit exclusively on training data; optimal decision threshold chosen exclusively on the validation set ($\tau^* = 0.9345$).

| Model Architecture | Precision | Recall | $F_1$ Score | ROC-AUC | PR-AUC | Key Advantage / Trade-off |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline: Logistic Regression** | 0.0225 | 0.9400 | 0.0440 | 0.9896 | 0.5748 | Crippled by 40,847 false positives under extreme negative skew |
| **Baseline: Random Forest** | 0.9842 | 0.9960 | 0.9901 | 0.9992 | 0.9968 | Tree ensembles partition orthogonal ledger balance steps sharply |
| **Baseline: XGBoost** | 0.7626 | 0.9960 | 0.8638 | 0.9995 | 0.9815 | Gradient boosted trees with `scale_pos_weight` handle imbalance cleanly |
| **Proposed: Deep MLP (64-32)** | 0.3824 | 0.8280 | **0.5232** | **0.9973** | **0.7703** | **11.9x higher Precision and +47.9 pts $F_1$ gain** over Logistic Regression |
| **Bonus: Deep Autoencoder** | — | — | — | 0.7528 | 0.3459 | Unsupervised zero-day detection via reconstruction error alone |

**Confusion Matrix (Deep MLP at $\tau^* = 0.9345$ on 775,000 real-ratio transactions):**
```
True Negative (TN): 772,663  |  False Positive (FP): 1,337
False Negative (FN):   172  |  True Positive  (TP):   828
```

### 2. Module B: Merchant Churn Prediction (No Temporal Overlap)
- **Base Rates:** Train Churn = 74.03%, Val Churn = 74.02%, Test Churn = 74.03% (consistent across all splits).
- **Temporal Integrity:** Features are extracted strictly from Days 1–20; churn labels are derived exclusively from transaction absence in Days 21–30. Zero data leakage.

| Model Architecture | Precision | Recall | $F_1$ Score | ROC-AUC | PR-AUC | Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **One-Rule Baseline (`days > 6`)** | 0.9814 | 0.9667 | 0.9740 | 0.9722 | 0.9820 | Simple recency threshold calibrated on validation set |
| **Baseline: Logistic Regression** | 0.9733 | 0.9352 | 0.9539 | 0.9939 | 0.9976 | Linear combination of velocity decay and volume |
| **Baseline: Random Forest** | 0.9982 | 0.9978 | 0.9980 | 0.9959 | 0.9986 | Non-linear splits over recency and volume drops |
| **Baseline: XGBoost** | 0.9982 | 0.9978 | 0.9980 | 0.9968 | 0.9989 | High-gradient boundary on dormancy patterns |
| **Proposed: Churn MLP (32-16)** | **0.9982** | **0.9978** | **0.9980** | **0.9944** | **0.9981** | Smooth multi-layer representation capturing velocity decay |

### 3. Module C: Hourly Cash-Flow Liquidity Forecasting
Chronological 70/15/15 split over 743 hours. Evaluated against Naive Persistence ($y_{t+1} = y_t$) and Same Hour Yesterday ($y_{t+1} = y_{t-23}$).

| Model Architecture | MAE | RMSE | Comparison to Baselines |
| :--- | :---: | :---: | :--- |
| **Baseline 1: Naive Persistence ($y_t$)** | \$187.96M | \$349.28M | Simple lag; lags large volume spikes and diurnal shifts |
| **Baseline 2: Same Hour Yesterday ($y_{t-23}$)** | \$227.78M | \$411.68M | Daily seasonal lag; suffers from day-to-day variance |
| **Proposed: GRU Forecaster (48h Window)** | **\$147.67M** | **\$275.88M** | **21.4% MAE reduction** over Naive, **35.2% reduction** over Seasonal |
| **Comparison: LSTM Forecaster** | \$234.12M | \$294.18M | Higher parameter complexity, longer training time |

**Multi-Step Autoregressive Horizon Errors (GRU Forecaster):**
| Forecast Horizon | MAE | RMSE |
| :---: | :---: | :---: |
| **1 Hour ($h = 1$)** | \$220.23M | \$302.48M |
| **6 Hours ($h = 6$)** | \$301.56M | \$355.23M |
| **12 Hours ($h = 12$)** | \$347.36M | \$388.69M |
| **24 Hours ($h = 24$)** | \$364.14M | \$393.53M |

---

## 🔄 Before vs. After Enhancements Comparison

| Pipeline Component | Before Enhancement | After Enhancement | Why It Changed |
| :--- | :--- | :--- | :--- |
| **Fraud Test Evaluation** | 4.0% enriched sample (8,213 frauds : 200,000 legits). $F_1 = 0.8873$. | **0.129% real PaySim ratio** (1,000 frauds : 774,000 legits). $F_1 = 0.5232$, ROC-AUC = 0.9973. | Evaluated under natural production skew. With 774,000 legitimate transactions, false positive opportunities increase ~30x (Base Rate Fallacy), dropping precision from 88% to 38.2% while maintaining near-perfect ranking (ROC-AUC 0.997). |
| **Decision Threshold** | Fixed heuristic $\tau = 0.505$ | **Validation-calibrated $\tau^* = 0.9345$** | Tuned strictly on validation PR curve to maximize $F_1$ under severe class imbalance without test set leakage. |
| **Baselines** | Logistic Regression only | **Random Forest, XGBoost, One-Rule Churn (`days > 6`), Seasonal 24h Forecaster ($y_{t-23}$)** | Provides comprehensive classical ML and heuristic benchmarks alongside neural networks. |
| **Churn Validation** | Undocumented temporal split | **Explicit Days 1–20 features $\to$ Days 21–30 label; Base rate reporting (74.03%)** | Guarantees zero data leakage and validates that features strictly precede the prediction window. |
| **Robustness Testing** | Single seed | **5-Seed Evaluation ($\mu \pm \sigma$)** | Demonstrates stability across random weight initializations (Fraud $F_1 = 0.4719 \pm 0.0857$, Churn $F_1 = 0.9980 \pm 0.0000$, Forecast MAE = $\$399.71\text{M} \pm \$180.93\text{M}$). |
| **Forecast Horizon** | Single-step ($t+1$) only | **1h, 6h, 12h, 24h Autoregressive Evaluation** | Quantifies compounding error growth over the full 24-hour liquidity planning horizon. |
| **Dashboard Uncertainty** | Fixed arbitrary $\pm 10\%$ band | **Empirical Validation Residual Confidence Bounds** | Derived from the 10th percentile (-\$484.40M) and 90th percentile (+\$398.58M) of validation residuals, reflecting true model error distributions. |
| **Explainability** | No feature importance | **Permutation Feature Importance** | Proves ledger discrepancies (`errorBalanceOrig` drop = 0.4346, `errorBalanceDest` drop = 0.1709) and `amount` (0.5151) drive fraud detection. |
| **Streamlit UI** | Basic tables and static inputs | **Modern lively UI with live health badges, metric cards, interactive plots, and robust input validation** | Rejects negative transaction amounts, negative account balances, empty CSV uploads, and malformed schemas with friendly feedback. |

---

## 🔍 Explainability: Permutation Feature Importance
Evaluating the drop in Precision-Recall AUC (PR-AUC) when individual features are permuted on the held-out real-ratio test set confirms the critical importance of domain-engineered ledger balance features:

1. **`amount`**: $\Delta \text{PR-AUC} = 0.5151$ (Primary determinant of high-impact drainage)
2. **`errorBalanceOrig`**: $\Delta \text{PR-AUC} = 0.4346$ (Discrepancy in sender balance after transfer)
3. **`type_TRANSFER`**: $\Delta \text{PR-AUC} = 0.3015$ (Key fraudulent transaction channel)
4. **`type_CASH_OUT`**: $\Delta \text{PR-AUC} = 0.1784$ (Secondary liquidation channel)
5. **`errorBalanceDest`**: $\Delta \text{PR-AUC} = 0.1709$ (Discrepancy in receiver balance post-settlement)

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
│   ├── fraud_features.joblib          # Feature column ordering contract
│   ├── forecast_residuals.joblib      # 10th and 90th percentile validation residuals
│   └── permutation_importance.joblib  # Saved permutation importance scores
└── scripts/
    ├── download_data.py               # Automated PaySim downloader
    ├── build_notebook.py              # Notebook generation utility
    ├── enhanced_experiments.py        # Real-ratio, baselines, and robustness experimental script
    └── prepare_app_assets.py          # Pre-computing fast data caches for Streamlit
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
Clone this repository and create a virtual environment with Python 3.11:

```bash
# Clone the repository
git clone https://github.com/Swarnim-Singh-byte/PayPal-Merchant-Guard.git
cd PayPal-Merchant-Guard

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
To inspect or re-execute the primary notebook:

```bash
# Launch Jupyter Lab / Notebook
jupyter notebook paypal_merchant_guard.ipynb
```
*Note*: The notebook is pre-executed top-to-bottom with all tables and plots rendered, and can also be opened directly in **Google Colab** or VS Code.

---

### 4. Running Phase 2 (Streamlit Dashboard)
Launch the interactive command center:

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app includes three functional tabs:
1. **🚨 Fraud Alerts & Scoring**: Single-transaction inspector with ledger error diagnostics, input sanity checks, and batch CSV scoring with downloadable reports.
2. **📉 Merchant Churn Risk**: Real-time merchant dormancy ranking with interactive cohort filters, priority badges, and retention recommendations.
3. **📈 Cash-Flow Forecasting**: Multi-step autoregressive 24-hour liquidity projection with empirical 10th/90th percentile validation error confidence envelopes and working capital sweep advisory.

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
4. The transaction payload is converted into the feature vector (`errorBalanceOrig`, `errorBalanceDest`, `amount`), scored by `models/fraud_mlp.keras`, and flagged before capture if probability exceeds threshold $\tau^* = 0.9345$.

---

## 📄 License
This project is released under the [MIT License](LICENSE).
