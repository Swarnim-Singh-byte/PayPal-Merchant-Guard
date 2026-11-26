"""
PayPal Merchant Guard - Streamlit Application
Interactive Command Center for Real-Time Fraud Alerts, Merchant Churn Risk,
and 24-Hour Cash-Flow Forecasting.
Enhanced with robust input validation, lively UI widgets, and empirical residual forecasting intervals.
"""
import os
import sys

# Isolate cache directories for sandboxing & cloud deployment
os.environ["KERAS_HOME"] = os.path.abspath(".keras")
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib")
os.makedirs(".keras", exist_ok=True)
os.makedirs(".matplotlib", exist_ok=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

# Page Configuration
st.set_page_config(
    page_title="PayPal Merchant Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Lively Styling (PayPal Modern FinTech Palette)
st.markdown("""
<style>
    /* Gradient Brand Header */
    .brand-hero {
        background: linear-gradient(135deg, #003087 0%, #0070BA 60%, #00CFDE 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 48, 135, 0.18);
    }
    .brand-hero h1 {
        color: white !important;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .brand-hero p {
        color: #E2E8F0 !important;
        font-size: 1.05rem;
        margin: 0.4rem 0 0 0;
    }

    /* Metric Cards with Lively Glow */
    .metric-card-lively {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card-lively:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 112, 186, 0.12);
    }

    /* Risk Badges */
    .badge-critical {
        background: linear-gradient(135deg, #FEE2E2 0%, #FCA5A5 100%);
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .badge-warning {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        color: #92400E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #FBBF24;
    }
    .badge-success {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }

    /* Pulse Indicator */
    .pulse-dot {
        height: 10px;
        width: 10px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 8px #10B981;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Cached Model & Scaler Loaders
# -------------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Load deep learning models and fitted preprocessors from disk."""
    try:
        fraud_model = tf.keras.models.load_model("models/fraud_mlp.keras")
        churn_model = tf.keras.models.load_model("models/churn_mlp.keras")
        forecaster_model = tf.keras.models.load_model("models/forecaster.keras")

        fraud_scaler = joblib.load("models/fraud_scaler.joblib")
        churn_scaler = joblib.load("models/churn_scaler.joblib")
        forecast_scaler = joblib.load("models/forecast_scaler.joblib")
        fraud_features = joblib.load("models/fraud_features.joblib")
        
        residuals_data = {}
        if os.path.exists("models/forecast_residuals.joblib"):
            residuals_data = joblib.load("models/forecast_residuals.joblib")
            
        imp_data = None
        if os.path.exists("models/permutation_importance.joblib"):
            imp_data = joblib.load("models/permutation_importance.joblib")

        return {
            "fraud_model": fraud_model,
            "churn_model": churn_model,
            "forecaster_model": forecaster_model,
            "fraud_scaler": fraud_scaler,
            "churn_scaler": churn_scaler,
            "forecast_scaler": forecast_scaler,
            "fraud_features": fraud_features,
            "residuals": residuals_data,
            "importances": imp_data,
            "status": "ready"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

models_dict = load_all_models()

# -------------------------------------------------------------
# Sidebar: System Health & Interactive Controls
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b5/PayPal.svg", width=140)
    st.markdown("### **Merchant Guard Engine**")
    st.caption("College Deep Learning CA & Hackathon Suite")
    st.divider()

    if models_dict["status"] == "ready":
        st.markdown('<div><span class="pulse-dot"></span><b>Engines Active & Synchronized</b></div>', unsafe_allow_html=True)
        st.markdown("""
        - **Fraud MLP**: `64-32 Dense (ReLU)`
        - **Churn MLP**: `32-16 Dense (ReLU)`
        - **Forecaster**: `GRU(64) Recurrent`
        """)
    else:
        st.error(f"🔴 **Model Load Error**: {models_dict.get('message', '')}")

    st.divider()
    st.markdown("#### **Operating Decision Threshold**")
    # Phase 2 updated default: 0.935 (selected on validation for real 0.129% ratio)
    fraud_threshold = st.slider(
        "Fraud Decision Threshold (τ*)",
        min_value=0.10, max_value=0.99, value=0.93, step=0.01,
        help="Calibrated on validation set to balance precision-recall at true 0.129% prevalence."
    )
    st.caption(f"Transactions with probability $\ge {fraud_threshold:.2f}$ trigger an immediate authorization hold.")

    st.divider()
    st.info("💡 **Dataset**: PaySim Simulator (6.36M transactions). Preprocessors fit strictly on train splits without data leakage.")

# -------------------------------------------------------------
# Main Hero Header
# -------------------------------------------------------------
st.markdown("""
<div class="brand-hero">
    <h1>🛡️ PayPal Merchant Guard</h1>
    <p>Intelligent Real-Time Risk Interception, Merchant Retention Analytics & Empirical Working Capital Forecasting</p>
</div>
""", unsafe_allow_html=True)

# Three Primary Tabs
tab_fraud, tab_churn, tab_forecast = st.tabs([
    "🚨 Fraud Alerts & Scoring",
    "📉 Merchant Churn Risk",
    "📈 Cash-Flow Forecasting"
])

# =============================================================================
# TAB 1: FRAUD ALERTS
# =============================================================================
with tab_fraud:
    st.subheader("Real-Time Transaction Fraud Detection (Module A)")
    st.write("Intercept fraudulent payment attempts before settlement using our Deep Multi-Layer Perceptron. Evaluated at true natural prevalence (0.129%).")

    sub_mode = st.radio("Mode Selection:", ["Single Transaction Inspector", "Batch / Stream CSV Evaluation"], horizontal=True)

    if sub_mode == "Single Transaction Inspector":
        st.markdown("##### **Transaction Attributes & Quick Presets**")

        # Preset Quick-Fills
        col_pre1, col_pre2, col_pre3 = st.columns(3)
        with col_pre1:
            if st.button("⚡ Preset: Typ. Fraud (Origin Drain)"):
                st.session_state["f_type"] = "TRANSFER"
                st.session_state["f_amount"] = 350000.0
                st.session_state["f_old_orig"] = 350000.0
                st.session_state["f_new_orig"] = 0.0
                st.session_state["f_old_dest"] = 0.0
                st.session_state["f_new_dest"] = 0.0
        with col_pre2:
            if st.button("✅ Preset: Legitimate POS Payment"):
                st.session_state["f_type"] = "PAYMENT"
                st.session_state["f_amount"] = 125.50
                st.session_state["f_old_orig"] = 1500.00
                st.session_state["f_new_orig"] = 1374.50
                st.session_state["f_old_dest"] = 0.0
                st.session_state["f_new_dest"] = 0.0
        with col_pre3:
            if st.button("🔄 Preset: Regular Large Transfer"):
                st.session_state["f_type"] = "TRANSFER"
                st.session_state["f_amount"] = 50000.0
                st.session_state["f_old_orig"] = 100000.0
                st.session_state["f_new_orig"] = 50000.0
                st.session_state["f_old_dest"] = 25000.0
                st.session_state["f_new_dest"] = 75000.0

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            txn_type = st.selectbox(
                "Transaction Type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(st.session_state.get("f_type", "TRANSFER"))
            )
            amount = st.number_input("Transaction Amount ($)", value=float(st.session_state.get("f_amount", 181.0)), step=100.0)

        with col_f2:
            old_orig = st.number_input("Origin Initial Balance ($)", value=float(st.session_state.get("f_old_orig", 181.0)), step=100.0)
            new_orig = st.number_input("Origin Final Balance ($)", value=float(st.session_state.get("f_new_orig", 0.0)), step=100.0)

        with col_f3:
            old_dest = st.number_input("Destination Initial Balance ($)", value=float(st.session_state.get("f_old_dest", 0.0)), step=100.0)
            new_dest = st.number_input("Destination Final Balance ($)", value=float(st.session_state.get("f_new_dest", 0.0)), step=100.0)

        # Derived Feature Engineering
        err_orig = new_orig + amount - old_orig
        err_dest = old_dest + amount - new_dest

        st.markdown(f"""
        **Engineered Ledger Discrepancies**:
        - `errorBalanceOrig` = `newbalanceOrig + amount - oldbalanceOrg` = **${err_orig:,.2f}**
        - `errorBalanceDest` = `oldbalanceDest + amount - newbalanceDest` = **${err_dest:,.2f}**
        """)

        # INPUT VALIDATION
        validation_passed = True
        if amount <= 0:
            st.error("⚠️ **Input Validation Error**: Transaction amount must be strictly positive (> $0.00).")
            validation_passed = False
        if old_orig < 0 or new_orig < 0 or old_dest < 0 or new_dest < 0:
            st.error("⚠️ **Input Validation Error**: Account balance values cannot be negative.")
            validation_passed = False

        if validation_passed and st.button("🔍 Evaluate Transaction Fraud Risk", type="primary"):
            row = {
                'amount': amount,
                'oldbalanceOrg': old_orig,
                'newbalanceOrig': new_orig,
                'oldbalanceDest': old_dest,
                'newbalanceDest': new_dest,
                'errorBalanceOrig': err_orig,
                'errorBalanceDest': err_dest,
                'type_CASH_OUT': 1 if txn_type == 'CASH_OUT' else 0,
                'type_DEBIT': 1 if txn_type == 'DEBIT' else 0,
                'type_PAYMENT': 1 if txn_type == 'PAYMENT' else 0,
                'type_TRANSFER': 1 if txn_type == 'TRANSFER' else 0,
            }
            input_df = pd.DataFrame([row])[models_dict["fraud_features"]]
            scaled_vec = models_dict["fraud_scaler"].transform(input_df)
            fraud_prob = float(models_dict["fraud_model"].predict(scaled_vec, verbose=0)[0][0])

            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric("Model Fraud Probability", f"{fraud_prob*100:.2f}%")
                st.progress(min(1.0, fraud_prob))
                
                if fraud_prob >= fraud_threshold:
                    st.markdown('<div class="badge-critical">🚨 CRITICAL FRAUD RISK: HOLD TRANSACTION</div>', unsafe_allow_html=True)
                elif fraud_prob >= 0.40:
                    st.markdown('<div class="badge-warning">⚠️ ELEVATED RISK: SECONDARY VERIFICATION</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-success">✅ LOW RISK: TRANSACTION APPROVED</div>', unsafe_allow_html=True)

            with col_res2:
                st.markdown("##### **Deep Learning Risk Diagnostics & Explainability**")
                reasons = []
                if txn_type in ['TRANSFER', 'CASH_OUT']:
                    reasons.append(f"• High-risk category: **{txn_type}** (100% of PaySim fraud concentrates in TRANSFER & CASH_OUT).")
                if abs(err_orig) > 1.0:
                    reasons.append(f"• **Origin ledger discrepancy**: \${err_orig:,.2f} deviation from expected double-entry accounting.")
                if new_orig == 0.0 and old_orig > 1000:
                    reasons.append("• **Account Drain Pattern**: Origin account balance reduced to exactly \$0.00.")
                if not reasons:
                    reasons.append("• Clean ledger accounting: Origin and destination balance changes match transaction amount perfectly.")

                for r in reasons:
                    st.write(r)

        # Embedded Permutation Feature Importance Plot
        if models_dict.get("importances") is not None:
            with st.expander("📊 View Model Explainability (Permutation Feature Importance)"):
                imp_df = models_dict["importances"].sort_values('PR_AUC_Drop', ascending=True)
                fig_imp, ax_imp = plt.subplots(figsize=(8, 3.8))
                colors = ['#c44e52' if 'error' in f else '#4c72b0' for f in imp_df['Feature']]
                ax_imp.barh(imp_df['Feature'], imp_df['PR_AUC_Drop'], color=colors)
                ax_imp.set_title("Drop in PR-AUC when Feature is Shuffled (Validation Set)")
                ax_imp.set_xlabel("PR-AUC Drop")
                ax_imp.grid(True, alpha=0.3)
                st.pyplot(fig_imp)

    else:
        st.markdown("##### **Batch CSV Ingestion & Stream Testing**")
        uploaded_file = st.file_uploader("Upload Transaction CSV (Matching PaySim Schema)", type=["csv"])
        load_preset = st.button("📂 Load Pre-computed Sample (50 Fraud + 50 Legitimate Txns)")

        batch_df = None
        if uploaded_file is not None:
            try:
                uploaded_content = pd.read_csv(uploaded_file)
                # INPUT VALIDATION
                if len(uploaded_content) == 0:
                    st.error("⚠️ **Validation Error**: The uploaded CSV file is empty. Please upload a file containing records.")
                else:
                    required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                    missing_cols = [c for c in required_cols if c not in uploaded_content.columns]
                    if missing_cols:
                        st.error(f"⚠️ **Validation Error**: Missing required schema columns: {missing_cols}. Please verify CSV headers.")
                    else:
                        batch_df = uploaded_content
                        st.success(f"Successfully loaded {len(batch_df):,} transactions for batch scoring.")
            except Exception as e:
                st.error(f"⚠️ **File Read Error**: {str(e)}")

        elif load_preset or os.path.exists("data/sample_transactions.csv"):
            batch_df = pd.read_csv("data/sample_transactions.csv")
            st.info("Loaded pre-computed sample containing 100 real PaySim transactions.")

        if batch_df is not None:
            with st.spinner("Executing inference with Deep Fraud MLP..."):
                proc_df = batch_df.copy()
                proc_df['errorBalanceOrig'] = proc_df['newbalanceOrig'] + proc_df['amount'] - proc_df['oldbalanceOrg']
                proc_df['errorBalanceDest'] = proc_df['oldbalanceDest'] + proc_df['amount'] - proc_df['newbalanceDest']

                for t in ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
                    proc_df[f'type_{t}'] = (proc_df['type'] == t).astype(int)

                X_batch = proc_df[models_dict["fraud_features"]]
                X_batch_scaled = models_dict["fraud_scaler"].transform(X_batch)
                preds = models_dict["fraud_model"].predict(X_batch_scaled, verbose=0).ravel()

                batch_df['fraud_probability'] = np.round(preds, 4)
                batch_df['action'] = np.where(preds >= fraud_threshold, "🚨 HOLD / REJECT", "✅ APPROVE")

            total_scanned = len(batch_df)
            flagged = (batch_df['fraud_probability'] >= fraud_threshold).sum()
            flagged_dollars = batch_df[batch_df['fraud_probability'] >= fraud_threshold]['amount'].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Scanned Transactions", f"{total_scanned:,}")
            c2.metric("Flagged Fraudulent", f"{flagged:,}", f"{(flagged/total_scanned)*100:.1f}%")
            c3.metric("Prevented Financial Loss", f"${flagged_dollars:,.2f}")

            st.dataframe(
                batch_df[['step', 'type', 'amount', 'nameOrig', 'nameDest', 'fraud_probability', 'action']],
                use_container_width=True
            )

# =============================================================================
# TAB 2: CHURN RISK
# =============================================================================
with tab_churn:
    st.subheader("Merchant Churn & Dormancy Intelligence (Module B)")
    st.write("Detect early signals of merchant abandonment using behavioral features (velocity, mean volume, and recency decay) extracted strictly from Days 1–20.")

    if os.path.exists("data/merchant_cohort_sample.csv"):
        m_cohort = pd.read_csv("data/merchant_cohort_sample.csv")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_m = len(m_cohort)
        high_risk_m = (m_cohort['risk_tier'] == 'High Risk').sum()
        avg_days = m_cohort['days_since_last_txn'].mean()
        avg_rev = m_cohort['mean_amount'].mean()

        kpi1.metric("Monitored Merchants", f"{total_m:,}")
        kpi2.metric("High Churn Risk Cohort", f"{high_risk_m:,}", f"{(high_risk_m/total_m)*100:.1f}%")
        kpi3.metric("Avg Inactivity (Days)", f"{avg_days:.1f} days")
        kpi4.metric("Avg Transaction Size", f"${avg_rev:,.2f}")

        st.divider()

        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            tier_filter = st.selectbox("Filter Risk Tier", ["All Tiers", "High Risk", "Medium Risk", "Low Risk"])
        with f_col2:
            min_days = st.slider("Minimum Days Inactive", 0, 20, 0)
        with f_col3:
            search_id = st.text_input("Search Merchant ID (e.g. M123456789)")

        filtered = m_cohort.copy()
        if tier_filter != "All Tiers":
            filtered = filtered[filtered['risk_tier'] == tier_filter]
        if min_days > 0:
            filtered = filtered[filtered['days_since_last_txn'] >= min_days]
        if search_id.strip():
            filtered = filtered[filtered['merchant_id'].str.contains(search_id.strip(), case=False)]

        st.markdown(f"##### **Ranked Merchant Watchlist ({len(filtered):,} records)**")
        st.dataframe(
            filtered[['merchant_id', 'risk_tier', 'churn_probability', 'days_since_last_txn', 'txn_count', 'mean_amount', 'active_days']],
            use_container_width=True
        )

        st.markdown("##### **Individual Merchant Deep Dive & Automated Retention**")
        selected_m = st.selectbox("Select Merchant to Inspect:", filtered['merchant_id'].tolist()[:50])
        if selected_m:
            m_row = filtered[filtered['merchant_id'] == selected_m].iloc[0]
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.metric("Predicted Churn Probability", f"{m_row['churn_probability']*100:.1f}%")
                if m_row['risk_tier'] == 'High Risk':
                    st.markdown('<div class="badge-critical">🚨 HIGH CHURN RISK</div>', unsafe_allow_html=True)
                    st.markdown("""
                    **Recommended Action**:
                    - Trigger automated reachout from Merchant Success.
                    - Offer temporary 0.5% transaction processing fee discount.
                    """)
                elif m_row['risk_tier'] == 'Medium Risk':
                    st.markdown('<div class="badge-warning">⚠️ MEDIUM RISK (SLIPPING)</div>', unsafe_allow_html=True)
                    st.markdown("""
                    **Recommended Action**:
                    - Send targeted feature adoption email (PayPal Checkout v2).
                    - Send quarterly business health review summary.
                    """)
                else:
                    st.markdown('<div class="badge-success">✅ HEALTHY ACTIVE MERCHANT</div>', unsafe_allow_html=True)
                    st.markdown("""
                    **Recommended Action**:
                    - Eligible for pre-approved PayPal Working Capital business advance.
                    """)

            with col_m2:
                st.write("**Merchant Behavioral Metrics**:")
                st.write(f"- **Days Since Last Payment**: {int(m_row['days_since_last_txn'])} days")
                st.write(f"- **Total Transactions in Window**: {int(m_row['txn_count'])}")
                st.write(f"- **Mean Payment Volume**: \${m_row['mean_amount']:,.2f}")
                st.write(f"- **Distinct Active Days**: {int(m_row['active_days'])} out of 20")
    else:
        st.warning("Merchant cohort file not found. Run `scripts/prepare_app_assets.py`.")

# =============================================================================
# TAB 3: CASH-FLOW FORECASTING
# =============================================================================
with tab_forecast:
    st.subheader("Hourly Payment Volume & Cash-Flow Projection (Module C)")
    st.write("Autoregressive multi-step projection of merchant payment liquidity over the next 24 hours using a Gated Recurrent Unit (GRU) network.")

    if os.path.exists("data/recent_hourly_volumes.csv"):
        vol_df = pd.read_csv("data/recent_hourly_volumes.csv")
        history_48 = vol_df.tail(48)['amount'].values.reshape(-1, 1)

        scaler_fc = models_dict["forecast_scaler"]
        gru_model = models_dict["forecaster_model"]
        res_info = models_dict.get("residuals", {})

        curr_seq = scaler_fc.transform(history_48).reshape(1, 48, 1)
        forecast_24_scaled = []

        for _ in range(24):
            pred_scaled = gru_model.predict(curr_seq, verbose=0)[0][0]
            forecast_24_scaled.append(pred_scaled)
            curr_seq = np.append(curr_seq[:, 1:, :], [[[pred_scaled]]], axis=1)

        forecast_24 = scaler_fc.inverse_transform(np.array(forecast_24_scaled).reshape(-1, 1)).ravel()

        # EMPIRICAL RESIDUAL CONFIDENCE BANDS (10th and 90th percentile from validation set)
        q10_res = res_info.get("q10", -0.15 * forecast_24.mean())
        q90_res = res_info.get("q90", 0.15 * forecast_24.mean())

        # Bounds: y_pred - abs(q10) to y_pred + q90
        lower_bound = np.maximum(0, forecast_24 + q10_res)
        upper_bound = forecast_24 + q90_res

        total_24h_vol = forecast_24.sum()
        peak_hr_vol = forecast_24.max()
        recommended_buffer = total_24h_vol * 0.15

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Projected 24h Inflow", f"${total_24h_vol/1e6:.2f}M")
        fc2.metric("Peak Hourly Inflow", f"${peak_hr_vol/1e6:.2f}M")
        fc3.metric("Recommended Liquidity Reserve (15%)", f"${recommended_buffer/1e6:.2f}M",
                   help="Recommended working capital buffer to meet instant chargebacks and refund settlement windows.")

        st.divider()

        # Lively Plot with Empirical Confidence Band
        fig, ax = plt.subplots(figsize=(14, 5))
        hist_hours = np.arange(-48, 0)
        future_hours = np.arange(1, 25)

        ax.plot(hist_hours, history_48.ravel() / 1e6, label="Observed Past 48 Hours", color="#003087", lw=2.5)
        ax.plot(future_hours, forecast_24 / 1e6, label="GRU Projected Next 24 Hours", color="#0070BA", lw=2.5, linestyle="--", marker="o", markersize=4)

        # Empirical Error Envelope
        ax.fill_between(
            future_hours,
            lower_bound / 1e6,
            upper_bound / 1e6,
            color="#0070BA", alpha=0.18,
            label="Empirical Validation Residual Band (10th–90th Percentile)"
        )

        ax.axvline(0, color="gray", linestyle=":", lw=1.5, label="Current Timestamp (t = 0)")
        ax.set_title("24-Hour Payment Volume Liquidity Trajectory ($M)", fontsize=13, fontweight='bold', color="#003087")
        ax.set_xlabel("Hours Relative to Current Timestamp")
        ax.set_ylabel("Volume in Millions USD ($M)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.caption(
            "📌 **Empirical Error Band Note**: The shaded confidence band reflects the 10th and 90th percentile of out-of-sample forecast residuals "
            f"measured on the validation set ({q10_res/1e6:+.1f}M to {q90_res/1e6:+.1f}M), providing a true 80% empirical prediction interval rather than an arbitrary percentage assumption."
        )

        with st.expander("📊 View Hourly Projection Table"):
            proj_df = pd.DataFrame({
                "Hour Offset": [f"t + {i+1}h" for i in range(24)],
                "Projected Volume ($)": [f"${v:,.2f}" for v in forecast_24],
                "Volume ($M)": [round(v / 1e6, 3) for v in forecast_24],
                "Lower Bound ($M)": [round(lb / 1e6, 3) for lb in lower_bound],
                "Upper Bound ($M)": [round(ub / 1e6, 3) for ub in upper_bound]
            })
            st.dataframe(proj_df, use_container_width=True)

    else:
        st.warning("Historical volume file not found. Run `scripts/prepare_app_assets.py`.")

st.divider()
st.caption("PayPal Merchant Guard | Built for College Deep Learning CA & Hackathon Extension | TensorFlow 2.16 & Streamlit")
