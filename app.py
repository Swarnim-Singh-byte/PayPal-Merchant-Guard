"""
PayPal Merchant Guard - Streamlit Application
Interactive Command Center for Real-Time Fraud Alerts, Merchant Churn Risk,
and 24-Hour Cash-Flow Forecasting.
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

# Custom Styling (PayPal Design Palette)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #003087;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #0070BA;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
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

        return {
            "fraud_model": fraud_model,
            "churn_model": churn_model,
            "forecaster_model": forecaster_model,
            "fraud_scaler": fraud_scaler,
            "churn_scaler": churn_scaler,
            "forecast_scaler": forecast_scaler,
            "fraud_features": fraud_features,
            "status": "ready"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

models_dict = load_all_models()

# -------------------------------------------------------------
# Sidebar: System Health & Navigation
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b5/PayPal.svg", width=140)
    st.markdown("### **Merchant Guard Engine**")
    st.caption("College Deep Learning CA Project & Hackathon Suite")
    st.divider()

    if models_dict["status"] == "ready":
        st.success("🟢 **Deep Learning Engines Loaded**")
        st.markdown("""
        - **Fraud MLP**: `64-32 Dense (ReLU)`
        - **Churn MLP**: `32-16 Dense (ReLU)`
        - **Forecaster**: `GRU(64) Recurrent`
        """)
    else:
        st.error(f"🔴 **Model Load Error**: {models_dict.get('message', '')}")

    st.divider()
    st.markdown("#### **Operating Thresholds**")
    fraud_threshold = st.slider("Fraud Decision Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                                help="Derived via optimal PR-AUC tuning in Phase 1.")
    st.caption(f"Transactions with probability $\ge {fraud_threshold:.2f}$ are flagged for immediate hold.")

    st.divider()
    st.info("💡 **Dataset**: PaySim Simulator (6.36M records). Models fit strictly on train splits without data leakage.")

# -------------------------------------------------------------
# Main Dashboard Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ PayPal Merchant Guard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Deep Learning Risk Mitigation, Retention Intelligence, and Working Capital Forecasting</div>', unsafe_allow_html=True)

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
    st.write("Score incoming payments against our Deep MLP trained on PaySim. The model automatically computes ledger discrepancy features to uncover unauthorized balance draining.")

    sub_mode = st.radio("Mode Selection:", ["Single Transaction Inspector", "Batch / Sample Stream Evaluation"], horizontal=True)

    if sub_mode == "Single Transaction Inspector":
        st.markdown("##### **Transaction Attributes**")

        # Preset Quick-Fills
        col_pre1, col_pre2, col_pre3 = st.columns(3)
        with col_pre1:
            if st.button("⚡ Preset: Typ. Fraud (Drain Origin)"):
                st.session_state["f_type"] = "TRANSFER"
                st.session_state["f_amount"] = 350000.0
                st.session_state["f_old_orig"] = 350000.0
                st.session_state["f_new_orig"] = 0.0
                st.session_state["f_old_dest"] = 0.0
                st.session_state["f_new_dest"] = 0.0
        with col_pre2:
            if st.button("✅ Preset: Legitimate Payment"):
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
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=float(st.session_state.get("f_amount", 181.0)), step=100.0)

        with col_f2:
            old_orig = st.number_input("Origin Initial Balance ($)", min_value=0.0, value=float(st.session_state.get("f_old_orig", 181.0)), step=100.0)
            new_orig = st.number_input("Origin Final Balance ($)", min_value=0.0, value=float(st.session_state.get("f_new_orig", 0.0)), step=100.0)

        with col_f3:
            old_dest = st.number_input("Destination Initial Balance ($)", min_value=0.0, value=float(st.session_state.get("f_old_dest", 0.0)), step=100.0)
            new_dest = st.number_input("Destination Final Balance ($)", min_value=0.0, value=float(st.session_state.get("f_new_dest", 0.0)), step=100.0)

        # Derived Feature Engineering
        err_orig = new_orig + amount - old_orig
        err_dest = old_dest + amount - new_dest

        st.markdown(f"""
        **Engineered Ledger Discrepancies**:
        - $\\text{{errorBalanceOrig}} = \\text{{newbalanceOrig}} + \\text{{amount}} - \\text{{oldbalanceOrg}} = \\mathbf{{\\${err_orig:,.2f}}}$
        - $\\text{{errorBalanceDest}} = \\text{{oldbalanceDest}} + \\text{{amount}} - \\text{{newbalanceDest}} = \\mathbf{{\\${err_dest:,.2f}}}$
        """)

        if st.button("🔍 Evaluate Transaction Fraud Risk", type="primary"):
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
                st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
                if fraud_prob >= fraud_threshold:
                    st.error("🚨 **DECISION: HIGH FRAUD RISK (HOLD / REJECT)**")
                elif fraud_prob >= 0.25:
                    st.warning("⚠️ **DECISION: ELEVATED RISK (MANUAL REVIEW)**")
                else:
                    st.success("✅ **DECISION: LOW RISK (AUTO-APPROVE)**")

            with col_res2:
                st.markdown("##### **Deep Learning Risk Diagnostics**")
                reasons = []
                if txn_type in ['TRANSFER', 'CASH_OUT']:
                    reasons.append(f"• High-risk transaction category (`{txn_type}`). 100% of PaySim fraud concentrates in TRANSFER & CASH_OUT.")
                if abs(err_orig) > 1.0:
                    reasons.append(f"• **Origin balance mismatch** detected: \${err_orig:,.2f} deviation from expected ledger continuity.")
                if new_orig == 0.0 and old_orig > 1000:
                    reasons.append("• **Complete balance liquidation**: Account origin emptied in single transaction.")
                if not reasons:
                    reasons.append("• Clean ledger accounting with balanced origin and destination books.")

                for r in reasons:
                    st.write(r)

    else:
        st.markdown("##### **Batch Upload & Stream Testing**")
        uploaded_file = st.file_uploader("Upload Transaction CSV (Matching PaySim Schema)", type=["csv"])
        load_preset = st.button("📂 Load Pre-computed Sample (50 Fraud + 50 Legitimate Txns)")

        batch_df = None
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
        elif load_preset or os.path.exists("data/sample_transactions.csv"):
            batch_df = pd.read_csv("data/sample_transactions.csv")
            st.info("Loaded pre-computed sample containing 100 real PaySim transactions.")

        if batch_df is not None:
            with st.spinner("Processing batch with Deep MLP..."):
                # Compute engineered features
                proc_df = batch_df.copy()
                proc_df['errorBalanceOrig'] = proc_df['newbalanceOrig'] + proc_df['amount'] - proc_df['oldbalanceOrg']
                proc_df['errorBalanceDest'] = proc_df['oldbalanceDest'] + proc_df['amount'] - proc_df['newbalanceDest']

                # One hot encode
                for t in ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
                    proc_df[f'type_{t}'] = (proc_df['type'] == t).astype(int)

                X_batch = proc_df[models_dict["fraud_features"]]
                X_batch_scaled = models_dict["fraud_scaler"].transform(X_batch)
                preds = models_dict["fraud_model"].predict(X_batch_scaled, verbose=0).ravel()

                batch_df['fraud_probability'] = np.round(preds, 4)
                batch_df['action'] = np.where(preds >= fraud_threshold, "🚨 HOLD / REJECT", "✅ APPROVE")

            # Metrics
            total_scanned = len(batch_df)
            flagged = (batch_df['fraud_probability'] >= fraud_threshold).sum()
            flagged_dollars = batch_df[batch_df['fraud_probability'] >= fraud_threshold]['amount'].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Scanned", f"{total_scanned:,}")
            c2.metric("Flagged Fraud", f"{flagged:,} ({(flagged/total_scanned)*100:.1f}%)")
            c3.metric("Prevented Loss", f"${flagged_dollars:,.2f}")

            st.dataframe(
                batch_df[['step', 'type', 'amount', 'nameOrig', 'nameDest', 'fraud_probability', 'action']],
                use_container_width=True
            )

# =============================================================================
# TAB 2: CHURN RISK
# =============================================================================
with tab_churn:
    st.subheader("Merchant Churn & Dormancy Intelligence (Module B)")
    st.write("Predict which merchants are exhibiting early-stage abandonment behavior based on Days 1–20 transaction velocity, recency decay, and revenue stability.")

    if os.path.exists("data/merchant_cohort_sample.csv"):
        m_cohort = pd.read_csv("data/merchant_cohort_sample.csv")

        # Top KPI Metrics
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

        # Filters
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

        # Merchant Detail Inspector
        st.markdown("##### **Individual Merchant Deep Dive**")
        selected_m = st.selectbox("Select Merchant to Inspect:", filtered['merchant_id'].tolist()[:50])
        if selected_m:
            m_row = filtered[filtered['merchant_id'] == selected_m].iloc[0]
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.metric("Predicted Churn Probability", f"{m_row['churn_probability']*100:.1f}%")
                if m_row['risk_tier'] == 'High Risk':
                    st.error("🚨 **Status: HIGH CHURN RISK**")
                    st.write("**Recommended Intervention**: Trigger automated account manager reachout with a 0.5% fee rebate for the next 30 days.")
                elif m_row['risk_tier'] == 'Medium Risk':
                    st.warning("⚠️ **Status: MEDIUM RISK (SLIPPING)**")
                    st.write("**Recommended Intervention**: Dispatch product update newsletter highlighting PayPal Checkout conversion improvements.")
                else:
                    st.success("✅ **Status: HEALTHY ACTIVE MERCHANT**")
                    st.write("**Recommended Intervention**: Merchant is actively processing payments. Eligible for pre-approved working capital loan.")

            with col_m2:
                st.write("**Merchant Behavioral Metrics**:")
                st.write(f"- **Days Since Last Payment**: {int(m_row['days_since_last_txn'])} days")
                st.write(f"- **Total Transactions in Window**: {int(m_row['txn_count'])}")
                st.write(f"- **Mean Payment Volume**: \${m_row['mean_amount']:,.2f}")
                st.write(f"- **Distinct Active Days**: {int(m_row['active_days'])} out of 20")
    else:
        st.warning("Merchant cohort file not found. Run `scripts/prepare_app_assets.py` to generate it.")

# =============================================================================
# TAB 3: CASH-FLOW FORECASTING
# =============================================================================
with tab_forecast:
    st.subheader("Hourly Payment Volume & Cash-Flow Projection (Module C)")
    st.write("Recurrent Gated Neural Network (GRU) projecting payment volume over the upcoming 24 hours based on the preceding 48-hour sequence.")

    if os.path.exists("data/recent_hourly_volumes.csv"):
        vol_df = pd.read_csv("data/recent_hourly_volumes.csv")
        history_48 = vol_df.tail(48)['amount'].values.reshape(-1, 1)

        # Autoregressive multi-step projection for 24 hours
        scaler_fc = models_dict["forecast_scaler"]
        gru_model = models_dict["forecaster_model"]

        curr_seq = scaler_fc.transform(history_48).reshape(1, 48, 1)
        forecast_24_scaled = []

        for _ in range(24):
            pred_scaled = gru_model.predict(curr_seq, verbose=0)[0][0]
            forecast_24_scaled.append(pred_scaled)
            # Roll forward sequence
            curr_seq = np.append(curr_seq[:, 1:, :], [[[pred_scaled]]], axis=1)

        forecast_24 = scaler_fc.inverse_transform(np.array(forecast_24_scaled).reshape(-1, 1)).ravel()

        # KPI Metrics
        total_24h_vol = forecast_24.sum()
        peak_hr_vol = forecast_24.max()
        recommended_buffer = total_24h_vol * 0.15

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Projected 24h Volume", f"${total_24h_vol/1e6:.2f}M")
        fc2.metric("Peak Hourly Inflow", f"${peak_hr_vol/1e6:.2f}M")
        fc3.metric("Recommended Liquidity Buffer (15%)", f"${recommended_buffer/1e6:.2f}M",
                   help="Recommended reserve to meet same-day chargebacks and refunds.")

        st.divider()

        # Plot Forecast
        fig, ax = plt.subplots(figsize=(14, 5))
        hist_hours = np.arange(-48, 0)
        future_hours = np.arange(1, 25)

        ax.plot(hist_hours, history_48.ravel() / 1e6, label="Observed Past 48 Hours", color="#003087", lw=2.5)
        ax.plot(future_hours, forecast_24 / 1e6, label="GRU Projected Next 24 Hours", color="#0070BA", lw=2.5, linestyle="--", marker="o", markersize=4)

        # Uncertainty envelope (+/- 10%)
        ax.fill_between(future_hours, (forecast_24 * 0.90) / 1e6, (forecast_24 * 1.10) / 1e6,
                        color="#0070BA", alpha=0.15, label="Expected Variance Interval (±10%)")

        ax.axvline(0, color="gray", linestyle=":", lw=1.5, label="Current Hour (t = 0)")
        ax.set_title("Hourly Payment Volume Liquidity Trajectory ($M)", fontsize=13, fontweight='bold', color="#003087")
        ax.set_xlabel("Hours Relative to Current Timestamp")
        ax.set_ylabel("Volume in Millions USD ($M)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Tabular View
        with st.expander("📊 View Hourly Projection Table"):
            proj_df = pd.DataFrame({
                "Hour Offset": [f"t + {i+1}h" for i in range(24)],
                "Projected Volume ($)": [f"${v:,.2f}" for v in forecast_24],
                "Volume ($M)": [round(v / 1e6, 3) for v in forecast_24]
            })
            st.dataframe(proj_df, use_container_width=True)

    else:
        st.warning("Historical volume file not found. Run `scripts/prepare_app_assets.py`.")

st.divider()
st.caption("PayPal Merchant Guard | Built for College Deep Learning CA & Hackathon Extension | TensorFlow 2.16 & Streamlit")
