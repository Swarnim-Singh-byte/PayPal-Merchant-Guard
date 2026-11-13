"""
Pre-compute fast-loading assets for the Streamlit dashboard:
1. data/sample_transactions.csv: Mix of real fraud & legitimate transactions.
2. data/merchant_cohort_sample.csv: Curated merchant behavior table with churn predictions.
3. data/recent_hourly_volumes.csv: Last 96 hours of hourly volume for the forecaster.
"""
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

os.environ["KERAS_HOME"] = os.path.abspath(".keras")
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib")

DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"

print("[*] Generating fast-loading data assets for Streamlit...")

# 1. Sample transactions for Fraud Alerts tab
dtypes = {
    'step': 'uint16',
    'type': 'category',
    'amount': 'float32',
    'nameOrig': 'string',
    'oldbalanceOrg': 'float32',
    'newbalanceOrig': 'float32',
    'nameDest': 'string',
    'oldbalanceDest': 'float32',
    'newbalanceDest': 'float32',
    'isFraud': 'uint8',
    'isFlaggedFraud': 'uint8'
}
df = pd.read_csv(DATA_PATH, dtype=dtypes)

# Get 50 real fraud transactions and 50 normal transactions
sample_fraud = df[df['isFraud'] == 1].sample(n=50, random_state=42)
sample_legit = df[df['isFraud'] == 0].sample(n=50, random_state=42)
sample_txns = pd.concat([sample_fraud, sample_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
sample_txns.to_csv("data/sample_transactions.csv", index=False)
print(f"[+] Saved data/sample_transactions.csv ({len(sample_txns)} rows)")

# 2. Curated Merchant Cohort with pre-computed churn scores
m_df = df[df['nameDest'].str.startswith('M')].copy()
m_df['day'] = (m_df['step'] - 1) // 24 + 1
m_d1_20 = m_df[m_df['day'] <= 20]

# Group by merchant
grouped = m_d1_20.groupby('nameDest')
# Take repeat merchants + sample
repeat_merchants = m_df['nameDest'].value_counts()[lambda x: x >= 2].index[:500]
sample_merchants = m_d1_20['nameDest'].drop_duplicates().sample(n=500, random_state=42)
selected_ids = list(repeat_merchants) + list(sample_merchants)

selected_df = m_d1_20[m_d1_20['nameDest'].isin(selected_ids)]
grp = selected_df.groupby('nameDest')

merchants_table = pd.DataFrame({
    'merchant_id': grp['nameDest'].first().index,
    'txn_count': grp['amount'].count().values,
    'mean_amount': grp['amount'].mean().round(2).values,
    'std_amount': grp['amount'].std().fillna(0).round(2).values,
    'days_since_last_txn': (20 - grp['day'].max()).values,
    'active_days': grp['day'].nunique().values
})

# Score with churn model
churn_scaler = joblib.load("models/churn_scaler.joblib")
churn_model = tf.keras.models.load_model("models/churn_mlp.keras")

X_c = merchants_table[['txn_count', 'mean_amount', 'std_amount', 'days_since_last_txn', 'active_days']]
X_c_scaled = churn_scaler.transform(X_c)
merchants_table['churn_probability'] = churn_model.predict(X_c_scaled).ravel().round(4)
merchants_table['risk_tier'] = pd.cut(
    merchants_table['churn_probability'],
    bins=[-0.01, 0.35, 0.70, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)
merchants_table = merchants_table.sort_values(by='churn_probability', ascending=False).reset_index(drop=True)
merchants_table.to_csv("data/merchant_cohort_sample.csv", index=False)
print(f"[+] Saved data/merchant_cohort_sample.csv ({len(merchants_table)} merchants)")

# 3. Recent hourly volumes for Forecaster tab
hourly = df.groupby('step')['amount'].sum().reset_index()
all_steps = pd.DataFrame({'step': np.arange(1, 744, dtype=np.uint16)})
hourly = pd.merge(all_steps, hourly, on='step', how='left').fillna(0)
# Save last 96 hours
hourly.tail(96).to_csv("data/recent_hourly_volumes.csv", index=False)
print(f"[+] Saved data/recent_hourly_volumes.csv (last 96 hours)")

print("[+] All assets generated successfully!")
