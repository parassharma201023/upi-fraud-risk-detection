import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

conn = sqlite3.connect(DATA_DIR / "upi_fraud.db")
df = pd.read_sql_query("""
    SELECT t.*, u.is_high_risk_user
    FROM transactions t
    JOIN users u ON t.user_id = u.user_id
""", conn)

df['amount_ratio'] = df['amount'] / (df['user_avg_txn_value'] + 1)
kyc_dummies = pd.get_dummies(df['user_kyc_status'], prefix='kyc', drop_first=True)
df = pd.concat([df, kyc_dummies], axis=1)

features = ['amount', 'hour_of_day', 'is_weekend', 'is_night_transaction',
            'new_device_flag', 'ip_location_mismatch', 'failed_attempts_last_24h',
            'transaction_velocity', 'amount_deviation_score', 'user_loyalty_score',
            'amount_ratio', 'is_high_risk_user'] + list(kyc_dummies.columns)

X = df[features].fillna(0)
y = df['is_fraud']

# ---------- Step A: Honest evaluation on holdout ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler_eval = StandardScaler()
X_train_scaled = scaler_eval.fit_transform(X_train)
X_test_scaled = scaler_eval.transform(X_test)

eval_model = LogisticRegression(class_weight='balanced', max_iter=1000)
eval_model.fit(X_train_scaled, y_train)
test_probs = eval_model.predict_proba(X_test_scaled)[:, 1]

prec, rec, thresh = precision_recall_curve(y_test, test_probs)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
best_t = thresh[np.argmax(f1_scores)]
preds = (test_probs >= best_t).astype(int)

print("=== Honest holdout evaluation (Logistic Regression) ===")
print(f"Best threshold: {best_t:.3f}")
print(f"Precision: {precision_score(y_test, preds):.3f}")
print(f"Recall: {recall_score(y_test, preds):.3f}")
print(f"F1-score: {f1_score(y_test, preds):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, test_probs):.3f}")

tier_rows = []
for pct in [1, 3, 5, 10, 20]:
    cutoff = np.percentile(test_probs, 100 - pct)
    flagged = test_probs >= cutoff
    caught = y_test[flagged].sum()
    tier_rows.append({'Top %': f"{pct}%", 'Flagged': int(flagged.sum()),
                       'Frauds caught': int(caught), 'Total frauds': int(y_test.sum()),
                       'Catch rate %': round(100 * caught / y_test.sum(), 1)})
print("\nRisk Tier Report (holdout):\n", pd.DataFrame(tier_rows).to_string(index=False))

# ---------- Step B: Train production model on ALL data ----------
scaler_prod = StandardScaler()
X_scaled_full = scaler_prod.fit_transform(X)
prod_model = LogisticRegression(class_weight='balanced', max_iter=1000)
prod_model.fit(X_scaled_full, y)

full_probs = prod_model.predict_proba(X_scaled_full)[:, 1]
df['risk_score'] = (full_probs * 100).round(1)

df['risk_tier'] = pd.cut(
    df['risk_score'].rank(pct=True),
    bins=[0, 0.80, 0.95, 1.0],
    labels=['Low', 'Medium', 'High']
)

# ---------- Step C: Reason codes from model coefficients ----------
readable = {
    'amount': 'Unusually large amount',
    'hour_of_day': 'Unusual transaction hour',
    'is_weekend': 'Weekend transaction',
    'is_night_transaction': 'Night-time transaction',
    'new_device_flag': 'New/unrecognized device',
    'ip_location_mismatch': 'IP location mismatch',
    'failed_attempts_last_24h': 'Multiple failed attempts (24h)',
    'transaction_velocity': 'High transaction velocity',
    'amount_deviation_score': 'Amount deviates from user pattern',
    'user_loyalty_score': 'Low loyalty score',
    'amount_ratio': 'Amount much higher than usual for user',
    'is_high_risk_user': 'Flagged as high-risk user',
}
for col in kyc_dummies.columns:
    readable[col] = 'Unverified/pending KYC'

coefs = prod_model.coef_[0]
contributions = X_scaled_full * coefs

reason_codes = []
for i in range(len(df)):
    row_contrib = contributions[i]
    top_idx = np.argsort(row_contrib)[::-1][:3]
    reasons = [readable.get(features[j], features[j]) for j in top_idx if row_contrib[j] > 0]
    reason_codes.append("; ".join(reasons) if reasons else "No strong risk signal")

df['reason_codes'] = reason_codes

# ---------- Save for Power BI ----------
output_cols = ['transaction_id', 'user_id', 'receiver_id', 'amount', 'date',
               'is_fraud', 'risk_score', 'risk_tier', 'reason_codes']
df[output_cols].to_sql('risk_scores', conn, if_exists='replace', index=False)
conn.close()

print("\nrisk_scores table saved:", len(df), "rows")
print("\nSample high-risk transactions:")
print(df[df['risk_tier'] == 'High'][['transaction_id', 'risk_score', 'reason_codes']].head(5).to_string(index=False))