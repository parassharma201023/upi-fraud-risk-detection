import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
conn.close()

df['amount_ratio'] = df['amount'] / (df['user_avg_txn_value'] + 1)

# KYC status ko text se number mein convert karo (one-hot encoding)
kyc_dummies = pd.get_dummies(df['user_kyc_status'], prefix='kyc', drop_first=True)
df = pd.concat([df, kyc_dummies], axis=1)

features = ['amount', 'hour_of_day', 'is_weekend', 'is_night_transaction',
            'new_device_flag', 'ip_location_mismatch', 'failed_attempts_last_24h',
            'transaction_velocity', 'amount_deviation_score', 'user_loyalty_score',
            'amount_ratio', 'is_high_risk_user'] + list(kyc_dummies.columns)

X = df[features].fillna(0)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6,
                                             class_weight='balanced_subsample', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                              max_depth=4, learning_rate=0.1, random_state=42)
}

def best_threshold(y_true, probs):
    prec, rec, thresh = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresh[best_idx] if best_idx < len(thresh) else 0.5

results = []
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

    t = best_threshold(y_test, probs)
    preds = (probs >= t).astype(int)

    results.append({
        'Model': name,
        'Best Threshold': round(t, 3),
        'Precision': round(precision_score(y_test, preds), 3),
        'Recall': round(recall_score(y_test, preds), 3),
        'F1-Score': round(f1_score(y_test, preds), 3),
        'ROC-AUC': round(roc_auc_score(y_test, probs), 3)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\nFeatures used:", features)