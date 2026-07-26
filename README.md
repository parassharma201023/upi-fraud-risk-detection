# 🇮🇳 UPI Transactions Dataset — Fraud Detection & Spending Analytics

> A comprehensive, realistic synthetic dataset simulating India's Unified Payments Interface (UPI) ecosystem — built for machine learning, fraud detection, and financial analytics.

---

## 📌 Problem Statement

India's UPI network processes **billions of transactions every month**, making it one of the world's largest real-time payment systems. With scale comes risk — fraud, anomalous behavior, and identity misuse are growing challenges for fintechs, banks, and regulators alike.

This dataset simulates **20,000 UPI transactions** across 2,000 users and 400 merchants, covering an entire calendar year (Jan–Dec 2024). It is structured to support:

- 🔍 **Fraud Detection** — Binary classification using behavioral and risk signals
- 📊 **Spending Behavior Analysis** — Cluster users by transaction patterns
- 🏪 **Merchant Analytics** — Understand category-level payment trends
- 📈 **Time-Series Modeling** — Temporal patterns in payment behavior
- 🎯 **Credit & Risk Scoring** — Build synthetic risk profiles

---

## 📂 Dataset Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `transactions.csv` | 20,000 | 30 | Core transaction log with all features |
| `users.csv` | 2,000 | 13 | Sender profiles and behavioral attributes |
| `merchants.csv` | 400 | 9 | Merchant/receiver metadata |
| `fraud_labels.csv` | 20,000 | 11 | Isolated fraud signals and labels |

> **Relational Key:** `user_id` links `transactions ↔ users`. `receiver_id` links `transactions ↔ merchants` (for P2M transactions).

---

## 🔑 Key Features at a Glance

### Transactions (`transactions.csv`)
```
transaction_id, user_id, receiver_id, receiver_type, amount, timestamp,
date, hour_of_day, day_of_week, is_weekend, is_night_transaction,
time_since_last_txn_min, transaction_type, payment_app, device_type,
status, user_city_tier, user_kyc_status, user_avg_monthly_txn,
user_avg_txn_value, user_loyalty_score, new_device_flag,
ip_location_mismatch, failed_attempts_last_24h, transaction_velocity,
amount_deviation_score, is_fraud, recurring_payment_flag,
balance_after_transaction, transaction_frequency_score
```

### Users (`users.csv`)
```
user_id, age_group, city, city_tier, kyc_status, account_age_days,
linked_bank_count, avg_monthly_transactions, avg_transaction_value,
preferred_app, preferred_device, user_loyalty_score, is_high_risk_user
```

### Merchants (`merchants.csv`)
```
merchant_id, merchant_name, merchant_category, merchant_size,
city, city_tier, avg_daily_transactions, is_registered, rating
```

---

## 📊 Dataset Statistics

| Metric | Value |
|---|---|
| Total transactions | 20,000 |
| Date range | Jan 2024 – Dec 2024 |
| Fraud rate | ~3.8% |
| Avg transaction amount | ₹876.85 |
| Transaction success rate | 88.1% |
| Payment apps covered | 6 (GPay, PhonePe, Paytm, BHIM, Amazon Pay, WhatsApp Pay) |
| Cities covered | 38 (Tier 1, 2, and 3) |
| Transaction types | P2P, P2M, Bill Payment, Recharge, EMI, Subscription |

---

## 🧠 Suggested Use Cases & Notebooks

### 1. 🔐 Fraud Detection (Classification)
**Target:** `is_fraud`  
**Suggested models:** XGBoost, LightGBM, Random Forest, Logistic Regression  
**Key features:** `new_device_flag`, `ip_location_mismatch`, `failed_attempts_last_24h`, `transaction_velocity`, `amount_deviation_score`, `is_night_transaction`, `user_kyc_status`

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = ['amount', 'hour_of_day', 'is_weekend', 'is_night_transaction',
            'new_device_flag', 'ip_location_mismatch', 'failed_attempts_last_24h',
            'transaction_velocity', 'amount_deviation_score', 'user_loyalty_score']

X = df[features].fillna(0)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)
```

### 2. 👥 User Spending Segmentation (Clustering)
**Target:** Unsupervised  
**Suggested models:** K-Means, DBSCAN, Hierarchical Clustering  
**Aggregate by `user_id`:** total spend, frequency, avg amount, night ratio, app preference

### 3. 📅 Time-Series Spending Patterns
**Use:** `timestamp`, `hour_of_day`, `day_of_week`, `is_weekend`  
**Explore:** peak transaction hours, weekend vs weekday volumes, monthly trends

### 4. 🏪 Merchant Category Analysis
**Join:** `transactions.csv` + `merchants.csv` via `receiver_id`  
**Explore:** category-wise revenue, fraud rates per category, city-tier spending gaps

---

## 🗂️ Data Dictionary

See `data_dictionary.csv` for full column-by-column descriptions, data types, and value ranges for all four files.

---

## ⚙️ Data Generation Methodology

This dataset was **fully synthetically generated** using Python (NumPy, Pandas, Faker). Key design decisions:

- **Amount distribution:** Log-normal (realistic skew — many small, few large transactions), capped at ₹1,00,000 per NPCI norms
- **Fraud labeling:** Probabilistic scoring model combining 8 risk signals (not random assignment). Fraud score = weighted sum of `new_device_flag + ip_mismatch + failed_attempts + velocity + amount_deviation + kyc_status + is_night + high_risk_user`
- **City-tier realism:** Tier 1 users have higher transaction frequency and value than Tier 2/3
- **Controlled noise:** ~2% random NaN values in `time_since_last_txn_min`, `transaction_velocity`, and `amount_deviation_score` to simulate real-world data quality
- **Temporal realism:** Transactions distributed across full 2024 calendar year

---

## 🔒 Privacy & Ethics

- ✅ **100% synthetic data** — no real individuals, accounts, or financial records
- ✅ No PII (no names, phone numbers, Aadhaar, or bank account numbers)
- ✅ User and merchant IDs are anonymized codes (`USR00001`, `MRC0001`)
- ✅ Designed following principles aligned with **NPCI data privacy guidelines**
- ✅ Safe for public research, education, and competition use

---

## 💡 Pro Tips for Explorers

1. **Class imbalance** — Fraud rate is ~3.8%. Use `class_weight='balanced'`, SMOTE, or threshold tuning
2. **Feature engineering** — `amount / user_avg_txn_value` ratio is a powerful derived feature
3. **Join the tables** — Merchant category + transaction data unlocks richer fraud patterns
4. **Temporal features** — `hour_of_day` and `is_night_transaction` are strong behavioral signals
5. **Missing values** — Intentional NaNs in 3 columns; handle them thoughtfully (don't just drop)

---

## 📜 License

This dataset is released under **CC0: Public Domain**. You are free to use, modify, and distribute it for any purpose without restriction.

---

## 🙏 Acknowledgements

Inspired by India's real UPI ecosystem operated by the **National Payments Corporation of India (NPCI)**. All data is synthetic and does not represent any real financial institution, user, or transaction.

---

*If you find this dataset useful, please upvote ⬆️ and share your notebooks — it helps the community!*
