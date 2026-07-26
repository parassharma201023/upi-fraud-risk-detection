# UPI Fraud Risk Intelligence System

An end-to-end fraud risk detection system for UPI transactions — built with a SQL data warehouse, a Python risk-scoring engine, an interactive Power BI dashboard, and an Excel cost-benefit model.

This isn't just a classification notebook. It's a small version of what a fraud ops analytics stack actually looks like: structured data storage → explainable risk scoring → an investigation-ready dashboard → a business-impact translation layer.

---

## Problem Statement

India's UPI network processes billions of transactions a month. Two mistakes are costly in different ways: missing real fraud (direct financial loss) and over-flagging genuine transactions (customer friction, lost trust). This project builds a system that doesn't just label transactions as "fraud" or "not fraud" — it produces a **0–100 risk score with human-readable reasons**, so a fraud ops team can prioritize review effort instead of treating every alert equally.

## Architecture

```
Raw UPI data (3 relational CSVs)
        │
        ▼
SQL data warehouse (SQLite, star schema)
   fact_transactions + dim_users + dim_merchants + dim_date
        │
        ▼
Python risk engine
   feature engineering → 3 models compared → best model selected
   → probability converted to 0-100 risk score → reason codes
        │
        ▼
Power BI dashboard (3 pages)
   Executive Overview | Investigation View (drill-through) | Time & Behavior Patterns
        │
        ▼
Excel cost-benefit model
   translates catch-rate into estimated monthly savings
```

## Dataset

[UPI Transactions 2024 Dataset](https://www.kaggle.com/) (Kaggle) — a synthetic dataset modeling India's UPI ecosystem, released under CC0.

- 20,000 transactions across 2,000 users and 400 merchants, full 2024 calendar year
- 6 payment apps (GPay, PhonePe, Paytm, BHIM, Amazon Pay, WhatsApp Pay), 38 Indian cities across Tier 1/2/3
- Fraud rate ~3.8%, fraud label generated probabilistically from 8 weighted risk signals (device, IP mismatch, failed attempts, velocity, amount deviation, KYC status, night-time flag, high-risk-user flag)
- 100% synthetic — no real individuals, accounts, or transactions. Safe for public research and education.

## Tech Stack

| Layer | Tools |
|---|---|
| Data warehouse | SQLite, SQL (JOINs, aggregation, star schema) |
| Modeling | Python — pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn |
| Visualization (EDA) | Matplotlib, Seaborn |
| Dashboard | Power BI (DAX, drill-through, star/galaxy schema) |
| Business impact | Excel |

## Modeling Approach

Three models were trained and compared — Logistic Regression, Random Forest, and XGBoost — using class-weighting to handle the ~3.8% fraud rate, with the classification threshold tuned per model via the precision-recall curve (the default 0.5 threshold performs poorly on this imbalanced data).

SMOTE oversampling was also tested, since it's a common recommendation for imbalanced classification. It made every model's ROC-AUC *worse* (0.75 → ~0.55), most likely because the dataset's probabilistic fraud-label generation introduces label noise, and SMOTE amplified that noise rather than a real signal. It was reverted in favor of class-weighting. This is reported here deliberately — not every standard technique works on every dataset, and the evaluation step is what tells you that.

**Final model: Logistic Regression** (best ROC-AUC among the three, and a natural fit since the dataset's label is itself generated from a linear weighted sum of risk signals).

### Holdout evaluation

| Metric | Score |
|---|---|
| Best threshold | 0.711 |
| Precision | 0.125 |
| Recall | 0.327 |
| F1-score | 0.181 |
| ROC-AUC | 0.745 |

Precision/Recall at a single threshold understate what this model is actually useful for. The metric that matters for a fraud ops team is **lift**: how much better than random review is the model, at a given review workload?

### Risk-tier lift report (holdout set)

| Top % reviewed | Frauds caught | Catch rate | Lift vs. random |
|---|---|---|---|
| 1% | 4 / 153 | 2.6% | 2.6x |
| 3% | 17 / 153 | 11.1% | 3.7x |
| 5% | 26 / 153 | 17.0% | 3.4x |
| 10% | 50 / 153 | 32.7% | 3.3x |
| 20% | 73 / 153 | 47.7% | 2.4x |

**Reviewing just the top 5% highest-risk transactions catches 3.4x more fraud than random sampling.**

### Explainability

Each transaction gets a risk score (model probability × 100) plus up to 3 plain-language reason codes, derived from each feature's coefficient contribution (`scaled feature value × logistic regression coefficient`) — a simplified, linear-model version of SHAP-style attribution. Example:

> **Risk score: 84.7** — New/unrecognized device; IP location mismatch; Amount much higher than usual for user

## Power BI Dashboard

**Page 1 — Executive Overview:** KPI cards (transaction volume, fraud rate, high-risk count, catch rate), fraud rate by merchant category, risk-tier split, fraud rate trend over time.

**Page 2 — Investigation View:** Sortable, conditionally-formatted transaction table with risk scores and reason codes. Supports **drill-through** from the category chart on the Executive Overview page — clicking a category filters straight to its suspicious transactions.

**Page 3 — Time & Behavior Patterns:** Fraud rate by hour of day and day of week, using a proper date dimension table for correct chronological sorting and time intelligence.

*(Screenshots: add yours to a `screenshots/` folder and reference them here, e.g. `![Executive Overview](screenshots/executive_overview.png)`)*

## Key Insights

- Reviewing the top 5% of transactions by risk score catches 17% of all fraud — a 3.4x lift over random review.
- Fuel, Insurance, and Grocery categories show the highest fraud rates, though some (like Fuel Tier-1) have small sample sizes and should be read with caution rather than as strong signals on their own.
- Fraud rate shows mild seasonality — dipping around July (~3.5%) and peaking in November (~4.4%), plausibly tied to festive-season shopping activity.
- The single best predictor structurally is that the dataset's fraud label is itself a linear combination of risk signals — which is why a simple, interpretable model (Logistic Regression) outperformed more complex ones here. Model choice should follow from understanding the data, not from picking the most advanced algorithm available.

## Business Impact (illustrative)

A simple Excel cost-benefit model translates the top-5% catch rate into estimated monthly savings, using illustrative assumptions (₹8,000 average fraud loss, ₹25 manual review cost, 50,000 monthly transactions — actual figures would vary by organization):

**Estimated net monthly savings: ₹25.2 lakh**, from focusing manual review effort on the top 5% highest-risk transactions instead of reviewing randomly or not at all.

## Project Structure

```
upi-fraud-risk-detection/
├── data/                    # Raw CSVs + SQLite database
├── notebooks/
│   ├── 01_load_data.py
│   ├── 02_join_query.py
│   ├── 03_eda.py
│   ├── 04_build_star_schema.py
│   ├── 05_train_models.py
│   ├── 06_generate_risk_scores.py
│   └── plots/
├── powerbi/
│   └── UPI_Fraud_Risk_Dashboard.pbix
├── excel/
│   └── fraud_cost_benefit_calculator.xlsx
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn

cd notebooks
python 01_load_data.py
python 04_build_star_schema.py
python 05_train_models.py
python 06_generate_risk_scores.py
```

Then open `powerbi/UPI_Fraud_Risk_Dashboard.pbix` in Power BI Desktop (Python-script data sources will need the file path in the query updated to your local path).

## Future Improvements

- Replace coefficient-based reason codes with full SHAP values for tree-based models
- Add a simple Flask/FastAPI endpoint to serve risk scores in near real-time
- Expand the date range and validate against multiple years of data
- Hyperparameter tuning via cross-validation (kept minimal here to prioritize the end-to-end pipeline)

## Author

**Paras Sharma**
Data Analyst | SQL · Python · Power BI · Excel
[LinkedIn](www.linkedin.com/in/paras-sharma-data-analyst) · [Email](paras07sharma07@gmail.com)

## Acknowledgements

Dataset inspired by India's UPI ecosystem operated by NPCI. All data used is synthetic and does not represent any real financial institution, user, or transaction.
