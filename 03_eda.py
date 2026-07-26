# Exploratory Data Analysis (EDA) for Fraud Detection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "notebooks" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "transactions.csv")

sns.set_style("whitegrid")

# 1) Fraud rate by hour of day
fraud_by_hour = df.groupby('hour_of_day')['is_fraud'].mean() * 100
plt.figure(figsize=(10, 5))
fraud_by_hour.plot(kind='bar', color='#d62728')
plt.title("Fraud Rate by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fraud_by_hour.png")
plt.close()

# 2) Night vs Day transactions
fraud_by_night = df.groupby('is_night_transaction')['is_fraud'].mean() * 100
print("\nFraud rate — Night vs Day transaction:\n", fraud_by_night)

# 3) New device flag impact
fraud_by_device = df.groupby('new_device_flag')['is_fraud'].mean() * 100
print("\nFraud rate — New device vs Known device:\n", fraud_by_device)

# 4) KYC status impact
fraud_by_kyc = df.groupby('user_kyc_status')['is_fraud'].mean() * 100
print("\nFraud rate — by KYC status:\n", fraud_by_kyc)

# 5) Payment app comparison
fraud_by_app = df.groupby('payment_app')['is_fraud'].mean() * 100
print("\nFraud rate — by Payment App:\n", fraud_by_app.sort_values(ascending=False))

# 6) Amount distribution: fraud vs genuine
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='is_fraud', y='amount', showfliers=False)
plt.title("Transaction Amount: Fraud vs Genuine")
plt.xlabel("Is Fraud (0=No, 1=Yes)")
plt.ylabel("Amount (₹)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "amount_fraud_comparison.png")
plt.close()

print("\nPlots saved in notebooks/plots/ folder.")
