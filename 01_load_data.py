import pandas as pd
import sqlite3
from pathlib import Path

# Script ki apni location se data folder ka absolute path nikalo
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

users = pd.read_csv(DATA_DIR / "users.csv")
merchants = pd.read_csv(DATA_DIR / "merchants.csv")
transactions = pd.read_csv(DATA_DIR / "transactions.csv")

print("Users:", users.shape)
print("Merchants:", merchants.shape)
print("Transactions:", transactions.shape)
print("\nFraud distribution:\n", transactions['is_fraud'].value_counts())
print("Fraud rate: {:.2f}%".format(transactions['is_fraud'].mean() * 100))

conn = sqlite3.connect(DATA_DIR / "upi_fraud.db")
users.to_sql('users', conn, if_exists='replace', index=False)
merchants.to_sql('merchants', conn, if_exists='replace', index=False)
transactions.to_sql('transactions', conn, if_exists='replace', index=False)
conn.close()

print("\nDatabase ready: 3 tables — users, merchants, transactions")