import pandas as pd
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

conn = sqlite3.connect(DATA_DIR / "upi_fraud.db")
transactions = pd.read_sql_query("SELECT * FROM transactions", conn)

# 1) DIM_DATE banao — unique dates nikalo transactions se
transactions['date'] = pd.to_datetime(transactions['date'])
unique_dates = pd.DataFrame({'full_date': transactions['date'].unique()})
unique_dates['date_id'] = unique_dates['full_date'].dt.strftime('%Y%m%d')
unique_dates['day_of_week'] = unique_dates['full_date'].dt.day_name()
unique_dates['month'] = unique_dates['full_date'].dt.month_name()
unique_dates['quarter'] = unique_dates['full_date'].dt.quarter
unique_dates['is_weekend'] = unique_dates['full_date'].dt.dayofweek.isin([5, 6]).astype(int)
unique_dates.to_sql('dim_date', conn, if_exists='replace', index=False)

# 2) FACT_TRANSACTIONS banao — slim table, sirf keys + measures
transactions['date_id'] = transactions['date'].dt.strftime('%Y%m%d')
fact_columns = ['transaction_id', 'user_id', 'receiver_id', 'date_id',
                 'amount', 'hour_of_day', 'is_night_transaction',
                 'new_device_flag', 'ip_location_mismatch',
                 'transaction_velocity', 'amount_deviation_score', 'is_fraud']
fact_transactions = transactions[fact_columns]
fact_transactions.to_sql('fact_transactions', conn, if_exists='replace', index=False)

print("dim_date rows:", len(unique_dates))
print("fact_transactions rows:", len(fact_transactions))
print("\nStar schema ready: fact_transactions + dim_users + dim_merchants + dim_date")

conn.close()