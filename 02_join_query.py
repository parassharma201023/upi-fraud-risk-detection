import pandas as pd
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

conn = sqlite3.connect(DATA_DIR / "upi_fraud.db")

query = """
SELECT 
    m.merchant_category,
    m.city_tier,
    COUNT(*) as total_txns,
    SUM(t.is_fraud) as fraud_txns,
    ROUND(100.0 * SUM(t.is_fraud) / COUNT(*), 2) as fraud_rate_pct
FROM transactions t
JOIN merchants m ON t.receiver_id = m.merchant_id
GROUP BY m.merchant_category, m.city_tier
ORDER BY fraud_rate_pct DESC
LIMIT 15;
"""

result = pd.read_sql_query(query, conn)
print(result.to_string(index=False))

conn.close()