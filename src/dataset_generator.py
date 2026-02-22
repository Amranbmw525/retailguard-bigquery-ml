import pandas as pd
import numpy as np

# Generate 5,000 rows of retail return data
np.random.seed(42)
data = {
    'product_id': np.random.randint(1000, 1050, 5000),
    'return_rate': np.random.uniform(0.01, 0.08, 5000), # Normal 1-8%
    'avg_transaction_amount': np.random.normal(150, 50, 5000),
    'days_since_purchase': np.random.randint(1, 90, 5000)
}
df = pd.DataFrame(data)

# Inject 50 "Anomalies" (High return rates + high amounts)
anomalies = pd.DataFrame({
    'product_id': [9999] * 50,
    'return_rate': np.random.uniform(0.30, 0.50, 50), # Suspicious 30-50%
    'avg_transaction_amount': np.random.uniform(800, 1200, 50),
    'days_since_purchase': np.random.randint(1, 5, 50)
})

df = pd.concat([df, anomalies]).sample(frac=1).reset_index(drop=True)
df.to_csv('returns_data.csv', index=False)
print("File 'returns_data.csv' created!")