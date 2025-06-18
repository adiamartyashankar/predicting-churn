import numpy as np
import pandas as pd
import random

# Configuration
np.random.seed(42)
random.seed(42)
n_users = 5000

# Step 1: Create base DataFrame
df = pd.DataFrame()
df['user_id'] = ['U' + str(i).zfill(5) for i in range(n_users)]
df['churned'] = np.random.choice([0, 1], size=n_users, p=[0.8, 0.2])  # 20% churn rate

# Step 2: Define number of features
feature_groups = {
    "Transactions": 12,
    "Usage": 12,
    "Demographics": 11,
    "TimeFeatures": 10,
    "Engineered": 10,
    "Noise": 5
}

# Step 3: Generate Features

# Transactions (e.g., order_count, avg_order_value)
for i in range(1, feature_groups["Transactions"] + 1):
    base = np.random.poisson(lam=5, size=n_users)
    effect = df['churned'] * np.random.normal(loc=2, scale=1, size=n_users)
    df[f'order_metric_{i}'] = base + effect

# Usage (e.g., app_opens, screen_time)
for i in range(1, feature_groups["Usage"] + 1):
    base = np.random.beta(a=2, b=5, size=n_users) * 10
    effect = df['churned'] * np.random.normal(loc=-1, scale=0.5, size=n_users)
    df[f'usage_metric_{i}'] = base + effect

# Demographics (e.g., age, income, city code)
for i in range(1, feature_groups["Demographics"] + 1):
    if i % 3 == 0:
        df[f'demo_feat_{i}'] = np.random.choice([0, 1, 2, 3], size=n_users)  # categorical like region
    else:
        df[f'demo_feat_{i}'] = np.random.normal(loc=30, scale=12, size=n_users)  # like age/income

# Time-based (e.g., days_since_last_order)
for i in range(1, feature_groups["TimeFeatures"] + 1):
    df[f'time_feature_{i}'] = np.random.exponential(scale=10, size=n_users) - df['churned'] * np.random.normal(3, 1, n_users)

# Engineered (e.g., derived patterns)
for i in range(1, feature_groups["Engineered"] + 1):
    a = random.randint(1, feature_groups["Transactions"])
    b = random.randint(1, feature_groups["Usage"])
    df[f'composite_feature_{i}'] = (
        0.6 * df[f'order_metric_{a}'] + 0.4 * df[f'usage_metric_{b}'] + np.random.normal(0, 1, n_users)
    )

# Noise (irrelevant features for testing robustness)
for i in range(1, feature_groups["Noise"] + 1):
    df[f'noise_feature_{i}'] = np.random.normal(0, 1, n_users)

# Step 4: Reorder columns
columns = ['user_id', 'churned'] + [col for col in df.columns if col not in ['user_id', 'churned']]
df = df[columns]

# Step 5: Save to Excel
df.to_excel("synthetic_churn_data.xlsx", index=False)
print("Excel file 'synthetic_churn_data.xlsx' created!")
print(" Shape of dataset:", df.shape)
