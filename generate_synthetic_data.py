import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, mode

np.random.seed(42)
random.seed(42)
n_users = 25000

df = pd.DataFrame()
df['user_id'] = ['U' + str(i).zfill(5) for i in range(n_users)]
df['churned'] = np.random.choice([0, 1], size=n_users, p=[0.8, 0.2])

feature_groups = {
    "Transactions": 12,
    "Usage": 12,
    "Demographics": 11,
    "TimeFeatures": 10,
    "Engineered": 10,
    "Noise": 5
}

for i in range(1, feature_groups["Transactions"] + 1):
    base = np.random.poisson(lam=5, size=n_users)
    effect = df['churned'] * np.random.normal(loc=2, scale=1, size=n_users)
    df[f'order_metric_{i}'] = base + effect

for i in range(1, feature_groups["Usage"] + 1):
    base = np.random.beta(a=2, b=5, size=n_users) * 10
    effect = df['churned'] * np.random.normal(loc=-1, scale=0.5, size=n_users)
    df[f'usage_metric_{i}'] = base + effect

for i in range(1, feature_groups["Demographics"] + 1):
    if i % 3 == 0:
        df[f'demo_feat_{i}'] = np.random.choice([0, 1, 2, 3], size=n_users)
    else:
        df[f'demo_feat_{i}'] = np.random.normal(loc=30, scale=12, size=n_users)

for i in range(1, feature_groups["TimeFeatures"] + 1):
    df[f'time_feature_{i}'] = np.random.exponential(scale=10, size=n_users) - df['churned'] * np.random.normal(3, 1, n_users)

for i in range(1, feature_groups["Engineered"] + 1):
    a = random.randint(1, feature_groups["Transactions"])
    b = random.randint(1, feature_groups["Usage"])
    df[f'composite_feature_{i}'] = (
        0.6 * df[f'order_metric_{a}'] + 0.4 * df[f'usage_metric_{b}'] + np.random.normal(0, 1, n_users)
    )

for i in range(1, feature_groups["Noise"] + 1):
    df[f'noise_feature_{i}'] = np.random.normal(0, 1, n_users)

columns = [
    'user_id', 'churned',
    'avg_order_value', 'total_orders', 'days_since_last_order', 'monthly_spend', 'order_frequency', 'cart_abandon_rate',
    'coupon_usage_rate', 'refund_count', 'items_per_order', 'total_spend', 'reorder_rate',
    'app_opens_per_day', 'session_duration_avg', 'days_active_last_30',
    'pages_visited_per_session', 'search_frequency', 'product_reviews_written',
    'payment_failures_count', 'support_chats_initiated', 'video_content_viewed', 'feature_usage_score',
    'age', 'gender', 'income_bracket', 'tenure_months', 'city_tier', 'household_size', 'marital_status',
    'has_children', 'education_level', 'employment_status', 'owns_vehicle',
    'avg_time_between_orders', 'days_since_account_creation', 'peak_order_hour', 'weekend_order_ratio',
    'holiday_purchase_ratio', 'first_purchase_time', 'last_app_open_hour', 'last_login_days_ago',
    'active_night_user', 'long_gap_before_churn', 'avg_spend_per_minute', 'value_per_order',
    'engagement_score', 'loyalty_score', 'discount_dependency', 'churn_risk_index', 'social_sharing_index',
    'mobile_vs_web_ratio', 'complexity_of_orders', 'return_rate_composite',
    'noise_1', 'noise_2', 'noise_3', 'noise_4',
    'area_locality', 'payment_method_preference', 'preferred_delivery_slot', 'loyalty_tier'
]
df.columns = columns

features_to_remove = [
    'cash_on_delivery_pct', 'wishlist_items_count', 'push_notifications_click_rate', 'noise_5'
]
existing_to_remove = [col for col in features_to_remove if col in df.columns]
if existing_to_remove:
    df = df.drop(columns=existing_to_remove)

localities = ["Whitefield", "Indiranagar", "Koramangala", "HSR Layout", "Jayanagar", "Malleshwaram"]
df["area_locality"] = np.random.choice(localities, size=n_users)
df['payment_method_preference'] = np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash'], size=n_users)
df['preferred_delivery_slot'] = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], size=n_users)
df['loyalty_tier'] = np.random.choice(['Membership', 'No Membership'], size=n_users)

columns = [
    'user_id', 'churned',
    'avg_order_value', 'total_orders', 'days_since_last_order', 'monthly_spend', 'order_frequency', 'cart_abandon_rate',
    'coupon_usage_rate', 'refund_count', 'items_per_order', 'total_spend', 'reorder_rate',
    'app_opens_per_day', 'session_duration_avg', 'days_active_last_30',
    'pages_visited_per_session', 'search_frequency', 'product_reviews_written',
    'payment_failures_count', 'support_chats_initiated', 'video_content_viewed', 'feature_usage_score',
    'age', 'gender', 'income_bracket', 'tenure_months', 'city_tier', 'household_size', 'marital_status',
    'has_children', 'education_level', 'employment_status', 'owns_vehicle',
    'avg_time_between_orders', 'days_since_account_creation', 'peak_order_hour', 'weekend_order_ratio',
    'holiday_purchase_ratio', 'first_purchase_time', 'last_app_open_hour', 'last_login_days_ago',
    'active_night_user', 'long_gap_before_churn', 'avg_spend_per_minute', 'value_per_order',
    'engagement_score', 'loyalty_score', 'discount_dependency', 'churn_risk_index', 'social_sharing_index',
    'mobile_vs_web_ratio', 'complexity_of_orders', 'return_rate_composite',
    'noise_1', 'noise_2', 'noise_3', 'noise_4',
    'area_locality', 'payment_method_preference', 'preferred_delivery_slot', 'loyalty_tier'
]
df = df[[col for col in columns if col in df.columns]]

feature_names = columns

df.columns = feature_names


int_columns = [
    'days_since_last_order', 'total_orders', 'items_per_order', 'refund_count',
    'payment_failures_count', 'household_size', 'tenure_months', 'days_active_last_30',
    'last_login_days_ago', 'days_since_account_creation', 'peak_order_hour', 'last_app_open_hour'
]
for col in int_columns:
    if col in df.columns:
        df[col] = np.ceil(df[col]).astype(int)

# Make some continuous features have repeated values to ensure mode is not empty
for col in ['avg_order_value', 'monthly_spend', 'session_duration_avg', 'engagement_score']:
    if col in df.columns:
        idx = df.sample(frac=0.2, random_state=42).index
        df.loc[idx, col] = df.loc[idx, col].round(0)

# Save the main Excel file
df.to_excel("synthetic_churn_data.xlsx", index=False)
print("Excel file 'synthetic_churn_data.xlsx' created!")
print(" Shape of dataset:", df.shape)

# --- DATA CLEANING SECTION ---
# Remove duplicate rows (if any)
df_clean = df.drop_duplicates()

# Handle missing values
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        # Fill missing categorical/text with mode
        mode_val = df_clean[col].mode()
        fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
        df_clean[col] = df_clean[col].fillna(fill_val)
    else:
        # Fill missing numeric with median
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Ensure correct dtypes for integer columns
for col in int_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(int)

# Save cleaned data
cleaned_path = 'synthetic_churn_data_cleaned.xlsx'
df_clean.to_excel(cleaned_path, index=False)
print(f'Cleaned data saved to {cleaned_path}')

# Generate summary statistics and plots
summary = []
for col in df.columns:
    col_data = df[col]
    col_type = 'categorical' if col_data.dtype == 'object' or col_data.nunique() < 10 else 'continuous'
    stats = {'feature': col, 'type': col_type}
    if col_type == 'continuous':
        stats['mean'] = col_data.mean()
        stats['median'] = col_data.median()
        m = mode(col_data, nan_policy='omit')
        stats['mode'] = m.mode[0] if hasattr(m.mode, '__len__') and len(m.mode) > 0 else 'No mode'
        stats['std'] = col_data.std()
        stats['q1'] = col_data.quantile(0.25)
        stats['q2'] = col_data.quantile(0.5)
        stats['q3'] = col_data.quantile(0.75)
        stats['q4'] = col_data.quantile(1.0)
        stats['p5'] = col_data.quantile(0.05)
        stats['p95'] = col_data.quantile(0.95)
        stats['skewness'] = skew(col_data.dropna()) if col_data.dropna().size > 0 else None
        stats['kurtosis'] = kurtosis(col_data.dropna()) if col_data.dropna().size > 0 else None
        stats['missing'] = col_data.isnull().sum()
        plt.figure(figsize=(6,3))
        sns.boxplot(x=df['churned'], y=col_data)
        plt.title(f'{col} vs Churned')
        plt.xlabel('Churned')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f'stats_{col}_vs_churned.png')
        plt.close()
    else:
        m = col_data.mode()
        stats['mean'] = None
        stats['median'] = None
        stats['mode'] = m.iloc[0] if not m.empty else 'No mode'
        stats['std'] = None
        stats['q1'] = None
        stats['q2'] = None
        stats['q3'] = None
        stats['q4'] = None
        stats['p5'] = None
        stats['p95'] = None
        stats['skewness'] = None
        stats['kurtosis'] = None
        stats['unique'] = col_data.nunique()
        stats['missing'] = col_data.isnull().sum()
        plt.figure(figsize=(7,4))
        try:
            sns.countplot(x=col_data, hue=df['churned'])
            plt.title(f'{col} vs Churned')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.legend(title='Churned')
            plt.tight_layout()
            plt.savefig(f'stats_{col}_vs_churned.png')
            plt.close()
        except Exception as e:
            print(f'Could not plot {col}: {e}')
    summary.append(stats)

summary_df = pd.DataFrame(summary)
cols = summary_df.columns.tolist()
for colname in ['mean', 'median', 'mode']:
    cols.remove(colname)
summary_df = summary_df[['feature', 'type', 'mean', 'median', 'mode'] + cols[2:]]
summary_df = summary_df.iloc[2:].reset_index(drop=True)

# Ensure the last 3 features are included in the summary statistics
for feature in ['area_locality', 'payment_method_preference', 'preferred_delivery_slot', 'loyalty_tier']:
    if feature not in summary_df['feature'].values:
        col_data = df[feature]
        col_type = 'categorical'
        stats = {'feature': feature, 'type': col_type}
        m = col_data.mode()
        stats['mean'] = None
        stats['median'] = None
        stats['mode'] = m.iloc[0] if not m.empty else 'No mode'
        stats['std'] = None
        stats['q1'] = None
        stats['q2'] = None
        stats['q3'] = None
        stats['q4'] = None
        stats['p5'] = None
        stats['p95'] = None
        stats['skewness'] = None
        stats['kurtosis'] = None
        stats['unique'] = col_data.nunique()
        stats['missing'] = col_data.isnull().sum()
        summary_df = pd.concat([summary_df, pd.DataFrame([stats])], ignore_index=True)

summary_df.to_excel('feature_summary_statistics.xlsx', index=False)
print('Summary statistics saved to feature_summary_statistics.xlsx')

def explore_feature(feature_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if feature_name not in df.columns:
        print(f"Feature '{feature_name}' not found in DataFrame.")
        return
    col_data = df[feature_name]
    print(f"\nSummary statistics for '{feature_name}':")
    print(col_data.describe())
    if col_data.dtype != 'object':
        print("Skewness:", col_data.skew())
        print("Kurtosis:", col_data.kurtosis())
    print("Missing:", col_data.isnull().sum())
    if feature_name != 'churned':
        plt.figure(figsize=(6,3))
        try:
            sns.boxplot(x=df['churned'], y=col_data)
            plt.title(f'{feature_name} vs Churned')
            plt.xlabel('Churned')
            plt.ylabel(feature_name)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot {feature_name}: {e}")
    else:
        print("No plot for target variable itself.")

feature_dict = pd.read_csv('feature_data_dictionary.csv')
