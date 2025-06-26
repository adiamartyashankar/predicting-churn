import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# 1. Load cleaned data
df = pd.read_excel('synthetic_churn_data_cleaned.xlsx')

# 2. Drop non-informative columns
df = df.drop(['user_id'], axis=1)

# 3. Encode categorical variables
def encode_categoricals(df):
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

df_encoded = encode_categoricals(df)

# 4. Feature selection using correlation with target (churned)
correlations = df_encoded.corr()['churned'].abs().sort_values(ascending=False)
# Exclude the target itself
selected_features = correlations.index[1:16].tolist()  # top 15 features
print('Top features used for logistic regression:')
print(selected_features)

X = df_encoded[selected_features]
y = df_encoded['churned']

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train logistic regression
logreg = LogisticRegression(max_iter=1000, solver='lbfgs')
logreg.fit(X_train_scaled, y_train)

def save_metrics_to_file(iteration, accuracy, report, matrix, filename="logreg_results.txt"):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as f:
        f.write(f"\n===== Iteration {iteration} =====\n")
        f.write(f"Logistic Regression Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))
        f.write("\n========================\n")

iteration_file = "logreg_results.txt"
if not os.path.exists(iteration_file):
    iteration = 1
else:
    with open(iteration_file, 'r') as f:
        lines = f.readlines()
        iteration = sum(1 for line in lines if line.startswith('===== Iteration')) + 1

y_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(report)
print('\nConfusion Matrix:')
print(matrix)
save_metrics_to_file(iteration, accuracy, report, matrix, iteration_file)

