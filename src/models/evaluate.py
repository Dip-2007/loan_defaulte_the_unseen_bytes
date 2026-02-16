# src/models/evaluate.py
"""Quick model evaluation script"""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, accuracy_score, f1_score
)

# Load data
df = pd.read_parquet('data/processed/enriched_loan_data.parquet')
X = df.drop(columns=['LoanID', 'Default'], errors='ignore')

# Encode categoricals
for c in X.select_dtypes(include=['object']).columns:
    X[c] = X[c].astype('category').cat.codes
X = X.fillna(-999).select_dtypes(include=[np.number])
y = df['Default']

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load model
m = joblib.load('data/models/xgboost_model.pkl')
y_proba = m['model'].predict_proba(X_test)[:, 1]
y_pred = m['model'].predict(X_test)

# Results
print("=" * 55)
print("  MODEL PERFORMANCE (Test Set - 20%)")
print("=" * 55)
print(f"  Test samples   : {len(X_test):,}")
print(f"  Features       : {len(m['feature_names'])}")
print(f"  Accuracy       : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  AUC-ROC        : {roc_auc_score(y_test, y_proba):.4f}")
print(f"  F1 Score       : {f1_score(y_test, y_pred):.4f}")
print("=" * 55)
print()
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
