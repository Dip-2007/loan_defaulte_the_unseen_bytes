# src/models/xgboost_model.py
"""
XGBoost model for pre-delinquency prediction.
Trains on the enriched 149-column dataset with segment-aware features.
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import os


class XGBoostPreDelinquencyModel:
    """XGBoost model for pre-delinquency prediction"""

    def __init__(self):
        self.model = None
        self.feature_names = None

    def prepare_data(self, df):
        """Prepare features and target from enriched dataset"""

        # Drop non-feature columns
        drop_cols = ['LoanID', 'Default', 'days_to_default']
        drop_cols = [c for c in drop_cols if c in df.columns]

        # Identify target
        target_col = 'Default' if 'Default' in df.columns else 'target'
        y = df[target_col]

        # Encode categoricals
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in drop_cols]

        X = df.drop(columns=drop_cols, errors='ignore')

        # One-hot encode categoricals (or label encode for XGBoost)
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes

        # Fill NaN with -999 (XGBoost handles this natively)
        X = X.fillna(-999)

        # Drop any remaining non-numeric
        X = X.select_dtypes(include=[np.number])

        self.feature_names = X.columns.tolist()
        return X, y

    def train(self, df):
        """Train XGBoost model"""

        X, y = self.prepare_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Features: {len(self.feature_names)}")
        print(f"Default rate in training: {y_train.mean()*100:.2f}%")

        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Train model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            early_stopping_rounds=30
        )

        print(f"\nTraining XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n{'='*60}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*60}")
        print(f"AUC Score: {auc:.4f}")
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Feature importance (top 30)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 30 Important Features:")
        print(importance_df.head(30).to_string())

        return self.model, importance_df

    def save_model(self, path='data/models/xgboost_model.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)
        print(f"\n✓ Model saved to: {path}")


# Main execution
if __name__ == "__main__":
    # Try enriched dataset first, fall back to synthetic
    data_path = 'data/processed/enriched_loan_data.parquet'
    fallback_path = 'data/raw/synthetic_predelinquency_data.parquet'

    if os.path.exists(data_path):
        print(f"Loading enriched dataset: {data_path}")
        df = pd.read_parquet(data_path)
    elif os.path.exists(fallback_path):
        print(f"Loading synthetic dataset: {fallback_path}")
        df = pd.read_parquet(fallback_path)
    else:
        print("ERROR: No dataset found. Run enrich_dataset.py first.")
        exit(1)

    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Train model
    model_trainer = XGBoostPreDelinquencyModel()
    model, importance = model_trainer.train(df)

    # Save model
    model_trainer.save_model()

    # Save feature importance
    os.makedirs('data/models', exist_ok=True)
    importance.to_csv('data/models/feature_importance.csv', index=False)
    print("✓ Feature importance saved to: data/models/feature_importance.csv")