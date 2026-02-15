# src/models/xgboost_model.py
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
        """Prepare features and target"""
        
        feature_cols = [
            'age', 'monthly_income', 'dti_ratio', 'credit_score',
            'app_login_duration_sec', 'notification_dismissal_speed_sec',
            'statement_open_rate', 'password_resets_30d',
            'subscription_cascade_phase', 'discretionary_spending_change_pct',
            'instant_cashouts_month', 'p2p_borrow_requests',
            'salary_delay_days', 'balance_trajectory_pct',
            'merchant_downgrade_score', 'healthcare_spending_multiplier',
            'cash_deposit_pattern_score', 'employer_health_score'
        ]
        
        X = df[feature_cols].copy()
        y = df['target']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, df):
        """Train XGBoost model"""
        
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Default rate in training: {y_train.mean()*100:.2f}%")
        
        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        print("\nTraining XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{'='*50}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"AUC Score: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(importance_df.head(10))
        
        return self.model, importance_df
    
    def save_model(self, path='data/models/xgboost_model.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"\n✓ Model saved to: {path}")

# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_parquet('data/raw/synthetic_predelinquency_data.parquet')
    
    # Train model
    model_trainer = XGBoostPreDelinquencyModel()
    model, importance = model_trainer.train(df)
    
    # Save model
    model_trainer.save_model()
    
    # Save feature importance
    importance.to_csv('data/models/feature_importance.csv', index=False)
    print("✓ Feature importance saved to: data/models/feature_importance.csv")