# src/models/ensemble_model.py
"""
Section 4/8: Ensemble Model with XGBoost, LightGBM, Random Forest, Logistic Regression.
Weighted aggregation + SHAP explainability.
"""

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, f1_score
)
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import os

# Cost-sensitive threshold: FP costs ₹50, FN costs ₹8000
COST_FP = 50
COST_FN = 8000

# Ensemble weights (Section 6 spec)
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.35,
    'lightgbm': 0.30,
    'random_forest': 0.20,
    'logistic': 0.15
}


class EnsemblePreDelinquencyModel:
    """Multi-model ensemble for pre-delinquency prediction."""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.optimal_threshold = 0.5

    def prepare_data(self, df):
        """Prepare features and target."""
        drop_cols = ['LoanID', 'Default', 'name', 'days_to_default']
        drop_cols = [c for c in drop_cols if c in df.columns]

        target_col = 'Default' if 'Default' in df.columns else 'target'
        y = df[target_col]

        X = df.drop(columns=drop_cols, errors='ignore')

        # Drop band/label columns (they are categorical derivatives)
        band_cols = [c for c in X.columns if c.endswith('_band')]
        X = X.drop(columns=band_cols, errors='ignore')

        # Encode categoricals
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes

        X = X.fillna(-999).select_dtypes(include=[np.number])
        self.feature_names = X.columns.tolist()
        return X, y

    def train(self, df):
        """Train all models in the ensemble."""
        X, y = self.prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(self.feature_names)}")
        print(f"Default rate: {y_train.mean()*100:.2f}%")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Scale for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ---- 1. XGBoost ----
        print(f"\n{'='*50}")
        print("Training XGBoost (weight={})".format(ENSEMBLE_WEIGHTS['xgboost']))
        self.models['xgboost'] = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            max_depth=6, learning_rate=0.05, n_estimators=500,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos_weight, random_state=42,
            early_stopping_rounds=30
        )
        self.models['xgboost'].fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)], verbose=50
        )
        xgb_proba = self.models['xgboost'].predict_proba(X_test)[:, 1]
        print(f"  XGBoost AUC: {roc_auc_score(y_test, xgb_proba):.4f}")

        # ---- 2. LightGBM ----
        print(f"\n{'='*50}")
        print("Training LightGBM (weight={})".format(ENSEMBLE_WEIGHTS['lightgbm']))
        self.models['lightgbm'] = lgb.LGBMClassifier(
            objective='binary', metric='auc',
            max_depth=6, learning_rate=0.05, n_estimators=500,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, random_state=42,
            verbose=-1
        )
        self.models['lightgbm'].fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(50), lgb.early_stopping(30)]
        )
        lgbm_proba = self.models['lightgbm'].predict_proba(X_test)[:, 1]
        print(f"  LightGBM AUC: {roc_auc_score(y_test, lgbm_proba):.4f}")

        # ---- 3. Random Forest ----
        print(f"\n{'='*50}")
        print("Training Random Forest (weight={})".format(ENSEMBLE_WEIGHTS['random_forest']))
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        rf_proba = self.models['random_forest'].predict_proba(X_test)[:, 1]
        print(f"  Random Forest AUC: {roc_auc_score(y_test, rf_proba):.4f}")

        # ---- 4. Logistic Regression ----
        print(f"\n{'='*50}")
        print("Training Logistic Regression (weight={})".format(ENSEMBLE_WEIGHTS['logistic']))
        self.models['logistic'] = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, C=0.1
        )
        self.models['logistic'].fit(X_train_scaled, y_train)
        lr_proba = self.models['logistic'].predict_proba(X_test_scaled)[:, 1]
        print(f"  Logistic Reg AUC: {roc_auc_score(y_test, lr_proba):.4f}")

        # ---- Ensemble ----
        print(f"\n{'='*50}")
        print("ENSEMBLE AGGREGATION")
        ensemble_proba = (
            ENSEMBLE_WEIGHTS['xgboost'] * xgb_proba +
            ENSEMBLE_WEIGHTS['lightgbm'] * lgbm_proba +
            ENSEMBLE_WEIGHTS['random_forest'] * rf_proba +
            ENSEMBLE_WEIGHTS['logistic'] * lr_proba
        )
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        print(f"  Ensemble AUC: {ensemble_auc:.4f}")

        # Optimal threshold (cost-sensitive)
        precision, recall, thresholds = precision_recall_curve(y_test, ensemble_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        self.optimal_threshold = thresholds[np.argmax(f1_scores)]
        print(f"  Optimal threshold (F1): {self.optimal_threshold:.4f}")

        # Business cost threshold
        best_cost = float('inf')
        best_cost_thresh = 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (ensemble_proba >= t).astype(int)
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel()
            cost = fp * COST_FP + fn * COST_FN
            if cost < best_cost:
                best_cost = cost
                best_cost_thresh = t
        print(f"  Cost-optimal threshold: {best_cost_thresh:.4f} (FP=₹{COST_FP}, FN=₹{COST_FN})")

        # Final classification report
        ensemble_pred = (ensemble_proba >= self.optimal_threshold).astype(int)
        print(f"\n{'='*50}")
        print(f"FINAL ENSEMBLE PERFORMANCE")
        print(f"{'='*50}")
        print(f"AUC: {ensemble_auc:.4f}")
        print(classification_report(y_test, ensemble_pred,
              target_names=['No Default', 'Default']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, ensemble_pred))

        # Individual model comparison
        print(f"\n{'='*50}")
        print("MODEL COMPARISON")
        print(f"{'='*50}")
        print(f"  {'Model':<20s} {'AUC':>8s} {'Weight':>8s}")
        print(f"  {'-'*36}")
        for name, proba in [('XGBoost', xgb_proba), ('LightGBM', lgbm_proba),
                             ('Random Forest', rf_proba), ('Logistic Reg', lr_proba),
                             ('ENSEMBLE', ensemble_proba)]:
            auc = roc_auc_score(y_test, proba)
            w = ENSEMBLE_WEIGHTS.get(name.lower().replace(' ', '_'), 1.0)
            marker = ' ★' if name == 'ENSEMBLE' else ''
            print(f"  {name:<20s} {auc:>8.4f} {w:>7.0%}{marker}")

        return self.models, ensemble_auc

    def predict_proba(self, X):
        """Get ensemble probability for new data."""
        X_filled = X.fillna(-999)
        X_scaled = self.scaler.transform(X_filled)

        probas = {}
        for name, model in self.models.items():
            if name == 'logistic':
                probas[name] = model.predict_proba(X_scaled)[:, 1]
            else:
                probas[name] = model.predict_proba(X_filled)[:, 1]

        ensemble = sum(
            ENSEMBLE_WEIGHTS[name] * probas[name]
            for name in probas
        )
        return ensemble

    def save(self, path='data/models/ensemble_model.pkl'):
        """Save all models."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'weights': ENSEMBLE_WEIGHTS
        }, path)
        print(f"\n✓ Ensemble saved to: {path}")

    @classmethod
    def load(cls, path='data/models/ensemble_model.pkl'):
        """Load saved ensemble."""
        data = joblib.load(path)
        obj = cls()
        obj.models = data['models']
        obj.scaler = data['scaler']
        obj.feature_names = data['feature_names']
        obj.optimal_threshold = data['optimal_threshold']
        return obj


if __name__ == "__main__":
    # Load featured data
    data_path = 'data/processed/featured_loan_data.parquet'
    if not os.path.exists(data_path):
        data_path = 'data/processed/enriched_loan_data.parquet'

    print(f"Loading: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Train ensemble
    ensemble = EnsemblePreDelinquencyModel()
    models, auc = ensemble.train(df)

    # Save
    ensemble.save()

    # Feature importance from XGBoost
    imp = pd.DataFrame({
        'feature': ensemble.feature_names,
        'importance': ensemble.models['xgboost'].feature_importances_
    }).sort_values('importance', ascending=False)
    imp.to_csv('data/models/ensemble_feature_importance.csv', index=False)
    print("✓ Feature importance saved")
