# src/models/shap_explainer.py
"""
SHAP Explainability Module (Section 8.3)
- TreeExplainer for XGBoost/LightGBM/RandomForest
- Per-customer risk factor decomposition
- Batch SHAP summary for dashboard
"""

import shap
import pandas as pd
import numpy as np
import joblib
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ENSEMBLE_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'ensemble_model.pkl')


class SHAPExplainer:
    """SHAP-based explainability for the ensemble model."""

    def __init__(self, ensemble_path=ENSEMBLE_PATH):
        self.ensemble_data = joblib.load(ensemble_path)
        self.models = self.ensemble_data['models']
        self.feature_names = self.ensemble_data['feature_names']
        self.scaler = self.ensemble_data['scaler']
        self.weights = self.ensemble_data.get('weights', {
            'xgboost': 0.35, 'lightgbm': 0.30,
            'random_forest': 0.20, 'logistic': 0.15
        })

        # Build TreeExplainer for XGBoost (primary model, highest weight)
        self.xgb_explainer = shap.TreeExplainer(self.models['xgboost'])

        # Optional: TreeExplainer for LightGBM
        self.lgbm_explainer = None
        if 'lightgbm' in self.models:
            try:
                self.lgbm_explainer = shap.TreeExplainer(self.models['lightgbm'])
            except Exception:
                pass  # Fall back to XGBoost-only SHAP

    def _prepare_row(self, row_dict):
        """Convert a customer dict/Series to model-ready DataFrame."""
        if isinstance(row_dict, pd.Series):
            row_dict = row_dict.to_dict()

        df = pd.DataFrame([row_dict])

        # Keep only model features
        available = [f for f in self.feature_names if f in df.columns]
        missing = [f for f in self.feature_names if f not in df.columns]

        X = df[available].copy()
        for col in missing:
            X[col] = 0  # Fill missing features with 0

        X = X[self.feature_names]  # Ensure correct order

        # Encode categoricals
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes

        X = X.fillna(-999).select_dtypes(include=[np.number])

        # Ensure all feature names present
        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[self.feature_names]

        return X

    def explain_customer(self, row_dict, top_n=10):
        """
        Explain a single customer's risk prediction.

        Returns dict with:
          - base_value: model's average prediction
          - risk_probability: predicted default probability
          - top_risk_up: features pushing risk UP (list of {feature, impact, value})
          - top_risk_down: features pushing risk DOWN
          - all_shap_values: full SHAP array
        """
        X = self._prepare_row(row_dict)

        # XGBoost SHAP values
        shap_values = self.xgb_explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification: [class_0, class_1]
            sv = shap_values[1][0]  # class 1 (default)
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = shap_values
        else:
            sv = np.array(shap_values)[0]

        base_value = float(self.xgb_explainer.expected_value)
        if isinstance(self.xgb_explainer.expected_value, (list, np.ndarray)):
            base_value = float(self.xgb_explainer.expected_value[1])

        # Build feature impact table
        impacts = pd.DataFrame({
            'feature': self.feature_names[:len(sv)],
            'shap_value': sv,
            'feature_value': X.iloc[0].values[:len(sv)]
        })

        # Split into risk-up and risk-down
        risk_up = impacts[impacts['shap_value'] > 0].nlargest(top_n, 'shap_value')
        risk_down = impacts[impacts['shap_value'] < 0].nsmallest(top_n, 'shap_value')

        # Get prediction
        prob = float(self.models['xgboost'].predict_proba(X)[:, 1][0])

        return {
            'base_value': base_value,
            'risk_probability': prob,
            'risk_score_0_to_100': round(prob * 100, 1),
            'top_risk_up': [
                {'feature': r['feature'], 'impact': round(float(r['shap_value']), 4),
                 'value': round(float(r['feature_value']), 2)}
                for _, r in risk_up.iterrows()
            ],
            'top_risk_down': [
                {'feature': r['feature'], 'impact': round(float(r['shap_value']), 4),
                 'value': round(float(r['feature_value']), 2)}
                for _, r in risk_down.iterrows()
            ],
            'all_shap_values': sv.tolist()
        }

    def explain_batch(self, df, max_customers=1000):
        """
        Compute SHAP values for a batch of customers.

        Returns DataFrame with SHAP values for each feature per customer.
        """
        # Prepare data
        drop_cols = ['LoanID', 'Default', 'name', 'days_to_default',
                     'risk_score_v2', 'risk_band', 'risk_category']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        band_cols = [c for c in X.columns if c.endswith('_band')]
        X = X.drop(columns=band_cols, errors='ignore')

        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
        X = X.fillna(-999).select_dtypes(include=[np.number])

        # Align to model features
        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[self.feature_names]

        # Subsample if too large
        if len(X) > max_customers:
            X = X.sample(max_customers, random_state=42)

        # Compute SHAP
        shap_values = self.xgb_explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        shap_df = pd.DataFrame(sv, columns=self.feature_names[:sv.shape[1]],
                               index=X.index)
        return shap_df

    def get_global_feature_importance(self, df, max_customers=5000):
        """
        Get mean absolute SHAP values across customers (global importance).
        Returns sorted DataFrame of feature name → mean |SHAP|.
        """
        shap_df = self.explain_batch(df, max_customers=max_customers)
        importance = shap_df.abs().mean().sort_values(ascending=False)
        return importance.reset_index().rename(
            columns={'index': 'feature', 0: 'mean_abs_shap'}
        )


if __name__ == "__main__":
    # Demo: explain a single customer
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scored_loan_data.parquet')
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'featured_loan_data.parquet')
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'enriched_loan_data.parquet')

    print(f"Loading: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded: {len(df):,} rows")

    explainer = SHAPExplainer()
    print("✓ SHAP Explainer initialized")

    # Explain first customer
    row = df.iloc[0]
    result = explainer.explain_customer(row)

    print(f"\n{'='*60}")
    print(f"CUSTOMER EXPLANATION (LoanID: {row.get('LoanID', 'N/A')})")
    print(f"{'='*60}")
    print(f"Base value: {result['base_value']:.4f}")
    print(f"Risk probability: {result['risk_probability']:.4f}")
    print(f"Risk score (0-100): {result['risk_score_0_to_100']}")

    print(f"\nFeatures pushing risk UP:")
    for f in result['top_risk_up'][:5]:
        print(f"  ↑ {f['feature']:35s} impact={f['impact']:+.4f}  value={f['value']:.2f}")

    print(f"\nFeatures pushing risk DOWN:")
    for f in result['top_risk_down'][:5]:
        print(f"  ↓ {f['feature']:35s} impact={f['impact']:+.4f}  value={f['value']:.2f}")

    # Global importance
    print(f"\n{'='*60}")
    print("GLOBAL FEATURE IMPORTANCE (top 15)")
    print(f"{'='*60}")
    importance = explainer.get_global_feature_importance(df, max_customers=1000)
    print(importance.head(15).to_string(index=False))

    # Save SHAP importance
    out_path = os.path.join(PROJECT_ROOT, 'data', 'models', 'shap_feature_importance.csv')
    importance.to_csv(out_path, index=False)
    print(f"\n✓ SHAP importance saved to: {out_path}")
