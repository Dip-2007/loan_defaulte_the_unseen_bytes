import pandas as pd
import numpy as np
from faker import Faker
import datetime
import os
import joblib

fake = Faker('en_IN')

# Use relative path from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Priority order: scored → featured → enriched
DATA_PATHS = [
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'scored_loan_data_v2.parquet'),
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'scored_loan_data.parquet'),
    os.path.join(PROJECT_ROOT, 'data', 'processed', 'featured_loan_data.parquet'),
]
ENSEMBLE_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'ensemble_model.pkl')
XGB_MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'xgboost_model.pkl')
IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'ensemble_feature_importance.csv')
SHAP_IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'shap_feature_importance.csv')

# 5-Band Risk Classification (Section 5.4)
RISK_BANDS = [
    (0,  25, 'SAFE',      '#00e676', '< 5%'),
    (26, 45, 'LOW RISK',  '#2979ff', '5-15%'),
    (46, 60, 'MODERATE',  '#ffab00', '15-35%'),
    (61, 75, 'HIGH RISK', '#ff6d00', '35-60%'),
    (76, 100, 'CRITICAL', '#ff1744', '> 60%'),
]

BAND_COLORS = {b[2]: b[3] for b in RISK_BANDS}


def classify_risk_band(score):
    """Classify a 0-100 risk score into 5-band risk level."""
    for lo, hi, band, _, _ in RISK_BANDS:
        if lo <= score <= hi:
            return band
    return 'CRITICAL' if score > 100 else 'SAFE'


def load_data():
    """Loads the best available dataset with risk scores and 5-band classification."""
    # Find best available data
    df = None
    for path in DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"[data_loader] Loaded: {path} ({len(df):,} rows, {len(df.columns)} cols)")
            break

    if df is None:
        raise FileNotFoundError(
            f"No data found. Run the pipeline first.\nSearched: {DATA_PATHS}"
        )

    # Add customer names (for display)
    if 'name' not in df.columns:
        np.random.seed(42)
        df['name'] = [fake.name() for _ in range(len(df))]

    # --- Risk Score ---
    # Use risk_score_v2 from risk_scorer.py if available
    if 'risk_score_v2' in df.columns:
        df['risk_score'] = df['risk_score_v2']
    elif 'risk_score' not in df.columns:
        # Generate from model
        if os.path.exists(ENSEMBLE_PATH):
            try:
                ensemble_data = joblib.load(ENSEMBLE_PATH)
                feature_names = ensemble_data['feature_names']
                models = ensemble_data['models']
                scaler = ensemble_data['scaler']
                weights = ensemble_data.get('weights', {
                    'xgboost': 0.35, 'lightgbm': 0.30,
                    'random_forest': 0.20, 'logistic': 0.15
                })

                X = df.drop(columns=['LoanID', 'Default', 'name', 'days_to_default'],
                            errors='ignore')
                band_cols = [c for c in X.columns if c.endswith('_band')]
                X = X.drop(columns=band_cols, errors='ignore')
                cat_cols = X.select_dtypes(include=['object', 'category']).columns
                for c in cat_cols:
                    X[c] = X[c].astype('category').cat.codes
                X = X.fillna(-999).select_dtypes(include=[np.number])

                # Align to model features
                for f in feature_names:
                    if f not in X.columns:
                        X[f] = 0
                X = X[feature_names]
                X_scaled = scaler.transform(X)

                # Ensemble probability
                probas = {}
                for name, model in models.items():
                    if name == 'logistic':
                        probas[name] = model.predict_proba(X_scaled)[:, 1]
                    else:
                        probas[name] = model.predict_proba(X)[:, 1]

                ensemble_prob = sum(weights[n] * probas[n] for n in probas)
                df['delinquency_probability'] = ensemble_prob
                df['risk_score'] = (ensemble_prob * 100).round(1)
                print(f"[data_loader] Risk scores generated from ensemble model")
            except Exception as e:
                print(f"[data_loader] Ensemble model failed: {e}, falling back to CreditScore")
                df['risk_score'] = ((900 - df['CreditScore']) / 6).clip(0, 100).round(1)
        elif os.path.exists(XGB_MODEL_PATH):
            m = joblib.load(XGB_MODEL_PATH)
            X = df.drop(columns=['LoanID', 'Default', 'name'], errors='ignore')
            cat_cols = X.select_dtypes(include=['object']).columns
            for c in cat_cols:
                X[c] = X[c].astype('category').cat.codes
            X = X.fillna(-999).select_dtypes(include=[np.number])
            df['delinquency_probability'] = m['model'].predict_proba(X)[:, 1]
            df['risk_score'] = (df['delinquency_probability'] * 100).round(1)
        else:
            # Fallback: use CreditScore inverse
            df['risk_score'] = ((900 - df['CreditScore']) / 6).clip(0, 100).round(1)

    # --- 5-Band Risk Classification ---
    if 'risk_band' in df.columns:
        df['risk_category'] = df['risk_band']
    else:
        df['risk_category'] = df['risk_score'].apply(classify_risk_band)

    # Use existing column mappings
    if 'employment_type' not in df.columns and 'EmploymentType' in df.columns:
        df['employment_type'] = df['EmploymentType']

    if 'monthly_income' not in df.columns and 'Income' in df.columns:
        df['monthly_income'] = df['Income']

    return df.copy()


def load_feature_importance():
    """Load feature importance (prefer SHAP, fallback to tree importance)."""
    if os.path.exists(SHAP_IMPORTANCE_PATH):
        return pd.read_csv(SHAP_IMPORTANCE_PATH)
    if os.path.exists(IMPORTANCE_PATH):
        return pd.read_csv(IMPORTANCE_PATH)
    return pd.DataFrame({'feature': [], 'importance': []})


def get_customer_timeline(customer_id):
    """Generates synthetic timeline data for a specific customer."""
    np.random.seed(hash(str(customer_id)) % 2**31)
    dates = pd.date_range(end=datetime.date.today(), periods=180).tolist()
    base_balance = np.random.uniform(10000, 200000)
    balance_history = []
    current_balance = base_balance

    for _ in dates:
        change = np.random.uniform(-2000, 2500)
        current_balance = max(0, current_balance + change)
        balance_history.append(current_balance)

    return pd.DataFrame({'date': dates, 'balance': balance_history})


def get_spending_breakdown(customer_id):
    """Generates synthetic spending breakdown."""
    np.random.seed(hash(str(customer_id)) % 2**31)
    categories = ['Rent/Mortgage', 'Groceries', 'Utilities', 'Dining Out',
                   'Entertainment', 'Shopping', 'Healthcare', 'Transport']
    amounts = np.random.randint(500, 20000, size=len(categories))
    return pd.DataFrame({'category': categories, 'amount': amounts})
