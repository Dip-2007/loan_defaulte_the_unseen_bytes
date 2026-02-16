# config/settings.py
"""
Centralized configuration for the Pre-Delinquency Engine.
Single source of truth for all thresholds, weights, paths, and parameters.
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# Dataset paths
RAW_LOAN_CSV = os.path.join(RAW_DATA_DIR, 'Loan_default.csv')
ENRICHED_PARQUET = os.path.join(PROCESSED_DATA_DIR, 'enriched_loan_data.parquet')
FEATURED_PARQUET = os.path.join(PROCESSED_DATA_DIR, 'featured_loan_data.parquet')
SCORED_PARQUET = os.path.join(PROCESSED_DATA_DIR, 'scored_loan_data.parquet')

# Model paths
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, 'feature_importance.csv')


# ============================================================
# RISK SCORING (Section 5)
# ============================================================

# 5.2 Component weights (sum = 1.00)
COMPONENT_WEIGHTS = {
    'income_stability':     0.12,
    'debt_burden':          0.15,
    'savings_adequacy':     0.08,
    'payment_history':      0.12,
    'emi_cushion':          0.08,
    'expenditure_pattern':  0.07,
    'cash_flow':            0.05,
    'credit_score_adj':     0.07,
    'employer_health':      0.05,
    'healthcare_costs':     0.03,
    'network_risk':         0.02,
    'behavioral_signals':   0.05,
    'life_events':          0.04,
    'age_vintage':          0.02,
    'external_factors':     0.05,
}

# 5.4 Risk band thresholds
RISK_BANDS = [
    {'min': 0,  'max': 25,  'band': 'SAFE',      'default_prob': '< 5%',  'action': 'Monitor only'},
    {'min': 26, 'max': 45,  'band': 'LOW RISK',   'default_prob': '5-15%', 'action': 'Gentle nudge'},
    {'min': 46, 'max': 60,  'band': 'MODERATE',    'default_prob': '15-35%','action': 'Proactive outreach'},
    {'min': 61, 'max': 75,  'band': 'HIGH RISK',   'default_prob': '35-60%','action': 'Urgent restructuring'},
    {'min': 76, 'max': 100, 'band': 'CRITICAL',    'default_prob': '> 60%', 'action': 'Immediate intervention'},
]


# ============================================================
# ENSEMBLE MODEL (Section 4.1)
# ============================================================

ENSEMBLE_WEIGHTS = {
    'xgboost':       0.35,
    'lightgbm':      0.30,
    'random_forest': 0.20,
    'logistic':      0.15,
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'eval_metric': 'auc',
    'use_label_encoder': False,
}


# ============================================================
# WORKFLOW (Section 6)
# ============================================================

# Intervention channels by risk band
INTERVENTION_CHANNELS = {
    'SAFE':      None,        # No intervention
    'LOW RISK':  'email',     # Gentle nudge
    'MODERATE':  'sms',       # Proactive outreach
    'HIGH RISK': 'phone',     # Urgent call
    'CRITICAL':  'phone',     # Immediate
}

# Workflow schedule (24-hour format)
WORKFLOW_SCHEDULE = {
    'step_1_ingest':       '01:00',
    'step_2_preprocess':   '02:00',
    'step_3_features':     '02:30',
    'step_4_inference':    '03:00',
    'step_5_stratify':     '04:00',
    'step_6_intervene':    '05:00',
    'step_7_monitor':      'continuous',
}


# ============================================================
# FEATURE ENGINEERING (Section 4.2)
# ============================================================

# ISI segment baseline coefficients of variation
SEGMENT_CV = {
    'EMPLOYED':       0.03,
    'BUSINESS_OWNER': 0.35,
    'SELF_EMPLOYED':  0.45,
    'RETIRED':        0.02,
    'STUDENT':        0.50,
}

# DTI risk bands
DTI_BANDS = [
    (0,  30, 'Low Risk'),
    (30, 40, 'Moderate Risk'),
    (40, 50, 'High Risk'),
    (50, 200, 'Critical Risk'),
]


# ============================================================
# API SETTINGS
# ============================================================

API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_VERSION = '2.0.0'


# ============================================================
# DASHBOARD
# ============================================================

DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '8050'))
DASHBOARD_DEBUG = os.getenv('DASHBOARD_DEBUG', 'False').lower() == 'true'
