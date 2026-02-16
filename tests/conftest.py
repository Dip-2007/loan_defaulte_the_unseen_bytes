# tests/conftest.py
"""
Shared pytest fixtures for Pre-Delinquency Engine tests.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_customer_row():
    """Minimal customer row for testing."""
    return pd.Series({
        'LoanID': 'TEST001',
        'Age': 32,
        'Income': 85000,
        'LoanAmount': 500000,
        'CreditScore': 720,
        'MonthsEmployed': 36,
        'NumCreditLines': 3,
        'InterestRate': 12.0,
        'LoanTerm': 36,
        'DTIRatio': 23.5,
        'HasCoSigner': 0,
        'Default': 0,
        'segment_category': 'EMPLOYED',
        'detailed_segment': 'Private Sector',
        'EmploymentType': 'Full-time',
        'MaritalStatus': 'Married',
        'HasDependents': 1,
        'num_dependents': 1,
        'Education': 'Bachelor',
        'ontime_payment_rate_12m': 0.83,
        'avg_delay_days': 1.0,
        'max_dpd_last_12m': 7,
        'num_dpd_30_plus': 0,
        'avg_monthly_balance_6m': 28000,
        'total_outstanding_debt': 300000,
        'credit_utilization_ratio': 40,
        'savings_rate': 33.0,
        'total_monthly_expense': 45000,
        'expense_housing': 15000,
        'expense_food_groceries': 8000,
        'expense_transportation': 5000,
        'expense_healthcare': 6000,
        'expense_lifestyle_entertainment': 8000,
        'expense_discretionary_vices': 3000,
        'salary_delay_days': 0,
        'employer_health_score': 85,
        'peer_default_rate': 0.02,
        'industry_stress_index': 0.1,
        'pharmacy_expense_monthly': 6000,
        'hospital_visits_6m': 2,
        'health_insurance_coverage': 500000,
        'app_login_frequency_month': 15,
        'subscription_cascade_phase': 1,
        'p2p_borrow_requests_30d': 0,
        'instant_cashouts_month': 2,
        'payment_day_consistency': 0.85,
        'auto_debit_success_rate': 0.95,
        'cash_deposit_frequency_month': 0,
        'cash_deposit_consistency_score': 0,
        'residential_status': 'Owned',
        'years_at_current_address': 5,
        'city_tier': 'Tier-1',
    })


@pytest.fixture
def sample_dataframe(sample_customer_row):
    """Small DataFrame for batch testing."""
    rows = []
    np.random.seed(42)
    for i in range(100):
        row = sample_customer_row.copy()
        row['LoanID'] = f'TEST{i:04d}'
        row['Income'] = np.random.uniform(20000, 150000)
        row['CreditScore'] = np.random.randint(400, 850)
        row['DTIRatio'] = np.random.uniform(10, 70)
        row['savings_rate'] = np.random.uniform(5, 50)
        row['ontime_payment_rate_12m'] = np.random.uniform(0.5, 1.0)
        row['Default'] = 1 if np.random.random() < 0.15 else 0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def enriched_data_path():
    """Path to the enriched dataset."""
    path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'enriched_loan_data.parquet')
    if os.path.exists(path):
        return path
    pytest.skip("enriched_loan_data.parquet not found")


@pytest.fixture
def scored_data_path():
    """Path to the scored dataset."""
    path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scored_loan_data.parquet')
    if os.path.exists(path):
        return path
    pytest.skip("scored_loan_data.parquet not found")


@pytest.fixture
def ensemble_model_path():
    """Path to the ensemble model."""
    path = os.path.join(PROJECT_ROOT, 'data', 'models', 'ensemble_model.pkl')
    if os.path.exists(path):
        return path
    pytest.skip("ensemble_model.pkl not found")
