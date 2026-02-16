# src/data_generation/config.py
"""
Configuration for all customer segments, feature distributions, and risk weights.
Maps the real Loan Default dataset columns to our comprehensive feature set.
"""

# ============================================================
# CUSTOMER SEGMENTS & SUB-CATEGORIES
# ============================================================
SEGMENTS = {
    'SALARIED_PRIVATE': {'category': 'EMPLOYED', 'sub': 'Private Sector', 'income_source': 'Monthly Salary'},
    'SALARIED_GOVT': {'category': 'EMPLOYED', 'sub': 'Government', 'income_source': 'Fixed Salary'},
    'SALARIED_CONTRACT': {'category': 'EMPLOYED', 'sub': 'Contract/Temporary', 'income_source': 'Variable Salary'},
    'BUSINESS_SMALL': {'category': 'BUSINESS_OWNER', 'sub': 'Small Business', 'income_source': 'Business Income'},
    'BUSINESS_MSME': {'category': 'BUSINESS_OWNER', 'sub': 'MSME', 'income_source': 'Business Profit'},
    'BUSINESS_CASH': {'category': 'BUSINESS_OWNER', 'sub': 'Cash-Only Business', 'income_source': 'Cash Revenue'},
    'SELF_FREELANCER': {'category': 'SELF_EMPLOYED', 'sub': 'Freelancer', 'income_source': 'Project-based'},
    'SELF_GIG': {'category': 'SELF_EMPLOYED', 'sub': 'Gig Worker', 'income_source': 'Daily/Weekly'},
    'SELF_PROFESSIONAL': {'category': 'SELF_EMPLOYED', 'sub': 'Professional Practice', 'income_source': 'Client Fees'},
    'RETIRED_GOVT': {'category': 'RETIRED', 'sub': 'Pension (Govt)', 'income_source': 'Monthly Pension'},
    'RETIRED_PRIVATE': {'category': 'RETIRED', 'sub': 'Pension (Private)', 'income_source': 'Monthly/Lump'},
    'STUDENT_UG': {'category': 'STUDENT', 'sub': 'Undergraduate', 'income_source': 'Family/Scholarship'},
    'STUDENT_PG': {'category': 'STUDENT', 'sub': 'Postgraduate', 'income_source': 'Assistantship/Loan'},
}

# Map original EmploymentType column to our detailed segments
EMPLOYMENT_TYPE_MAP = {
    'Full-time': ['SALARIED_PRIVATE', 'SALARIED_GOVT', 'SALARIED_CONTRACT'],
    'Part-time': ['SELF_FREELANCER', 'SELF_GIG', 'SELF_PROFESSIONAL'],
    'Self-employed': ['BUSINESS_SMALL', 'BUSINESS_MSME', 'BUSINESS_CASH'],
    'Unemployed': ['STUDENT_UG', 'STUDENT_PG', 'RETIRED_GOVT', 'RETIRED_PRIVATE'],
}

EMPLOYMENT_TYPE_WEIGHTS = {
    'Full-time': [0.55, 0.30, 0.15],
    'Part-time': [0.40, 0.35, 0.25],
    'Self-employed': [0.40, 0.35, 0.25],
    'Unemployed': [0.25, 0.25, 0.25, 0.25],
}

# ============================================================
# LOAN TYPES
# ============================================================
LOAN_PURPOSES = {
    'Home': {'typical_amount': (1000000, 10000000), 'typical_tenure': [120, 180, 240, 360]},
    'Auto': {'typical_amount': (300000, 2000000), 'typical_tenure': [36, 48, 60, 84]},
    'Education': {'typical_amount': (200000, 3000000), 'typical_tenure': [60, 84, 120]},
    'Business': {'typical_amount': (500000, 5000000), 'typical_tenure': [36, 60, 84, 120]},
    'Other': {'typical_amount': (50000, 1000000), 'typical_tenure': [12, 24, 36, 48, 60]},
}

# ============================================================
# INCOME DISTRIBUTION PARAMETERS BY SEGMENT (Monthly, INR)
# ============================================================
INCOME_PARAMS = {
    'SALARIED_PRIVATE': {'mean': 65000, 'std': 30000, 'min': 20000},
    'SALARIED_GOVT': {'mean': 55000, 'std': 15000, 'min': 25000},
    'SALARIED_CONTRACT': {'mean': 45000, 'std': 20000, 'min': 15000},
    'BUSINESS_SMALL': {'mean': 70000, 'std': 40000, 'min': 15000},
    'BUSINESS_MSME': {'mean': 90000, 'std': 50000, 'min': 20000},
    'BUSINESS_CASH': {'mean': 50000, 'std': 30000, 'min': 10000},
    'SELF_FREELANCER': {'mean': 55000, 'std': 35000, 'min': 10000},
    'SELF_GIG': {'mean': 30000, 'std': 15000, 'min': 8000},
    'SELF_PROFESSIONAL': {'mean': 80000, 'std': 40000, 'min': 20000},
    'RETIRED_GOVT': {'mean': 40000, 'std': 12000, 'min': 15000},
    'RETIRED_PRIVATE': {'mean': 30000, 'std': 15000, 'min': 8000},
    'STUDENT_UG': {'mean': 12000, 'std': 5000, 'min': 0},
    'STUDENT_PG': {'mean': 18000, 'std': 8000, 'min': 0},
}

# ============================================================
# EXPENDITURE CATEGORY PERCENTAGES (of income)
# ============================================================
EXPENDITURE_RATIOS = {
    'housing': {'mean': 0.30, 'std': 0.08},
    'food_groceries': {'mean': 0.15, 'std': 0.05},
    'transportation': {'mean': 0.08, 'std': 0.03},
    'healthcare': {'mean': 0.05, 'std': 0.04},
    'education': {'mean': 0.06, 'std': 0.05},
    'lifestyle_entertainment': {'mean': 0.08, 'std': 0.05},
    'insurance_investments': {'mean': 0.10, 'std': 0.06},
    'communication': {'mean': 0.02, 'std': 0.01},
    'personal_care': {'mean': 0.03, 'std': 0.02},
    'family_social': {'mean': 0.05, 'std': 0.04},
    'discretionary_vices': {'mean': 0.03, 'std': 0.04},
}

# ============================================================
# RISK SCORE WEIGHTS (for logistic model)
# ============================================================
RISK_WEIGHTS = {
    'dti_ratio': 0.15,
    'credit_score': -0.08,
    'income_stability': -0.10,
    'payment_history': -0.12,
    'expense_to_income': 0.10,
    'digital_engagement': -0.05,
    'employer_health': -0.06,
    'cash_flow_health': -0.08,
    'asset_coverage': -0.07,
    'behavioral_stress': 0.12,
}
