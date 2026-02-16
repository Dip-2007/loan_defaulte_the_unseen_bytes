# notebooks/demo.py
"""
Pre-Delinquency Engine — Demo Script
=====================================
Run this to see the full pipeline in action:
  1. Feature Engineering on sample data
  2. Risk Scoring (single customer + batch)
  3. Rajesh Kumar worked example validation
  4. API endpoint demo

Usage:
  python notebooks/demo.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 70)
print("  PRE-DELINQUENCY ENGINE — DEMO")
print("=" * 70)


# ============================================================
# 1. FEATURE ENGINEERING DEMO
# ============================================================
print("\n" + "─" * 70)
print("  STEP 1: Feature Engineering (Section 4)")
print("─" * 70)

from feature_engineering.feature_engineer import FeatureEngineer

# Create sample data
np.random.seed(42)
n = 100
sample_data = pd.DataFrame({
    'LoanID': [f'DEMO{i:04d}' for i in range(n)],
    'Income': np.random.uniform(20000, 200000, n),
    'LoanAmount': np.random.uniform(100000, 2000000, n),
    'LoanTerm': np.random.choice([12, 24, 36, 48, 60], n),
    'CreditScore': np.random.randint(400, 850, n),
    'DTIRatio': np.random.uniform(10, 70, n),
    'InterestRate': np.random.uniform(8, 20, n),
    'Default': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    'Age': np.random.randint(22, 65, n),
    'segment_category': np.random.choice(
        ['EMPLOYED', 'BUSINESS_OWNER', 'SELF_EMPLOYED', 'RETIRED', 'STUDENT'], n
    ),
    'avg_monthly_balance_6m': np.random.uniform(5000, 150000, n),
    'total_monthly_expense': np.random.uniform(15000, 100000, n),
    'expense_housing': np.random.uniform(5000, 30000, n),
    'expense_food_groceries': np.random.uniform(3000, 15000, n),
    'expense_transportation': np.random.uniform(1000, 10000, n),
    'expense_healthcare': np.random.uniform(500, 10000, n),
    'expense_lifestyle_entertainment': np.random.uniform(1000, 20000, n),
    'expense_discretionary_vices': np.random.uniform(0, 5000, n),
    'salary_delay_days': np.random.uniform(0, 10, n),
    'savings_rate': np.random.uniform(5, 50, n),
    'ontime_payment_rate_12m': np.random.uniform(0.5, 1.0, n),
    'payment_day_consistency': np.random.uniform(0.3, 1.0, n),
    'max_dpd_last_12m': np.random.randint(0, 30, n),
    'cash_deposit_frequency_month': np.random.uniform(0, 10, n),
    'cash_deposit_consistency_score': np.random.uniform(0, 1, n),
    'cash_deposit_avg_amount': np.random.uniform(0, 50000, n),
    'subscription_cascade_phase': np.random.choice([0, 1, 2, 3, 4], n),
    'p2p_borrow_requests_30d': np.random.randint(0, 5, n),
    'instant_cashouts_month': np.random.randint(0, 15, n),
    'peer_default_rate': np.random.uniform(0, 0.15, n),
    'income_cv_daily': np.random.uniform(0.1, 1.2, n),
    'industry_stress_index': np.random.uniform(10, 50, n),
})

fe = FeatureEngineer(sample_data)
df_featured = fe.run()
print(f"\n✓ Output: {len(df_featured)} rows × {len(df_featured.columns)} columns")


# ============================================================
# 2. RISK SCORING DEMO
# ============================================================
print("\n" + "─" * 70)
print("  STEP 2: Risk Scoring (Section 5)")
print("─" * 70)

from risk_scoring.risk_scorer import RiskScorer, validate_worked_example

scorer = RiskScorer()

# Score a single customer
customer = pd.Series({
    'isi': 15.0,
    'computed_dti': 35.0,
    'DTIRatio': 35.0,
    'savings_rate_pct': 25.0,
    'ontime_payment_rate_12m': 0.92,
    'max_dpd_last_12m': 5,
    'eci': 1.5,
    'evs_ratio': 0.25,
    'abt': 0.05,
    'CreditScore': 720,
    'employer_risk_multiplier': 1.1,
    'healthcare_spike_ratio': 1.0,
    'peer_default_rate': 0.03,
    'subscription_cascade_score': 0.25,
    'wallet_velocity': 0.2,
    'Age': 35,
    'oldest_credit_line_years': 8,
})

score, components = scorer.score_customer(customer)
band, prob, action = scorer.classify_risk(score)

print(f"\n  Customer Risk Score: {score:.1f} / 100")
print(f"  Risk Band:          {band}")
print(f"  Default Probability: {prob}")
print(f"  Action:             {action}")
print(f"\n  Top 5 Risk Components:")
for name, val in sorted(components.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    • {name:25s} {val:6.1f}")


# ============================================================
# 3. WORKED EXAMPLE VALIDATION
# ============================================================
print("\n" + "─" * 70)
print("  STEP 3: Rajesh Kumar Worked Example (Section 5.5)")
print("─" * 70)

wex_score, wex_band = validate_worked_example()
print(f"\n  Rajesh Kumar Score: {wex_score:.1f}")
print(f"  Classification:    {wex_band}")
print(f"  Expected:          ~38.5, LOW RISK")
print(f"  ✓ Validation: {'PASS' if wex_band == 'LOW RISK' else 'FAIL'}")


# ============================================================
# 4. BATCH SCORING
# ============================================================
print("\n" + "─" * 70)
print("  STEP 4: Batch Scoring")
print("─" * 70)

scored_df = scorer.score_dataframe(df_featured)
risk_dist = scored_df['risk_band'].value_counts()
print(f"\n  Risk Distribution ({len(scored_df)} customers):")
for band_name, count in risk_dist.items():
    pct = count / len(scored_df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {band_name:12s} {count:4d} ({pct:5.1f}%) {bar}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  DEMO COMPLETE")
print("=" * 70)
print(f"""
  Features Engineered:  {len(df_featured.columns)}
  Customers Scored:     {len(scored_df)}
  Risk Bands Used:      {len(risk_dist)}
  Worked Example:       {'PASS ✓' if wex_band == 'LOW RISK' else 'FAIL ✗'}

  Next Steps:
    • API:       python -m uvicorn src.api.app:app --reload
    • Dashboard: python src/dashboard/app.py
    • Tests:     pytest tests/ -v
""")
