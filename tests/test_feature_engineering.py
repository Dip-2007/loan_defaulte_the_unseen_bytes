# tests/test_feature_engineering.py
"""
Tests for Section 4: Feature Engineering (10 core + 6 advanced formulas).
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering.feature_engineer import FeatureEngineer


@pytest.fixture
def small_df():
    """Create a small DataFrame with required columns for feature engineering."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        'LoanID': [f'FE{i:04d}' for i in range(n)],
        'Income': np.random.uniform(20000, 150000, n),
        'LoanAmount': np.random.uniform(100000, 1000000, n),
        'LoanTerm': np.random.choice([12, 24, 36, 48, 60], n),
        'CreditScore': np.random.randint(400, 850, n),
        'DTIRatio': np.random.uniform(10, 70, n),
        'InterestRate': np.random.uniform(8, 18, n),
        'Default': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'segment_category': np.random.choice(
            ['EMPLOYED', 'BUSINESS_OWNER', 'GIG_WORKER', 'RETIRED', 'STUDENT'], n
        ),
        'detailed_segment': np.random.choice(
            ['Private Sector', 'Small Business', 'Gig Worker', 'Pension (Govt)', 'Undergraduate'], n
        ),
        'avg_monthly_balance_6m': np.random.uniform(5000, 100000, n),
        'total_monthly_expense': np.random.uniform(15000, 80000, n),
        'expense_housing': np.random.uniform(5000, 25000, n),
        'expense_food_groceries': np.random.uniform(3000, 15000, n),
        'expense_transportation': np.random.uniform(1000, 8000, n),
        'expense_healthcare': np.random.uniform(500, 10000, n),
        'expense_lifestyle_entertainment': np.random.uniform(1000, 15000, n),
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
        'pharmacy_expense_monthly': np.random.uniform(500, 5000, n),
        'hospital_visits_6m': np.random.randint(0, 5, n),
        'peer_default_rate': np.random.uniform(0, 0.15, n),
        'platform_daily_earning_avg': np.random.uniform(300, 1500, n),
        'income_cv_daily': np.random.uniform(0.1, 1.2, n),
        'industry_stress_index': np.random.uniform(10, 50, n),
    })
    return df


class TestCoreFormulas:
    """Test the 10 core mathematical formulas from Section 4.2."""

    def test_income_stability_index(self, small_df):
        """Formula 1: ISI = (σ_income / μ_income) × 100"""
        fe = FeatureEngineer(small_df)
        fe.calc_income_stability_index()
        assert 'isi' in fe.df.columns
        assert 'isi_band' in fe.df.columns
        assert (fe.df['isi'] >= 0).all()

    def test_debt_to_income(self, small_df):
        """Formula 2: DTI = (Total Monthly EMIs / Net Monthly Income) × 100"""
        fe = FeatureEngineer(small_df)
        fe.calc_debt_to_income()
        assert 'computed_dti' in fe.df.columns
        assert 'dti_band' in fe.df.columns
        valid = fe.df['computed_dti'].dropna()
        assert (valid >= 0).all()

    def test_savings_rate(self, small_df):
        """Formula 3: SR = (Average Monthly Balance / Net Income) × 100"""
        fe = FeatureEngineer(small_df)
        fe.calc_savings_rate()
        assert 'savings_rate_pct' in fe.df.columns
        assert 'savings_band' in fe.df.columns

    def test_emi_cushion_index(self, small_df):
        """Formula 4: ECI = (Disposable Income - EMI) / EMI"""
        fe = FeatureEngineer(small_df)
        fe.calc_emi_cushion_index()
        assert 'eci' in fe.df.columns
        assert 'eci_band' in fe.df.columns

    def test_expenditure_volatility(self, small_df):
        """Formula 5: EVS = std dev of monthly expenditure"""
        fe = FeatureEngineer(small_df)
        fe.calc_expenditure_volatility()
        assert 'evs' in fe.df.columns
        assert 'evs_ratio' in fe.df.columns
        valid = fe.df['evs'].dropna()
        assert (valid >= 0).all()

    def test_salary_delay_trend(self, small_df):
        """Formula 6: SDT = Avg salary delay"""
        fe = FeatureEngineer(small_df)
        fe.calc_salary_delay_trend()
        assert 'sdt' in fe.df.columns
        assert 'sdt_band' in fe.df.columns

    def test_cash_deposit_pattern_score(self, small_df):
        """Formula 7: CDPS for business owners"""
        fe = FeatureEngineer(small_df)
        fe.calc_cash_deposit_pattern_score()
        assert 'cdps' in fe.df.columns
        assert 'cash_sufficiency_ratio' in fe.df.columns
        assert 'cdps_band' in fe.df.columns

    def test_discretionary_spending_ratio(self, small_df):
        """Formula 8: DSR = (Discretionary / Total) × 100"""
        fe = FeatureEngineer(small_df)
        fe.calc_discretionary_spending_ratio()
        assert 'dsr' in fe.df.columns
        assert 'dsr_band' in fe.df.columns
        valid = fe.df['dsr'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_payment_timing_consistency(self, small_df):
        """Formula 9: PTC = 1 - (σ_payment_day / Expected_payment_day)"""
        fe = FeatureEngineer(small_df)
        fe.calc_payment_timing_consistency()
        assert 'ptc' in fe.df.columns
        assert 'ptc_band' in fe.df.columns

    def test_account_balance_trajectory(self, small_df):
        """Formula 10: ABT = (End - Start) / Start"""
        fe = FeatureEngineer(small_df)
        fe.calc_account_balance_trajectory()
        assert 'abt' in fe.df.columns
        assert 'abt_band' in fe.df.columns


class TestAdvancedBehavioral:
    """Test the 6 advanced behavioral features from Section 4.3."""

    def test_subscription_cascade(self, small_df):
        fe = FeatureEngineer(small_df)
        fe.calc_subscription_cascade_index()
        assert 'subscription_cascade_score' in fe.df.columns
        valid = fe.df['subscription_cascade_score'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_digital_wallet_velocity(self, small_df):
        fe = FeatureEngineer(small_df)
        fe.calc_digital_wallet_velocity()
        assert 'wallet_velocity' in fe.df.columns
        assert 'wallet_velocity_band' in fe.df.columns
        valid = fe.df['wallet_velocity'].dropna()
        assert (valid >= 0).all()

    def test_merchant_downgrade(self, small_df):
        fe = FeatureEngineer(small_df)
        fe.calc_merchant_downgrade_score()
        assert 'merchant_downgrade_score' in fe.df.columns
        assert 'merchant_downgrade_band' in fe.df.columns

    def test_healthcare_spike(self, small_df):
        fe = FeatureEngineer(small_df)
        fe.calc_healthcare_spike_indicator()
        assert 'healthcare_spike_ratio' in fe.df.columns
        assert 'healthcare_spike_band' in fe.df.columns

    def test_employer_contagion(self, small_df):
        fe = FeatureEngineer(small_df)
        fe.calc_employer_contagion_risk()
        assert 'employer_risk_multiplier' in fe.df.columns
        valid = fe.df['employer_risk_multiplier'].dropna()
        assert (valid >= 1.0).all()
        assert (valid <= 2.5).all()

    def test_gig_income_variance(self, small_df):
        fe = FeatureEngineer(small_df)
        fe.calc_gig_income_variance()
        assert 'gig_cv_band' in fe.df.columns


class TestFullPipeline:
    """Test the complete feature engineering pipeline."""

    def test_run_produces_features(self, small_df):
        fe = FeatureEngineer(small_df)
        result = fe.run()
        # Should have more columns than input
        assert len(result.columns) > len(small_df.columns)
        # Should have same number of rows
        assert len(result) == len(small_df)

    def test_no_all_nan_columns(self, small_df):
        fe = FeatureEngineer(small_df)
        result = fe.run()
        # No column should be entirely NaN
        all_nan_cols = [c for c in result.columns if result[c].isna().all()]
        assert len(all_nan_cols) == 0, f"All-NaN columns: {all_nan_cols}"
