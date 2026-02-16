# src/data_generation/enrich_dataset.py
"""
Enriches the real Kaggle Loan Default dataset with 200+ segment-specific features.
Maps original columns, assigns detailed customer segments, and generates
synthetic behavioral/financial/lifestyle features anchored to real profiles.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_generation.config import (
    SEGMENTS, EMPLOYMENT_TYPE_MAP, EMPLOYMENT_TYPE_WEIGHTS,
    INCOME_PARAMS, EXPENDITURE_RATIOS, LOAN_PURPOSES
)

np.random.seed(42)


class DatasetEnricher:
    """Enrich real loan default data with comprehensive features."""

    def __init__(self, df):
        self.df = df.copy()
        self.n = len(df)

    # ==============================================================
    # SECTION 1: MAP & ENRICH DEMOGRAPHICS
    # ==============================================================
    def enrich_demographics(self):
        """Add universal demographic features (Section 2.1)"""
        df = self.df

        # Already have: Age, MaritalStatus, HasDependents, Education
        # Add missing demographics
        df['num_dependents'] = np.where(
            df['HasDependents'] == 'Yes',
            np.random.choice([1, 2, 3, 4], self.n, p=[0.35, 0.35, 0.20, 0.10]),
            0
        )
        df['gender'] = np.random.choice(['M', 'F'], self.n, p=[0.58, 0.42])
        df['residential_status'] = np.random.choice(
            ['OWNED', 'RENTED', 'FAMILY', 'COMPANY_PROVIDED'],
            self.n, p=[0.35, 0.40, 0.20, 0.05]
        )
        df['years_at_current_address'] = np.clip(
            np.random.exponential(scale=5, size=self.n), 0.5, 30
        ).round(1)
        df['mobile_stability_years'] = np.clip(
            np.random.exponential(scale=4, size=self.n), 0.5, 20
        ).round(1)
        df['city_tier'] = np.random.choice(
            ['METRO', 'TIER_1', 'TIER_2', 'TIER_3', 'RURAL'],
            self.n, p=[0.25, 0.20, 0.25, 0.20, 0.10]
        )

        return df

    # ==============================================================
    # SECTION 2: ASSIGN DETAILED SEGMENTS
    # ==============================================================
    def assign_segments(self):
        """Map EmploymentType to detailed 13 sub-categories"""
        df = self.df

        detailed_segments = []
        for _, row in df.iterrows():
            emp = row['EmploymentType']
            age = row['Age']

            # Override for retired/students
            if emp == 'Unemployed':
                if age >= 55:
                    seg = np.random.choice(['RETIRED_GOVT', 'RETIRED_PRIVATE'], p=[0.6, 0.4])
                elif age <= 25:
                    seg = np.random.choice(['STUDENT_UG', 'STUDENT_PG'], p=[0.6, 0.4])
                else:
                    options = EMPLOYMENT_TYPE_MAP[emp]
                    weights = EMPLOYMENT_TYPE_WEIGHTS[emp]
                    seg = np.random.choice(options, p=weights)
            else:
                options = EMPLOYMENT_TYPE_MAP.get(emp, ['SALARIED_PRIVATE'])
                weights = EMPLOYMENT_TYPE_WEIGHTS.get(emp, [1.0])
                seg = np.random.choice(options, p=weights)

            detailed_segments.append(seg)

        df['detailed_segment'] = detailed_segments
        df['segment_category'] = df['detailed_segment'].map(
            lambda s: SEGMENTS[s]['category']
        )
        return df

    # ==============================================================
    # SECTION 3: CREDIT HISTORY (expanded)
    # ==============================================================
    def enrich_credit_history(self):
        """Expand credit history features (Section 2.1)"""
        df = self.df

        # Already have: CreditScore, NumCreditLines
        df['total_outstanding_debt'] = (
            df['LoanAmount'] * np.random.uniform(0.3, 1.0, self.n)
        ).round(0)
        df['credit_utilization_ratio'] = np.clip(
            np.random.beta(2, 5, self.n) * 100, 0, 100
        ).round(1)
        # Past DPD - driven by credit risk factors, NOT target
        # Lower credit score & higher DTI => more likely to have past DPD
        dpd_risk = (1 - df['CreditScore'] / 900) * 30 + (df['DTIRatio'] / 100) * 20
        df['max_dpd_last_12m'] = np.clip(
            dpd_risk + np.random.normal(0, 12, self.n), 0, 180
        ).astype(int)
        df['num_dpd_30_plus'] = np.where(
            df['max_dpd_last_12m'] >= 30,
            np.random.choice([1, 2, 3], self.n, p=[0.5, 0.3, 0.2]),
            0
        )
        df['loan_restructuring_history'] = np.random.choice(
            [0, 1], self.n, p=[0.92, 0.08]
        )
        df['settlement_history'] = np.random.choice(
            [0, 1], self.n, p=[0.95, 0.05]
        )
        df['credit_inquiries_6m'] = np.random.poisson(lam=1.5, size=self.n)
        df['oldest_credit_line_years'] = np.clip(
            np.random.exponential(scale=5, size=self.n), 0.5, 30
        ).round(1)

        return df

    # ==============================================================
    # SECTION 4: BANKING BEHAVIOR
    # ==============================================================
    def enrich_banking_behavior(self):
        """Add banking behavior features (Section 2.1)"""
        df = self.df

        df['num_bank_accounts'] = np.random.choice(
            [1, 2, 3, 4, 5], self.n, p=[0.20, 0.35, 0.25, 0.15, 0.05]
        )
        df['avg_monthly_balance_6m'] = np.clip(
            df['Income'] * np.random.uniform(0.5, 3.0, self.n), 5000, 500000
        ).round(0)
        df['min_balance_violations_6m'] = np.random.poisson(lam=0.8, size=self.n)
        df['cheque_bounce_count_12m'] = np.random.poisson(lam=0.3, size=self.n)
        df['auto_debit_success_rate'] = np.clip(
            np.random.beta(8, 2, self.n), 0.5, 1.0
        ).round(3)
        df['manual_payment_pct'] = (1 - df['auto_debit_success_rate']).round(3)

        return df

    # ==============================================================
    # SECTION 5: PAYMENT HISTORY (Current Loan)
    # ==============================================================
    def enrich_payment_history(self):
        """Add payment history features (Section 2.1)"""
        df = self.df

        # Payment history driven by credit risk factors, NOT target
        # Higher credit score => better on-time rate
        base_ontime = 0.5 + 0.4 * (df['CreditScore'] / 900) - 0.15 * (df['DTIRatio'] / 100)
        df['ontime_payment_rate_12m'] = np.clip(
            base_ontime + np.random.normal(0, 0.12, self.n), 0.3, 1
        ).round(3)
        df['avg_delay_days'] = np.where(
            df['ontime_payment_rate_12m'] < 0.9,
            np.clip(np.random.exponential(scale=5, size=self.n), 0, 60).round(0),
            0
        )
        # Partial payments driven by DTI ratio (higher burden => more partials)
        partial_lam = 0.3 + 2.5 * np.clip(df['DTIRatio'] / 100, 0, 1)
        df['partial_payment_count_12m'] = np.random.poisson(
            lam=partial_lam, size=self.n
        )
        df['payment_channel'] = np.random.choice(
            ['AUTO_DEBIT', 'NET_BANKING', 'UPI', 'CASH', 'CHEQUE'],
            self.n, p=[0.40, 0.25, 0.20, 0.10, 0.05]
        )
        df['payment_day_consistency'] = np.clip(
            np.random.beta(5, 2, self.n), 0, 1
        ).round(3)

        return df

    # ==============================================================
    # SECTION 6: SALARIED-SPECIFIC FEATURES
    # ==============================================================
    def enrich_salaried_features(self):
        """Salary, employer health, income stability (Section 2.2A)"""
        df = self.df
        is_sal = df['detailed_segment'].str.startswith('SALARIED')

        df['salary_credit_day'] = np.where(is_sal, np.random.choice(range(1, 8), self.n), 0)
        df['salary_delay_days'] = np.where(
            is_sal, np.clip(np.random.exponential(1, self.n), 0, 20).round(0).astype(int), 0
        )
        df['salary_increment_pct_yoy'] = np.where(
            is_sal, np.clip(np.random.normal(8, 5, self.n), -10, 30).round(1), 0
        )
        df['employer_health_score'] = np.where(
            is_sal, np.clip(np.random.normal(75, 15, self.n), 10, 100).round(1), np.nan
        )
        df['employer_stock_change_pct'] = np.where(
            is_sal & (df['detailed_segment'] == 'SALARIED_PRIVATE'),
            np.random.normal(5, 20, self.n).round(1), 0
        )
        df['peer_default_rate'] = np.where(
            is_sal, np.clip(np.random.beta(2, 20, self.n), 0, 0.3).round(3), np.nan
        )
        df['industry_stress_index'] = np.where(
            is_sal, np.clip(np.random.normal(30, 15, self.n), 0, 100).round(1), np.nan
        )
        df['pf_contribution_regular'] = np.where(
            is_sal, np.random.choice([0, 1], self.n, p=[0.1, 0.9]), np.nan
        )
        df['bonus_received_last_year'] = np.where(
            is_sal, np.random.choice([0, 1], self.n, p=[0.35, 0.65]), np.nan
        )
        df['variable_pay_pct'] = np.where(
            is_sal, np.clip(np.random.normal(15, 10, self.n), 0, 50).round(1), 0
        )

        return df

    # ==============================================================
    # SECTION 7: BUSINESS OWNER FEATURES
    # ==============================================================
    def enrich_business_features(self):
        """Business fundamentals, cash monitoring, GST (Section 2.2B)"""
        df = self.df
        is_biz = df['segment_category'] == 'BUSINESS_OWNER'

        df['business_vintage_years'] = np.where(
            is_biz, np.clip(np.random.exponential(5, self.n), 0.5, 30).round(1), np.nan
        )
        df['gst_filing_regular'] = np.where(
            is_biz, np.random.choice([0, 1], self.n, p=[0.2, 0.8]), np.nan
        )
        df['monthly_revenue'] = np.where(
            is_biz, np.clip(np.random.lognormal(11.5, 0.8, self.n), 50000, 5000000).round(0), np.nan
        )
        df['revenue_variance_cv'] = np.where(
            is_biz, np.clip(np.random.exponential(0.3, self.n), 0.05, 2.0).round(3), np.nan
        )
        df['profit_margin_pct'] = np.where(
            is_biz, np.clip(np.random.normal(15, 10, self.n), -20, 50).round(1), np.nan
        )
        df['working_capital_days'] = np.where(
            is_biz, np.clip(np.random.normal(45, 20, self.n), 5, 120).round(0), np.nan
        )

        # CRITICAL: Cash deposit monitoring
        is_cash_biz = df['detailed_segment'] == 'BUSINESS_CASH'
        df['cash_deposit_frequency_month'] = np.where(
            is_biz, np.random.poisson(lam=4, size=self.n), 0
        )
        df['cash_deposit_avg_amount'] = np.where(
            is_biz, np.clip(np.random.lognormal(9.5, 0.8, self.n), 5000, 500000).round(0), 0
        )
        df['cash_deposit_before_emi_days'] = np.where(
            is_biz, np.random.choice(range(0, 10), self.n), np.nan
        )
        df['cash_deposit_consistency_score'] = np.where(
            is_biz, np.clip(np.random.beta(3, 2, self.n) * 100, 0, 100).round(1), np.nan
        )
        df['cash_to_digital_ratio'] = np.where(
            is_cash_biz, np.clip(np.random.beta(5, 3, self.n), 0.2, 0.95).round(3),
            np.where(is_biz, np.clip(np.random.beta(2, 5, self.n), 0, 0.5).round(3), np.nan)
        )
        df['num_employees'] = np.where(
            is_biz, np.clip(np.random.lognormal(1.5, 1, self.n), 1, 200).round(0), np.nan
        )
        df['customer_concentration_risk'] = np.where(
            is_biz, np.clip(np.random.beta(2, 5, self.n) * 100, 5, 80).round(1), np.nan
        )
        df['supplier_payment_delay_days'] = np.where(
            is_biz, np.clip(np.random.exponential(10, self.n), 0, 60).round(0), np.nan
        )

        return df

    # ==============================================================
    # SECTION 8: GIG WORKER FEATURES
    # ==============================================================
    def enrich_gig_features(self):
        """Platform income, volatility, active days (Section 2.2C)"""
        df = self.df
        is_gig = df['detailed_segment'] == 'SELF_GIG'

        df['platform_daily_earning_avg'] = np.where(
            is_gig, np.clip(np.random.normal(800, 300, self.n), 200, 3000).round(0), np.nan
        )
        df['platform_active_days_month'] = np.where(
            is_gig, np.clip(np.random.normal(22, 5, self.n), 5, 30).round(0), np.nan
        )
        df['zero_income_days_month'] = np.where(
            is_gig, np.clip(np.random.poisson(3, self.n), 0, 20), np.nan
        )
        df['income_cv_daily'] = np.where(
            is_gig, np.clip(np.random.exponential(0.4, self.n), 0.1, 1.5).round(3), np.nan
        )
        df['platform_rating'] = np.where(
            is_gig, np.clip(np.random.normal(4.3, 0.5, self.n), 1, 5).round(2), np.nan
        )
        df['peak_to_offpeak_ratio'] = np.where(
            is_gig, np.clip(np.random.normal(1.5, 0.4, self.n), 1, 3).round(2), np.nan
        )
        df['multi_platform_count'] = np.where(
            is_gig, np.random.choice([1, 2, 3], self.n, p=[0.50, 0.35, 0.15]), np.nan
        )
        df['wallet_to_bank_frequency'] = np.where(
            is_gig, np.random.poisson(lam=8, size=self.n), np.nan
        )

        return df

    # ==============================================================
    # SECTION 9: RETIRED / PENSIONER FEATURES
    # ==============================================================
    def enrich_retired_features(self):
        """Pension, healthcare costs, family support (Section 2.2D)"""
        df = self.df
        is_ret = df['segment_category'] == 'RETIRED'

        df['monthly_pension'] = np.where(
            is_ret,
            np.where(df['detailed_segment'] == 'RETIRED_GOVT',
                     np.clip(np.random.normal(45000, 12000, self.n), 15000, 100000),
                     np.clip(np.random.normal(25000, 10000, self.n), 5000, 60000)).round(0),
            np.nan
        )
        df['pharmacy_expense_monthly'] = np.where(
            is_ret, np.clip(np.random.lognormal(7.5, 0.8, self.n), 500, 30000).round(0), np.nan
        )
        df['hospital_visits_6m'] = np.where(
            is_ret, np.random.poisson(lam=2, size=self.n), np.nan
        )
        df['health_insurance_coverage'] = np.where(
            is_ret, np.random.choice([0, 1], self.n, p=[0.3, 0.7]), np.nan
        )
        df['family_transfer_monthly'] = np.where(
            is_ret, np.clip(np.random.exponential(8000, self.n), 0, 50000).round(0), np.nan
        )
        df['fd_rd_interest_monthly'] = np.where(
            is_ret, np.clip(np.random.exponential(3000, self.n), 0, 30000).round(0), np.nan
        )
        df['rental_income_monthly'] = np.where(
            is_ret & (np.random.random(self.n) < 0.3),
            np.clip(np.random.normal(15000, 5000, self.n), 5000, 40000).round(0), np.nan
        )

        return df

    # ==============================================================
    # SECTION 10: STUDENT FEATURES
    # ==============================================================
    def enrich_student_features(self):
        """Academic status, family financial health (Section 2.2E)"""
        df = self.df
        is_stu = df['segment_category'] == 'STUDENT'

        df['gpa'] = np.where(
            is_stu, np.clip(np.random.normal(7.5, 1.5, self.n), 3, 10).round(2), np.nan
        )
        df['institution_ranking'] = np.where(
            is_stu, np.clip(np.random.lognormal(4, 1, self.n), 1, 500).round(0), np.nan
        )
        df['scholarship_amount_monthly'] = np.where(
            is_stu & (np.random.random(self.n) < 0.3),
            np.clip(np.random.normal(5000, 2000, self.n), 1000, 15000).round(0), np.nan
        )
        df['parent_income_monthly'] = np.where(
            is_stu, np.clip(np.random.normal(50000, 25000, self.n), 10000, 200000).round(0), np.nan
        )
        df['parent_income_stable'] = np.where(
            is_stu, np.random.choice([0, 1], self.n, p=[0.2, 0.8]), np.nan
        )
        df['moratorium_months_remaining'] = np.where(
            is_stu, np.clip(np.random.normal(12, 6, self.n), 0, 48).round(0), np.nan
        )
        df['expected_starting_salary'] = np.where(
            is_stu, np.clip(np.random.normal(40000, 15000, self.n), 15000, 100000).round(0), np.nan
        )
        df['campus_placement_rate'] = np.where(
            is_stu, np.clip(np.random.beta(5, 3, self.n) * 100, 20, 100).round(1), np.nan
        )

        return df

    # ==============================================================
    # SECTION 11: EXPENDITURE & LIFESTYLE (Section 3.1)
    # ==============================================================
    def enrich_expenditure(self):
        """Comprehensive expenditure by category"""
        df = self.df

        for category, params in EXPENDITURE_RATIOS.items():
            ratio = np.clip(
                np.random.normal(params['mean'], params['std'], self.n),
                0, params['mean'] * 3
            )
            df[f'expense_{category}'] = (df['Income'] * ratio).round(0)

        # Sub-categories of food
        df['dining_out_monthly'] = (df['expense_food_groceries'] * np.random.uniform(0.1, 0.4, self.n)).round(0)
        df['food_delivery_monthly'] = (df['expense_food_groceries'] * np.random.uniform(0.05, 0.25, self.n)).round(0)
        df['grocery_online_pct'] = np.clip(np.random.beta(2, 5, self.n), 0, 0.8).round(3)

        # Sub-categories of transport
        df['fuel_expense_monthly'] = (df['expense_transportation'] * np.random.uniform(0.3, 0.7, self.n)).round(0)
        df['cab_rides_monthly'] = np.random.poisson(lam=4, size=self.n)
        df['cab_expense_monthly'] = (df['cab_rides_monthly'] * np.random.uniform(100, 400, self.n)).round(0)

        # Sub-categories of lifestyle
        df['ott_subscriptions_count'] = np.random.choice([0, 1, 2, 3, 4, 5], self.n, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        df['ott_monthly_spend'] = (df['ott_subscriptions_count'] * np.random.uniform(100, 500, self.n)).round(0)
        df['gym_membership'] = np.random.choice([0, 1], self.n, p=[0.65, 0.35])
        df['shopping_monthly'] = (df['expense_lifestyle_entertainment'] * np.random.uniform(0.2, 0.5, self.n)).round(0)

        # Total expense & savings
        expense_cols = [c for c in df.columns if c.startswith('expense_')]
        df['total_monthly_expense'] = df[expense_cols].sum(axis=1).round(0)
        df['savings_rate'] = np.clip(
            (df['Income'] - df['total_monthly_expense']) / df['Income'], -0.5, 0.8
        ).round(3)

        return df

    # ==============================================================
    # SECTION 12: ASSETS & VEHICLES (Section 3.2)
    # ==============================================================
    def enrich_assets(self):
        """Vehicle, property, financial assets"""
        df = self.df

        # Vehicles
        df['num_two_wheelers'] = np.random.choice([0, 1, 2], self.n, p=[0.30, 0.50, 0.20])
        df['num_four_wheelers'] = np.random.choice([0, 1, 2], self.n, p=[0.45, 0.45, 0.10])
        df['vehicle_age_years'] = np.where(
            (df['num_two_wheelers'] + df['num_four_wheelers']) > 0,
            np.clip(np.random.exponential(4, self.n), 0.5, 15).round(1), 0
        )
        df['vehicle_loan_outstanding'] = np.where(
            df['num_four_wheelers'] > 0,
            np.where(np.random.random(self.n) < 0.4,
                     np.clip(np.random.normal(300000, 150000, self.n), 50000, 1000000).round(0), 0), 0
        )

        # Property
        df['property_owned_count'] = np.random.choice([0, 1, 2, 3], self.n, p=[0.40, 0.40, 0.15, 0.05])
        df['property_value_total'] = np.where(
            df['property_owned_count'] > 0,
            (df['property_owned_count'] * np.clip(
                np.random.lognormal(14.5, 0.8, self.n), 1000000, 50000000
            )).round(0), 0
        )
        df['property_loan_outstanding'] = np.where(
            df['HasMortgage'] == 'Yes',
            np.clip(df['property_value_total'] * np.random.uniform(0.2, 0.7, self.n), 0, 30000000).round(0), 0
        )

        # Financial assets
        df['fd_rd_total'] = np.clip(
            df['Income'] * np.random.exponential(6, self.n), 0, 5000000
        ).round(0)
        df['mutual_fund_value'] = np.where(
            np.random.random(self.n) < 0.3,
            np.clip(np.random.lognormal(11, 1, self.n), 10000, 5000000).round(0), 0
        )
        df['stock_portfolio_value'] = np.where(
            np.random.random(self.n) < 0.15,
            np.clip(np.random.lognormal(10.5, 1.2, self.n), 5000, 3000000).round(0), 0
        )
        df['gold_value'] = np.where(
            np.random.random(self.n) < 0.4,
            np.clip(np.random.lognormal(10, 0.8, self.n), 10000, 2000000).round(0), 0
        )
        df['epf_ppf_balance'] = np.clip(
            df['Income'] * df['MonthsEmployed'] * np.random.uniform(0.05, 0.15, self.n),
            0, 5000000
        ).round(0)

        # Total assets
        df['total_financial_assets'] = (
            df['fd_rd_total'] + df['mutual_fund_value'] +
            df['stock_portfolio_value'] + df['gold_value'] + df['epf_ppf_balance']
        )
        df['total_assets'] = df['total_financial_assets'] + df['property_value_total']
        df['asset_to_debt_ratio'] = np.where(
            df['total_outstanding_debt'] > 0,
            (df['total_assets'] / df['total_outstanding_debt']).round(2), 999
        )

        return df

    # ==============================================================
    # SECTION 13: DIGITAL BEHAVIOR & BEHAVIORAL SIGNALS
    # ==============================================================
    def enrich_digital_behavior(self):
        """App behavior, notification patterns, digital stress signals"""
        df = self.df

        df['app_login_frequency_month'] = np.random.poisson(lam=15, size=self.n)
        df['app_login_duration_avg_sec'] = np.clip(
            np.random.lognormal(3.5, 0.8, self.n), 2, 300
        ).round(1)
        df['notification_dismiss_speed_sec'] = np.clip(
            np.random.lognormal(1.5, 0.6, self.n), 0.5, 30
        ).round(2)
        df['statement_open_rate'] = np.clip(
            np.random.beta(3, 2, self.n), 0, 1
        ).round(3)
        df['password_resets_30d'] = np.random.poisson(lam=0.3, size=self.n)
        df['customer_service_calls_30d'] = np.random.poisson(lam=0.5, size=self.n)
        df['emi_calculator_usage_30d'] = np.random.poisson(lam=0.8, size=self.n)
        df['loan_inquiry_page_visits'] = np.random.poisson(lam=0.4, size=self.n)

        # Subscription cascade phase
        df['subscription_cascade_phase'] = np.random.choice(
            [1, 2, 3, 4], self.n, p=[0.70, 0.15, 0.10, 0.05]
        )

        # P2P borrow requests - driven by income stress, NOT target
        p2p_lam = 0.2 + 1.5 * np.clip(1 - df['Income'] / 80000, 0, 1)
        df['p2p_borrow_requests_30d'] = np.random.poisson(
            lam=p2p_lam, size=self.n
        )

        # Instant cashout behavior
        df['instant_cashouts_month'] = np.random.poisson(
            lam=np.where(df['CreditScore'] > 700, 0.5, 2.0), size=self.n
        )

        return df

    # ==============================================================
    # SECTION 14: DERIVED RATIOS & COMPUTED FEATURES
    # ==============================================================
    def compute_derived_features(self):
        """Compute cross-feature ratios and composite scores"""
        df = self.df

        # Financial health ratios
        df['emi_to_income_ratio'] = np.clip(
            df['LoanAmount'] / (df['LoanTerm'] * df['Income'] + 1), 0, 5
        ).round(4)
        df['expense_to_income_ratio'] = np.clip(
            df['total_monthly_expense'] / (df['Income'] + 1), 0, 3
        ).round(3)
        df['liquidity_ratio'] = np.clip(
            df['avg_monthly_balance_6m'] / (df['total_monthly_expense'] + 1), 0, 20
        ).round(3)
        df['debt_service_coverage'] = np.clip(
            df['Income'] / (df['LoanAmount'] / df['LoanTerm'] + 1), 0, 50
        ).round(3)
        df['net_worth'] = (df['total_assets'] - df['total_outstanding_debt']).round(0)
        df['financial_cushion_months'] = np.clip(
            df['total_financial_assets'] / (df['total_monthly_expense'] + 1), 0, 120
        ).round(1)

        # Behavioral risk composite
        df['behavioral_risk_score'] = (
            0.2 * (df['p2p_borrow_requests_30d'] > 0).astype(int) +
            0.15 * (df['instant_cashouts_month'] > 3).astype(int) +
            0.15 * (df['subscription_cascade_phase'] > 1).astype(int) +
            0.2 * (df['password_resets_30d'] > 1).astype(int) +
            0.15 * (df['statement_open_rate'] < 0.3).astype(int) +
            0.15 * (df['customer_service_calls_30d'] > 2).astype(int)
        ).round(3)

        # Income stability index (segment-dependent)
        df['income_stability_index'] = np.where(
            df['segment_category'] == 'EMPLOYED', 0.85,
            np.where(df['segment_category'] == 'BUSINESS_OWNER', 0.65,
            np.where(df['segment_category'] == 'SELF_EMPLOYED', 0.55,
            np.where(df['segment_category'] == 'RETIRED', 0.90, 0.30)))
        )
        # Add noise
        df['income_stability_index'] = np.clip(
            df['income_stability_index'] + np.random.normal(0, 0.1, self.n), 0, 1
        ).round(3)

        return df

    # ==============================================================
    # MAIN PIPELINE
    # ==============================================================
    def run(self):
        """Execute full enrichment pipeline"""
        print("=" * 60)
        print("COMPREHENSIVE DATA ENRICHMENT PIPELINE")
        print("=" * 60)
        print(f"Input: {self.n:,} rows, {len(self.df.columns)} columns")

        steps = [
            ("Demographics", self.enrich_demographics),
            ("Segments", self.assign_segments),
            ("Credit History", self.enrich_credit_history),
            ("Banking Behavior", self.enrich_banking_behavior),
            ("Payment History", self.enrich_payment_history),
            ("Salaried Features", self.enrich_salaried_features),
            ("Business Features", self.enrich_business_features),
            ("Gig Worker Features", self.enrich_gig_features),
            ("Retired Features", self.enrich_retired_features),
            ("Student Features", self.enrich_student_features),
            ("Expenditure & Lifestyle", self.enrich_expenditure),
            ("Assets & Vehicles", self.enrich_assets),
            ("Digital Behavior", self.enrich_digital_behavior),
            ("Derived Features", self.compute_derived_features),
        ]

        for name, fn in steps:
            self.df = fn()
            print(f"  ✓ {name:30s} → {len(self.df.columns)} columns")

        print(f"\n{'=' * 60}")
        print(f"OUTPUT: {self.n:,} rows, {len(self.df.columns)} columns")
        print(f"Default rate: {self.df['Default'].mean() * 100:.2f}%")
        print(f"Segments: {self.df['detailed_segment'].nunique()} unique")
        print(f"Segment distribution:")
        print(self.df['detailed_segment'].value_counts().to_string())
        print(f"{'=' * 60}")

        return self.df


# ==============================================================
# MAIN EXECUTION
# ==============================================================
if __name__ == "__main__":
    # Load real dataset
    input_path = 'data/raw/Loan_default.csv'
    print(f"Loading real dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Enrich
    enricher = DatasetEnricher(df)
    enriched_df = enricher.run()

    # Save
    os.makedirs('data/processed', exist_ok=True)

    parquet_path = 'data/processed/enriched_loan_data.parquet'
    enriched_df.to_parquet(parquet_path, index=False)
    print(f"\n✓ Saved to: {parquet_path}")

    csv_path = 'data/processed/enriched_sample.csv'
    enriched_df.head(1000).to_csv(csv_path, index=False)
    print(f"✓ Sample saved to: {csv_path}")

    print(f"\nAll columns ({len(enriched_df.columns)}):")
    for i, col in enumerate(enriched_df.columns, 1):
        print(f"  {i:3d}. {col}")
