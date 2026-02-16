# src/data_generation/synthetic_data_generator.py
"""
Comprehensive synthetic data generator for the Pre-Delinquency Engine.
Generates realistic, correlated features covering:
  - Demographics & employment (segmented profiles)
  - Loan & credit bureau attributes
  - Detailed expense categories (housing, food, healthcare, etc.)
  - Digital & behavioral signals
  - Network / employer contagion features
  - Asset & liability snapshots
  - Segment-specific fields (gig, student, retired, business)
  - Causal default probability via logistic model
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

# ────────────────────────────────────────────────────────────────
# SEGMENT DEFINITIONS — realistic Indian banking profiles
# ────────────────────────────────────────────────────────────────

SEGMENTS = {
    'SALARIED_PRIVATE': {
        'weight': 0.30,
        'income_mu': 65000, 'income_sigma': 25000,
        'age_lo': 23, 'age_hi': 58,
        'credit_mu': 730, 'credit_sigma': 60,
        'months_employed_mu': 48, 'months_employed_sigma': 30,
    },
    'SALARIED_GOVT': {
        'weight': 0.12,
        'income_mu': 55000, 'income_sigma': 15000,
        'age_lo': 25, 'age_hi': 60,
        'credit_mu': 750, 'credit_sigma': 50,
        'months_employed_mu': 96, 'months_employed_sigma': 60,
    },
    'SELF_EMPLOYED': {
        'weight': 0.18,
        'income_mu': 55000, 'income_sigma': 35000,
        'age_lo': 26, 'age_hi': 62,
        'credit_mu': 700, 'credit_sigma': 75,
        'months_employed_mu': 60, 'months_employed_sigma': 40,
    },
    'BUSINESS_OWNER': {
        'weight': 0.15,
        'income_mu': 100000, 'income_sigma': 60000,
        'age_lo': 28, 'age_hi': 65,
        'credit_mu': 710, 'credit_sigma': 70,
        'months_employed_mu': 72, 'months_employed_sigma': 48,
    },
    'GIG_WORKER': {
        'weight': 0.12,
        'income_mu': 30000, 'income_sigma': 15000,
        'age_lo': 20, 'age_hi': 45,
        'credit_mu': 660, 'credit_sigma': 80,
        'months_employed_mu': 18, 'months_employed_sigma': 12,
    },
    'RETIRED': {
        'weight': 0.06,
        'income_mu': 35000, 'income_sigma': 12000,
        'age_lo': 55, 'age_hi': 80,
        'credit_mu': 740, 'credit_sigma': 55,
        'months_employed_mu': 360, 'months_employed_sigma': 60,
    },
    'STUDENT': {
        'weight': 0.07,
        'income_mu': 12000, 'income_sigma': 5000,
        'age_lo': 18, 'age_hi': 28,
        'credit_mu': 650, 'credit_sigma': 90,
        'months_employed_mu': 6, 'months_employed_sigma': 6,
    },
}


class PreDelinquencyDataGenerator:
    """Generate synthetic pre-delinquency data with 120+ realistic features."""

    def __init__(self, n_customers=50000):
        self.n_customers = n_customers

    # ────────────────────────────────────────────────────────────
    # CORE PROFILE
    # ────────────────────────────────────────────────────────────
    def _make_profile(self, cid):
        """Demographic, employment, and base financial info."""
        seg_names = list(SEGMENTS.keys())
        seg_weights = [SEGMENTS[s]['weight'] for s in seg_names]
        seg = np.random.choice(seg_names, p=seg_weights)
        s = SEGMENTS[seg]

        age = np.random.randint(s['age_lo'], s['age_hi'] + 1)
        gender = np.random.choice(['M', 'F'], p=[0.58, 0.42])
        marital = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'],
                                   p=[0.35, 0.50, 0.10, 0.05])
        education = np.random.choice(
            ["High School", "Bachelor's", "Master's", "PhD"],
            p=[0.25, 0.40, 0.28, 0.07])
        num_dependents = np.random.choice([0, 1, 2, 3, 4, 5],
                                          p=[0.25, 0.25, 0.22, 0.15, 0.08, 0.05])
        has_dependents = 1 if num_dependents > 0 else 0

        income = max(8000, np.random.normal(s['income_mu'], s['income_sigma']))
        months_employed = max(0, int(np.random.normal(
            s['months_employed_mu'], s['months_employed_sigma'])))

        credit_score = int(np.clip(np.random.normal(
            s['credit_mu'], s['credit_sigma']), 300, 900))

        city_tier = np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25])
        residential_status = np.random.choice(
            ['OWN', 'RENT', 'FAMILY', 'PG'],
            p=[0.30, 0.35, 0.25, 0.10])
        years_at_address = np.random.randint(0, 20)
        mobile_years = round(np.random.uniform(0.5, 15), 1)

        return {
            'customer_id': f'CUST_{cid:06d}',
            'Age': age,
            'gender': gender,
            'MaritalStatus': marital,
            'Education': education,
            'EmploymentType': seg,
            'MonthsEmployed': months_employed,
            'Income': round(income, 2),
            'CreditScore': credit_score,
            'num_dependents': num_dependents,
            'HasDependents': has_dependents,
            'city_tier': city_tier,
            'residential_status': residential_status,
            'years_at_current_address': years_at_address,
            'mobile_stability_years': mobile_years,
            'segment_category': seg,
            'detailed_segment': f"{seg}_{city_tier}",
        }

    # ────────────────────────────────────────────────────────────
    # LOAN & CREDIT
    # ────────────────────────────────────────────────────────────
    def _make_loan(self, p):
        income = p['Income']
        purpose = np.random.choice(
            ['Home', 'Auto', 'Education', 'Personal', 'Business', 'Medical',
             'DebtConsolidation', 'Other'],
            p=[0.20, 0.15, 0.10, 0.20, 0.12, 0.08, 0.10, 0.05])

        # Loan amount correlates with income and purpose
        mult = {'Home': 60, 'Auto': 8, 'Education': 12, 'Personal': 4,
                'Business': 20, 'Medical': 3, 'DebtConsolidation': 5, 'Other': 3}
        base = income * mult.get(purpose, 5)
        loan_amount = max(50000, np.random.normal(base, base * 0.4))

        tenure = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240],
                                  p=[0.05, 0.10, 0.15, 0.15, 0.20, 0.15, 0.10, 0.07, 0.03])
        rate = np.clip(np.random.normal(11.5, 2.5), 7.0, 24.0)

        # EMI calculation
        r = rate / 12 / 100
        n = tenure
        emi = loan_amount * r * (1 + r)**n / ((1 + r)**n - 1) if r > 0 else loan_amount / n
        dti = (emi / income) * 100

        num_credit_lines = np.random.poisson(3) + 1
        has_cosigner = int(np.random.random() < 0.15)
        has_mortgage = int(np.random.random() < (0.35 if purpose == 'Home' else 0.15))
        credit_inquiries_6m = np.random.poisson(1.5)
        oldest_credit_years = round(np.random.uniform(0.5, 25), 1)

        credit_util = np.clip(np.random.beta(2.5, 4) * 100, 0, 100)

        return {
            'LoanID': f'LN_{np.random.randint(100000, 999999)}',
            'LoanAmount': round(loan_amount, 2),
            'LoanTerm': tenure,
            'InterestRate': round(rate, 2),
            'LoanPurpose': purpose,
            'emi': round(emi, 2),
            'DTIRatio': round(dti, 2),
            'NumCreditLines': num_credit_lines,
            'HasCoSigner': has_cosigner,
            'HasMortgage': has_mortgage,
            'credit_inquiries_6m': credit_inquiries_6m,
            'oldest_credit_line_years': oldest_credit_years,
            'credit_utilization_ratio': round(credit_util, 2),
            'credit_lines_per_year': round(num_credit_lines / max(oldest_credit_years, 0.5), 2),
            'loan_restructuring_history': int(np.random.random() < 0.06),
            'settlement_history': int(np.random.random() < 0.03),
            'moratorium_months_remaining': np.random.choice([0, 0, 0, 0, 1, 2, 3]),
        }

    # ────────────────────────────────────────────────────────────
    # EXPENSE BREAKDOWN (12 categories)
    # ────────────────────────────────────────────────────────────
    def _make_expenses(self, p):
        income = p['Income']
        # Base fractions of income for each category
        housing = income * np.clip(np.random.normal(0.25, 0.08), 0.05, 0.50)
        food = income * np.clip(np.random.normal(0.15, 0.05), 0.05, 0.30)
        transport = income * np.clip(np.random.normal(0.08, 0.03), 0.01, 0.20)
        healthcare_base = income * np.clip(np.random.normal(0.04, 0.02), 0.01, 0.15)
        education_exp = income * np.clip(np.random.normal(0.05, 0.03), 0, 0.15)
        insurance = income * np.clip(np.random.normal(0.05, 0.02), 0, 0.12)
        lifestyle = income * np.clip(np.random.normal(0.08, 0.04), 0, 0.25)
        personal_care = income * np.clip(np.random.normal(0.03, 0.015), 0, 0.08)
        family = income * np.clip(np.random.normal(0.04, 0.02), 0, 0.12)
        communication = income * np.clip(np.random.normal(0.02, 0.008), 0.005, 0.05)
        discretionary = income * np.clip(np.random.normal(0.06, 0.03), 0, 0.20)

        total_expense = (housing + food + transport + healthcare_base +
                         education_exp + insurance + lifestyle + personal_care +
                         family + communication + discretionary)

        # Healthcare spike (5% get emergency)
        hc_spike = 1.0
        if np.random.random() < 0.05:
            hc_spike = np.random.choice([2.0, 3.0, 5.0, 8.0])

        pharmacy = max(200, np.random.normal(1500, 800))
        hospital_visits = np.random.poisson(0.4)

        return {
            'expense_housing': round(housing, 2),
            'expense_food_groceries': round(food, 2),
            'expense_transportation': round(transport, 2),
            'expense_healthcare': round(healthcare_base * hc_spike, 2),
            'expense_education': round(education_exp, 2),
            'expense_insurance_investments': round(insurance, 2),
            'expense_lifestyle_entertainment': round(lifestyle, 2),
            'expense_personal_care': round(personal_care, 2),
            'expense_family_social': round(family, 2),
            'expense_communication': round(communication, 2),
            'expense_discretionary_vices': round(discretionary, 2),
            'total_monthly_expense': round(total_expense, 2),
            'expense_to_income_ratio': round(total_expense / income, 3),
            'healthcare_current_spend': round(healthcare_base * hc_spike, 2),
            'healthcare_spending_multiplier': round(hc_spike, 2),
            'pharmacy_expense_monthly': round(pharmacy, 2),
            'hospital_visits_6m': hospital_visits,
            'health_insurance_coverage': int(np.random.random() < 0.55),
            # Lifestyle detail
            'dining_out_monthly': round(max(0, np.random.normal(2000, 1200)), 2),
            'food_delivery_monthly': round(max(0, np.random.normal(1500, 900)), 2),
            'shopping_monthly': round(max(0, np.random.normal(3000, 2000)), 2),
            'fuel_expense_monthly': round(max(0, np.random.normal(2500, 1500)), 2),
            'cab_rides_monthly': np.random.poisson(4),
            'cab_expense_monthly': round(max(0, np.random.normal(1200, 800)), 2),
            'ott_subscriptions_count': np.random.choice([0, 1, 2, 3, 4, 5],
                                                        p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]),
            'ott_monthly_spend': round(max(0, np.random.normal(500, 300)), 2),
            'gym_membership': int(np.random.random() < 0.20),
            'grocery_online_pct': round(np.clip(np.random.beta(2, 3) * 100, 0, 100), 1),
        }

    # ────────────────────────────────────────────────────────────
    # PAYMENT HISTORY & ACCOUNT HEALTH
    # ────────────────────────────────────────────────────────────
    def _make_payment_history(self, p):
        cs = p['CreditScore']
        # Higher credit score → better payment history
        ontime_base = 0.50 + 0.50 * (cs - 300) / 600
        ontime = np.clip(np.random.normal(ontime_base, 0.08), 0, 1)

        max_dpd = 0
        if np.random.random() > ontime:
            max_dpd = np.random.choice([0, 15, 30, 60, 90, 120],
                                       p=[0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
        num_dpd_30 = np.random.poisson(max(0, (1 - ontime) * 3))

        bounce_count = np.random.poisson(max(0, (1 - ontime) * 2))
        partial_count = np.random.poisson(max(0, (1 - ontime) * 1.5))
        avg_delay = max(0, np.random.normal((1 - ontime) * 8, 2))

        # Payment channel
        channel = np.random.choice(
            ['AUTO_DEBIT', 'UPI', 'NET_BANKING', 'CHEQUE', 'CASH', 'WALLET'],
            p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05])

        auto_success = np.clip(np.random.normal(0.92, 0.06), 0.5, 1.0) if channel == 'AUTO_DEBIT' else 0.0
        manual_pct = 0.0 if channel == 'AUTO_DEBIT' else np.clip(np.random.normal(0.6, 0.2), 0, 1)

        consistency = np.clip(np.random.normal(ontime * 0.8, 0.1), 0, 1)

        # Salary credit day (salaried only)
        sal_day = 0
        sal_delay = 0
        if p['EmploymentType'].startswith('SALARIED'):
            sal_day = np.random.choice([1, 5, 7, 10, 15, 25, 28])
            if np.random.random() < 0.12:
                sal_delay = np.random.randint(2, 20)

        return {
            'ontime_payment_rate_12m': round(ontime, 3),
            'max_dpd_last_12m': max_dpd,
            'num_dpd_30_plus': num_dpd_30,
            'cheque_bounce_count_12m': bounce_count,
            'partial_payment_count_12m': partial_count,
            'avg_delay_days': round(avg_delay, 1),
            'payment_channel': channel,
            'auto_debit_success_rate': round(auto_success, 3),
            'manual_payment_pct': round(manual_pct, 3),
            'payment_day_consistency': round(consistency, 3),
            'salary_credit_day': sal_day,
            'salary_delay_days': sal_delay,
            'salary_increment_pct_yoy': round(np.random.normal(8, 5), 1) if p['EmploymentType'].startswith('SALARIED') else 0,
            'variable_pay_pct': round(np.clip(np.random.normal(15, 10), 0, 60), 1) if p['EmploymentType'] == 'SALARIED_PRIVATE' else 0,
            'bonus_received_last_year': int(np.random.random() < 0.40) if p['EmploymentType'].startswith('SALARIED') else 0,
        }

    # ────────────────────────────────────────────────────────────
    # DIGITAL & BEHAVIORAL SIGNALS
    # ────────────────────────────────────────────────────────────
    def _make_behavioral(self, p):
        cs = p['CreditScore']
        stress_factor = max(0, (700 - cs) / 400)  # 0..1 stress

        app_login_freq = max(0, np.random.poisson(8 + stress_factor * 12))
        app_login_dur = round(max(3, np.random.lognormal(3.5 - stress_factor, 0.8)), 2)
        notif_dismiss = round(max(0.5, np.random.lognormal(1.5 - stress_factor * 0.5, 0.6)), 2)
        statement_open = round(np.clip(np.random.beta(2 + (1 - stress_factor) * 3, 2), 0, 1), 3)
        password_resets = np.random.poisson(0.3 + stress_factor * 1.5)
        emi_calc_usage = np.random.poisson(0.2 + stress_factor * 2)
        loan_inquiry_visits = np.random.poisson(0.1 + stress_factor * 1.5)
        cust_service_calls = np.random.poisson(0.3 + stress_factor * 2)

        # Subscription cascade (1=stable → 4=cancelling everything)
        sub_cascade = np.random.choice([1, 2, 3, 4],
                                       p=[0.65 - stress_factor * 0.2,
                                          0.20 + stress_factor * 0.05,
                                          0.10 + stress_factor * 0.10,
                                          0.05 + stress_factor * 0.05])

        # Wallet & digital payments
        wallet_velocity = round(np.clip(np.random.normal(5 + stress_factor * 8, 3), 0, 30), 2)
        wallet_to_bank = np.random.poisson(2 + stress_factor * 3)
        instant_cashouts = np.random.poisson(0.4 + stress_factor * 2)
        p2p_borrow = np.random.poisson(0.15 + stress_factor * 1.5)
        cash_to_digital = round(np.clip(np.random.normal(0.3, 0.15), 0.05, 0.95), 2)
        peak_offpeak = round(np.clip(np.random.normal(1.2 + stress_factor * 0.5, 0.3), 0.5, 3), 2)

        # Spending change
        disc_change = round(np.random.normal(-stress_factor * 15, 15), 2)

        # Digital engagement composite
        engagement = round(np.clip(
            0.3 * statement_open + 0.3 * (app_login_freq / 20) + 0.2 * (1 - stress_factor) + 0.2 * (notif_dismiss / 10),
            0, 1) * 100, 1)

        # Account balance trajectory
        bal_traj = round(np.random.normal(3 - stress_factor * 20, 10), 2)

        return {
            'app_login_frequency_month': app_login_freq,
            'app_login_duration_avg_sec': app_login_dur,
            'notification_dismiss_speed_sec': notif_dismiss,
            'statement_open_rate': statement_open,
            'password_resets_30d': password_resets,
            'emi_calculator_usage_30d': emi_calc_usage,
            'loan_inquiry_page_visits': loan_inquiry_visits,
            'customer_service_calls_30d': cust_service_calls,
            'subscription_cascade_phase': int(sub_cascade),
            'wallet_velocity': wallet_velocity,
            'wallet_to_bank_frequency': wallet_to_bank,
            'instant_cashouts_month': instant_cashouts,
            'p2p_borrow_requests_30d': p2p_borrow,
            'cash_to_digital_ratio': cash_to_digital,
            'peak_to_offpeak_ratio': peak_offpeak,
            'discretionary_spending_change_pct': disc_change,
            'digital_engagement_score': engagement,
            'balance_trajectory_pct': bal_traj,
            'merchant_downgrade_score': int(np.random.normal(-stress_factor * 3, 2)),
        }

    # ────────────────────────────────────────────────────────────
    # ASSETS, LIABILITIES & SAVINGS
    # ────────────────────────────────────────────────────────────
    def _make_assets_liabilities(self, p, loan):
        income = p['Income']
        age = p['Age']

        # Savings & account balances
        savings_rate = round(np.clip(np.random.normal(12, 8), -5, 40), 2)
        avg_balance_6m = max(500, np.random.normal(income * 2, income))
        min_bal_violations = np.random.poisson(0.8)

        # Fixed deposits, investments
        fd_rd = max(0, np.random.normal(income * 6, income * 4))
        fd_interest = round(fd_rd * 0.06 / 12, 2)  # ~6% p.a.
        mf_value = max(0, np.random.normal(income * 8, income * 6))
        stock_value = max(0, np.random.normal(income * 4, income * 5))
        gold_value = max(0, np.random.normal(income * 3, income * 3))
        epf = max(0, np.random.normal(income * age * 0.15, income * 5))

        # Properties
        prop_count = np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.10])
        prop_value = prop_count * max(500000, np.random.normal(3000000, 2000000))
        prop_loan = round(prop_value * np.random.uniform(0, 0.6), 2) if prop_count > 0 else 0

        # Vehicles
        two_wheelers = np.random.choice([0, 1, 2], p=[0.40, 0.45, 0.15])
        four_wheelers = np.random.choice([0, 1, 2], p=[0.60, 0.35, 0.05])
        vehicle_age = round(np.random.uniform(0, 12), 1)
        vehicle_loan = max(0, np.random.normal(150000, 100000)) if four_wheelers > 0 and np.random.random() < 0.4 else 0

        total_financial = fd_rd + mf_value + stock_value + epf
        total_assets = total_financial + prop_value + gold_value
        total_debt = loan['LoanAmount'] + prop_loan + vehicle_loan

        # Rental & other income
        rental = round(max(0, np.random.normal(8000, 5000)), 2) if prop_count > 0 and np.random.random() < 0.3 else 0
        pension = round(max(0, np.random.normal(25000, 8000)), 2) if p['EmploymentType'] == 'RETIRED' else 0

        return {
            'savings_rate': savings_rate,
            'savings_rate_pct': savings_rate,
            'avg_monthly_balance_6m': round(avg_balance_6m, 2),
            'min_balance_violations_6m': min_bal_violations,
            'fd_rd_total': round(fd_rd, 2),
            'fd_rd_interest_monthly': fd_interest,
            'mutual_fund_value': round(mf_value, 2),
            'stock_portfolio_value': round(stock_value, 2),
            'gold_value': round(gold_value, 2),
            'epf_ppf_balance': round(epf, 2),
            'property_owned_count': prop_count,
            'property_value_total': round(prop_value, 2),
            'property_loan_outstanding': round(prop_loan, 2),
            'num_two_wheelers': two_wheelers,
            'num_four_wheelers': four_wheelers,
            'vehicle_age_years': vehicle_age,
            'vehicle_loan_outstanding': round(vehicle_loan, 2),
            'total_financial_assets': round(total_financial, 2),
            'total_assets': round(total_assets, 2),
            'total_outstanding_debt': round(total_debt, 2),
            'total_debt_to_assets': round(total_debt / max(total_assets, 1), 4),
            'asset_to_debt_ratio': round(max(total_assets, 1) / max(total_debt, 1), 4),
            'net_worth': round(total_assets - total_debt, 2),
            'rental_income_monthly': rental,
            'monthly_pension': pension,
            'pf_contribution_regular': int(np.random.random() < 0.55) if p['EmploymentType'].startswith('SALARIED') else 0,
            'num_bank_accounts': np.random.choice([1, 2, 3, 4], p=[0.30, 0.40, 0.20, 0.10]),
        }

    # ────────────────────────────────────────────────────────────
    # NETWORK & EMPLOYER RISK
    # ────────────────────────────────────────────────────────────
    def _make_network(self, p):
        seg = p['EmploymentType']
        employer_health = round(np.clip(np.random.normal(75, 15), 20, 100), 1)
        employer_stock = round(np.random.normal(2, 8), 2)
        peer_default = round(np.clip(np.random.beta(1.5, 15), 0, 0.5), 4)
        industry_stress = round(np.clip(np.random.normal(30, 15), 0, 100), 1)
        cust_concentration = round(np.clip(np.random.beta(2, 8), 0, 1), 3) if seg == 'BUSINESS_OWNER' else 0

        return {
            'employer_health_score': employer_health,
            'employer_stock_change_pct': employer_stock,
            'peer_default_rate': peer_default,
            'industry_stress_index': industry_stress,
            'customer_concentration_risk': cust_concentration,
        }

    # ────────────────────────────────────────────────────────────
    # SEGMENT-SPECIFIC FEATURES
    # ────────────────────────────────────────────────────────────
    def _make_segment_specific(self, p):
        seg = p['EmploymentType']
        income = p['Income']
        feats = {}

        # Cash deposits (business / self-employed)
        feats['cash_deposit_frequency_month'] = np.random.poisson(3) if seg in ('BUSINESS_OWNER', 'SELF_EMPLOYED') else 0
        feats['cash_deposit_avg_amount'] = round(max(0, np.random.normal(25000, 15000)), 2) if feats['cash_deposit_frequency_month'] > 0 else 0
        feats['cash_deposit_before_emi_days'] = np.random.randint(1, 10) if feats['cash_deposit_frequency_month'] > 0 else 0
        feats['cash_deposit_consistency_score'] = round(np.clip(np.random.normal(70, 20), 0, 100), 1) if seg in ('BUSINESS_OWNER', 'SELF_EMPLOYED') else 0
        feats['cash_sufficiency_ratio'] = round(np.clip(np.random.normal(1.2, 0.4), 0.3, 3.0), 2) if seg in ('BUSINESS_OWNER', 'SELF_EMPLOYED') else 0

        # Business specific
        feats['monthly_revenue'] = round(max(0, np.random.normal(income * 3, income)), 2) if seg == 'BUSINESS_OWNER' else 0
        feats['profit_margin_pct'] = round(np.clip(np.random.normal(18, 8), -5, 50), 1) if seg == 'BUSINESS_OWNER' else 0
        feats['num_employees'] = np.random.poisson(8) if seg == 'BUSINESS_OWNER' else 0
        feats['gst_filing_regular'] = int(np.random.random() < 0.85) if seg == 'BUSINESS_OWNER' else 0
        feats['supplier_payment_delay_days'] = np.random.poisson(5) if seg == 'BUSINESS_OWNER' else 0
        feats['business_vintage_years'] = round(np.random.uniform(1, 25), 1) if seg == 'BUSINESS_OWNER' else 0
        feats['revenue_variance_cv'] = round(np.clip(np.random.normal(0.25, 0.15), 0.02, 1.0), 3) if seg == 'BUSINESS_OWNER' else 0
        feats['working_capital_days'] = np.random.randint(15, 120) if seg == 'BUSINESS_OWNER' else 0

        # Gig specific
        feats['multi_platform_count'] = np.random.choice([1, 2, 3, 4]) if seg == 'GIG_WORKER' else 0
        feats['platform_rating'] = round(np.clip(np.random.normal(4.2, 0.5), 2.0, 5.0), 1) if seg == 'GIG_WORKER' else 0
        feats['platform_active_days_month'] = np.random.randint(10, 30) if seg == 'GIG_WORKER' else 0
        feats['platform_daily_earning_avg'] = round(max(200, np.random.normal(900, 350)), 2) if seg == 'GIG_WORKER' else 0
        feats['zero_income_days_month'] = np.random.poisson(3) if seg == 'GIG_WORKER' else 0
        feats['income_cv_daily'] = round(np.clip(np.random.normal(0.45, 0.15), 0.1, 1.0), 3) if seg == 'GIG_WORKER' else 0

        # Student
        feats['institution_ranking'] = np.random.randint(1, 500) if seg == 'STUDENT' else 0
        feats['gpa'] = round(np.clip(np.random.normal(7.5, 1.5), 4.0, 10.0), 2) if seg == 'STUDENT' else 0
        feats['campus_placement_rate'] = round(np.clip(np.random.normal(75, 15), 30, 100), 1) if seg == 'STUDENT' else 0
        feats['expected_starting_salary'] = round(max(15000, np.random.normal(35000, 15000)), 2) if seg == 'STUDENT' else 0
        feats['scholarship_amount_monthly'] = round(max(0, np.random.normal(3000, 2000)), 2) if seg == 'STUDENT' else 0
        feats['parent_income_monthly'] = round(max(15000, np.random.normal(50000, 20000)), 2) if seg == 'STUDENT' else 0
        feats['parent_income_stable'] = int(np.random.random() < 0.8) if seg == 'STUDENT' else 0

        # Family transfer (all)
        feats['family_transfer_monthly'] = round(max(0, np.random.normal(2000, 3000)), 2)

        return feats

    # ────────────────────────────────────────────────────────────
    # DEFAULT TARGET (causal logistic model)
    # ────────────────────────────────────────────────────────────
    def _compute_target(self, row):
        """Realistic logistic default model with ~7-12% default rate."""
        logit = -3.5  # base intercept → ~3% base default

        # ── Financial stress signals ──────────────────────
        logit += 0.025 * (row['DTIRatio'] - 35)
        logit -= 0.006 * (row['CreditScore'] - 680)
        logit += 0.015 * max(0, row['credit_utilization_ratio'] - 50)
        logit -= 0.02 * row['savings_rate']

        debt_assets = row.get('total_debt_to_assets', 0.5)
        logit += 0.8 * max(0, debt_assets - 0.6)

        # ── Payment history ──────────────────────────────
        logit -= 2.0 * (row['ontime_payment_rate_12m'] - 0.85)
        logit += 0.03 * row['max_dpd_last_12m']
        logit += 0.15 * row['cheque_bounce_count_12m']
        logit += 0.12 * row['partial_payment_count_12m']

        # ── Behavioral signals ───────────────────────────
        if row['statement_open_rate'] < 0.2:
            logit += 0.5
        logit += 0.25 * row['password_resets_30d']
        logit += 0.35 * (row['subscription_cascade_phase'] - 1)
        logit += 0.15 * row['instant_cashouts_month']
        logit += 0.20 * row['p2p_borrow_requests_30d']

        if row['discretionary_spending_change_pct'] < -25:
            logit += 0.7
        if row['balance_trajectory_pct'] < -15:
            logit += 0.9

        # ── Employment & network ─────────────────────────
        if row['salary_delay_days'] > 5:
            logit += 0.6
        logit += 1.0 * row['peer_default_rate']
        if row['employer_health_score'] < 50:
            logit += 0.4

        # ── Healthcare shock ─────────────────────────────
        if row['healthcare_spending_multiplier'] > 2:
            logit += 0.8

        # ── Segment-specific ─────────────────────────────
        seg = row['EmploymentType']
        if seg == 'GIG_WORKER':
            logit += 0.3 * max(0, row.get('income_cv_daily', 0.3) - 0.4)
            logit += 0.05 * row.get('zero_income_days_month', 0)
        elif seg == 'BUSINESS_OWNER':
            if row.get('revenue_variance_cv', 0.2) > 0.4:
                logit += 0.3
        elif seg == 'STUDENT':
            if row.get('parent_income_stable', 1) == 0:
                logit += 0.5

        # Age & stability
        if row['Age'] < 25:
            logit += 0.2
        if row['MonthsEmployed'] < 12:
            logit += 0.3

        # Random noise
        logit += np.random.normal(0, 0.4)

        prob = 1 / (1 + np.exp(-logit))
        default = 1 if np.random.random() < prob else 0
        days_to_default = np.random.randint(5, 60) if default else None

        # ── Financial stress composite (0-100) ───────────
        stress = np.clip(
            15 * max(0, row['DTIRatio'] - 30) / 30 +
            20 * (1 - row['ontime_payment_rate_12m']) +
            15 * row['credit_utilization_ratio'] / 100 +
            10 * min(row['cheque_bounce_count_12m'], 5) / 5 +
            10 * (row['subscription_cascade_phase'] - 1) / 3 +
            10 * min(row['p2p_borrow_requests_30d'], 4) / 4 +
            10 * max(0, -row['balance_trajectory_pct']) / 30 +
            10 * max(0, row.get('healthcare_spending_multiplier', 1) - 1) / 4,
            0, 100)

        # ── Behavioral risk composite (0-100) ────────────
        behav = np.clip(
            20 * (1 - row['statement_open_rate']) +
            15 * min(row['password_resets_30d'], 5) / 5 +
            15 * (row['subscription_cascade_phase'] - 1) / 3 +
            15 * min(row['instant_cashouts_month'], 5) / 5 +
            10 * min(row['p2p_borrow_requests_30d'], 4) / 4 +
            10 * max(0, -row['discretionary_spending_change_pct']) / 40 +
            15 * min(row.get('customer_service_calls_30d', 0), 5) / 5,
            0, 100)

        return {
            'Default': default,
            'target': default,
            'delinquency_prob': round(prob, 4),
            'days_to_default': days_to_default,
            'financial_stress_score': round(stress, 1),
            'financial_stress_index': round(stress, 1),
            'behavioral_risk_score': round(behav, 1),
            'income_stability_index': round(np.clip(100 - stress * 0.6 + np.random.normal(0, 5), 0, 100), 1),
        }

    # ────────────────────────────────────────────────────────────
    # MAIN GENERATOR
    # ────────────────────────────────────────────────────────────
    def generate_dataset(self):
        print(f"Generating {self.n_customers:,} synthetic customers with 120+ features...")

        rows = []
        for i in range(self.n_customers):
            if i % 10000 == 0 and i > 0:
                print(f"  {i:,} / {self.n_customers:,}")

            profile = self._make_profile(i)
            loan = self._make_loan(profile)
            expenses = self._make_expenses(profile)
            payments = self._make_payment_history(profile)
            behavioral = self._make_behavioral(profile)
            assets = self._make_assets_liabilities(profile, loan)
            network = self._make_network(profile)
            segment = self._make_segment_specific(profile)

            # Combine all
            row = {**profile, **loan, **expenses, **payments,
                   **behavioral, **assets, **network, **segment}

            # Compute target & composite scores
            target = self._compute_target(row)
            row.update(target)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Derived convenience columns
        df['emi_to_income_ratio'] = (df['emi'] / df['Income'].clip(lower=1) * 100).round(2)
        df['emi_to_savings_ratio'] = (df['emi'] / df['avg_monthly_balance_6m'].clip(lower=1) * 100).round(2)
        df['liquidity_ratio'] = (df['avg_monthly_balance_6m'] / df['emi'].clip(lower=1)).round(3)
        df['financial_cushion_months'] = (df['total_financial_assets'] / df['total_monthly_expense'].clip(lower=1)).round(1)
        df['debt_service_coverage'] = (df['Income'] / df['emi'].clip(lower=1)).round(3)
        df['months_of_runway'] = (df['avg_monthly_balance_6m'] / df['total_monthly_expense'].clip(lower=1)).round(1)
        df['bounce_per_credit_line'] = (df['cheque_bounce_count_12m'] / df['NumCreditLines'].clip(lower=1)).round(3)
        df['loan_burden_score'] = ((df['DTIRatio'] * 0.4 + df['credit_utilization_ratio'] * 0.3 +
                                    (1 - df['ontime_payment_rate_12m']) * 100 * 0.3)).round(1)

        print(f"\n✓ Dataset: {len(df):,} rows × {len(df.columns)} columns")
        print(f"✓ Default rate: {df['Default'].mean()*100:.1f}%")

        return df


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)

    generator = PreDelinquencyDataGenerator(n_customers=50000)
    df = generator.generate_dataset()

    # Save
    out = 'data/raw/synthetic_predelinquency_data.parquet'
    df.to_parquet(out, index=False)
    print(f"\n✓ Saved: {out}")

    sample = 'data/raw/sample_data.csv'
    df.head(1000).to_csv(sample, index=False)
    print(f"✓ Sample: {sample}")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Customers:    {len(df):,}")
    print(f"Features:     {len(df.columns)}")
    print(f"Default rate: {df['Default'].mean()*100:.1f}%")
    print(f"\nSegment breakdown:")
    for seg, cnt in df['EmploymentType'].value_counts().items():
        rate = df[df['EmploymentType'] == seg]['Default'].mean() * 100
        print(f"  {seg:25s} {cnt:6,} ({rate:.1f}% default)")
    print(f"\nColumns:\n  {', '.join(sorted(df.columns.tolist()))}")