# src/workflow/workflow_engine.py
"""
Section 6: End-to-End Workflow Implementation (7 Steps)
Section 7: Special Scenario Handling  (6 Scenarios)
"""

import pandas as pd
import numpy as np
import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# INTERVENTION TEMPLATES
# ============================================================
INTERVENTION_TEMPLATES = {
    'SAFE': None,
    'LOW RISK': {
        'channel': ['SMS'],
        'timing': 'Within 48 hours',
        'template': (
            "Hi {name}, a quick wellness tip: maintaining a savings buffer "
            "of 3x your EMI (₹{emi_3x:,.0f}) helps avoid payment stress. "
            "Your next EMI of ₹{emi:,.0f} is due on {due_date}. Stay on track! 🌟"
        )
    },
    'MODERATE': {
        'channel': ['SMS', 'Email'],
        'timing': 'Within 24 hours',
        'template': (
            "Hi {name}, we noticed some changes in your financial pattern. "
            "Your EMI of ₹{emi:,.0f} is due in {days_to_emi} days. "
            "We can help with:\n"
            "1️⃣ EMI date shift\n"
            "2️⃣ Flexible payment plan\n"
            "Reply YES to explore options."
        )
    },
    'HIGH RISK': {
        'channel': ['WhatsApp', 'SMS'],
        'timing': 'Same day (by 10 AM)',
        'template': (
            "Hi {name}, your financial wellness is important to us. "
            "We see your EMI of ₹{emi:,.0f} is coming up in {days_to_emi} days. "
            "We'd like to offer:\n"
            "1️⃣ EMI restructuring at lower rate\n"
            "2️⃣ 1-month payment holiday\n"
            "3️⃣ Emergency bridge loan at 0% for 30 days\n"
            "Call us at 1800-XXX-XXXX or reply HELP."
        )
    },
    'CRITICAL': {
        'channel': ['Phone Call', 'WhatsApp'],
        'timing': 'Immediate (within 1 hour)',
        'template': (
            "URGENT: Hi {name}, we understand you may be facing financial difficulty. "
            "Your EMI of ₹{emi:,.0f} is due in {days_to_emi} days. "
            "We're here to help, not to pressure:\n"
            "1️⃣ 3-month EMI pause\n"
            "2️⃣ Loan restructuring\n"
            "3️⃣ Emergency ₹{emergency_loan:,.0f} loan at 0% for 60 days\n"
            "4️⃣ Insurance claim assistance\n"
            "A support specialist will call you within the hour."
        )
    }
}


# ============================================================
# SCENARIO HANDLERS (Section 7)
# ============================================================

class ScenarioHandler:
    """Handle 6 special scenarios from Section 7."""

    @staticmethod
    def scenario_cash_businessman(row, scorer_result):
        """Scenario 1: Cash-only businessman with low balance before EMI."""
        if row.get('segment_category') != 'BUSINESS_OWNER':
            return None

        cdps = row.get('cdps', 0) or 0
        csr = row.get('cash_sufficiency_ratio', 0) or 0
        cash_before_emi = row.get('cash_deposit_before_emi_days', 99)
        balance = row.get('avg_monthly_balance_6m', 0)
        emi = row.get('LoanAmount', 0) / max(row.get('LoanTerm', 1), 1)

        if balance < emi and cash_before_emi <= 3:
            if cdps > 70 and csr >= 1.1:
                return {
                    'scenario': 'CASH_BUSINESSMAN_RELIABLE',
                    'risk_override': 'MODERATE',
                    'message': (
                        f"Reminder: EMI of ₹{emi:,.0f} due in {int(cash_before_emi)} days. "
                        "Your cash deposit pattern is consistent. "
                        "Please ensure deposit before EMI date."
                    ),
                    'reason': f'CDPS={cdps:.0f}%, CSR={csr:.2f} — reliable cash pattern'
                }
            else:
                return {
                    'scenario': 'CASH_BUSINESSMAN_STRESSED',
                    'risk_override': 'CRITICAL',
                    'message': (
                        f"Your EMI of ₹{emi:,.0f} is due in {int(cash_before_emi)} days. "
                        f"Current balance: ₹{balance:,.0f}. "
                        "Need help? We can offer flexible options. Call us."
                    ),
                    'reason': f'CDPS={cdps:.0f}%, CSR={csr:.2f} — irregular cash flow'
                }
        return None

    @staticmethod
    def scenario_student_family(row, scorer_result):
        """Scenario 2: Student loan with family support dependency."""
        segment = row.get('detailed_segment', '')
        if 'STUDENT' not in str(segment):
            return None

        parent_stable = row.get('parent_income_stable', 1)
        parent_income = row.get('parent_income_monthly', 0)
        scholarship = row.get('scholarship_amount_monthly', 0)
        emi = row.get('LoanAmount', 0) / max(row.get('LoanTerm', 1), 1)

        if parent_stable == 0:
            return {
                'scenario': 'STUDENT_FAMILY_STRESS',
                'risk_override': 'HIGH RISK',
                'message': (
                    "We notice your family may be facing some financial changes. "
                    "Your education loan EMI of ₹{emi:,.0f} is protected. "
                    "Options: moratorium extension, income-based repayment. "
                    "Focus on your studies — we'll work with your family."
                ),
                'reason': 'Parent income instability detected'
            }
        elif parent_income > 0 and parent_income > emi * 3:
            return {
                'scenario': 'STUDENT_FAMILY_STRONG',
                'risk_override': 'LOW RISK',
                'message': None,
                'reason': 'Strong family financial support'
            }
        return None

    @staticmethod
    def scenario_multiple_loans(row, scorer_result):
        """Scenario 3: Customer with multiple EMIs on different dates."""
        num_loans = row.get('NumCreditLines', 1)
        if num_loans < 3:
            return None

        dti = row.get('computed_dti', row.get('DTIRatio', 30))
        if dti > 45:
            return {
                'scenario': 'MULTI_LOAN_STRESS',
                'risk_override': None,  # Don't override, just flag
                'message': (
                    f"Managing {int(num_loans)} active credit lines with DTI of {dti:.1f}% "
                    "requires careful planning. Would you like us to help consolidate? "
                    "We can restructure into a single lower EMI."
                ),
                'reason': f'{int(num_loans)} loans, DTI={dti:.1f}% — consolidation recommended'
            }
        return None

    @staticmethod
    def scenario_retired_healthcare(row, scorer_result):
        """Scenario 4: Retiree with healthcare emergency."""
        segment = row.get('segment_category', '')
        if segment != 'RETIRED':
            return None

        spike = row.get('healthcare_spike_ratio', 1.0) or 1.0
        pension_stable = row.get('monthly_pension', 0) > 0

        if spike > 3.0:
            emi = row.get('LoanAmount', 0) / max(row.get('LoanTerm', 1), 1)
            return {
                'scenario': 'RETIRED_MEDICAL_EMERGENCY',
                'risk_override': 'MODERATE' if pension_stable else 'CRITICAL',
                'message': (
                    "We understand medical emergencies are stressful. "
                    "We can offer:\n"
                    f"1. ₹{emi*3:,.0f} medical emergency loan at 0% for 60 days\n"
                    "2. 3-month EMI payment holiday\n"
                    "3. Insurance claim assistance\n"
                    "Your pension is safe. Let us help."
                ),
                'reason': f'Healthcare spike {spike:.1f}x, pension={"stable" if pension_stable else "none"}'
            }
        return None

    @staticmethod
    def scenario_gig_suspension(row, scorer_result):
        """Scenario 5: Gig worker platform suspension (zero income)."""
        segment = row.get('segment_category', '')
        if segment != 'SELF_EMPLOYED':
            return None

        zero_days = row.get('zero_income_days_month', 0) or 0
        multi_platform = row.get('multi_platform_count', 1) or 1

        if zero_days > 10:
            emi = row.get('LoanAmount', 0) / max(row.get('LoanTerm', 1), 1)
            if multi_platform > 1:
                return {
                    'scenario': 'GIG_PARTIAL_SUSPENSION',
                    'risk_override': 'MODERATE',
                    'message': (
                        "We noticed your primary platform earnings stopped. "
                        "Since you earn on other platforms, we can offer "
                        "a flexible EMI plan until your account is restored."
                    ),
                    'reason': f'{int(zero_days)} zero-income days but {int(multi_platform)} platforms active'
                }
            else:
                return {
                    'scenario': 'GIG_FULL_SUSPENSION',
                    'risk_override': 'CRITICAL',
                    'message': (
                        "We noticed your platform earnings stopped. "
                        "Is everything okay? We can help:\n"
                        "1. 30-day payment holiday\n"
                        f"2. ₹{emi:,.0f} emergency loan to cover EMI\n"
                        "3. Connect to alternate gig platforms\n"
                        "Reply YES to speak with support."
                    ),
                    'reason': f'{int(zero_days)} zero-income days, single platform dependency'
                }
        return None

    @staticmethod
    def scenario_employer_layoff(row, scorer_result):
        """Scenario 6: Employer layoff news / peer defaults."""
        segment = row.get('segment_category', '')
        if segment != 'EMPLOYED':
            return None

        peer_rate = row.get('peer_default_rate', 0) or 0
        industry_stress = row.get('industry_stress_index', 30) or 30
        employer_health = row.get('employer_health_score', 70) or 70

        if peer_rate > 0.15 or (industry_stress > 70 and employer_health < 40):
            return {
                'scenario': 'EMPLOYER_LAYOFF_RISK',
                'risk_override': None,
                'message': (
                    "We noticed news about your industry/employer. "
                    "Your financial wellness is important to us. "
                    "If you're affected, we can:\n"
                    "1. Pause EMIs for 3 months\n"
                    "2. Provide job placement assistance\n"
                    "3. Offer emergency funds\n"
                    "This is proactive support, not a collection call."
                ),
                'reason': f'Peer default rate={peer_rate:.1%}, industry stress={industry_stress}'
            }
        return None


# ============================================================
# WORKFLOW ENGINE (Section 6)
# ============================================================

class WorkflowEngine:
    """End-to-end daily workflow per Section 6."""

    def __init__(self):
        self.scenario_handler = ScenarioHandler()
        self.scenario_methods = [
            self.scenario_handler.scenario_cash_businessman,
            self.scenario_handler.scenario_student_family,
            self.scenario_handler.scenario_multiple_loans,
            self.scenario_handler.scenario_retired_healthcare,
            self.scenario_handler.scenario_gig_suspension,
            self.scenario_handler.scenario_employer_layoff,
        ]

    def step1_ingest(self, data_path):
        """Step 1: Data Ingestion."""
        print(f"  STEP 1: Data Ingestion")
        df = pd.read_parquet(data_path)
        print(f"    Loaded {len(df):,} customers from {data_path}")
        return df

    def step2_preprocess(self, df):
        """Step 2: Data Preprocessing."""
        print(f"  STEP 2: Preprocessing")
        initial = len(df)
        df = df.drop_duplicates(subset=['LoanID'], keep='last')
        print(f"    Deduplication: {initial} → {len(df)} rows")
        return df

    def step3_feature_engineering(self, df):
        """Step 3: Feature Engineering (300+ features)."""
        print(f"  STEP 3: Feature Engineering")
        from feature_engineering.feature_engineer import FeatureEngineer
        fe = FeatureEngineer(df)
        df = fe.run()
        return df

    def step4_model_inference(self, df):
        """Step 4: ML Model Inference (ensemble)."""
        print(f"\n  STEP 4: Model Inference")
        import joblib
        model_path = 'data/models/ensemble_model.pkl'
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            models = data['models']
            scaler = data['scaler']
            feature_names = data['feature_names']
            weights = data['weights']

            # Prepare features
            X = df[feature_names].fillna(-999) if all(f in df.columns for f in feature_names) else None
            if X is not None:
                X_scaled = scaler.transform(X)
                probas = {}
                for name, model in models.items():
                    if name == 'logistic':
                        probas[name] = model.predict_proba(X_scaled)[:, 1]
                    else:
                        probas[name] = model.predict_proba(X)[:, 1]

                df['ensemble_probability'] = sum(
                    weights[name] * probas[name] for name in probas
                )
                print(f"    Ensemble predictions generated for {len(df):,} customers")
            else:
                print("    ⚠ Feature mismatch, using risk scorer only")
                df['ensemble_probability'] = 0.5
        else:
            print(f"    ⚠ No ensemble model found at {model_path}, using risk scorer only")
            df['ensemble_probability'] = 0.5
        return df

    def step5_risk_stratification(self, df):
        """Step 5: Risk Scoring & Stratification."""
        print(f"\n  STEP 5: Risk Stratification (15-component)")
        from risk_scoring.risk_scorer import RiskScorer
        scorer = RiskScorer()
        df = scorer.score_dataframe(df)

        for band in ['SAFE', 'LOW RISK', 'MODERATE', 'HIGH RISK', 'CRITICAL']:
            count = (df['risk_band'] == band).sum()
            pct = count / len(df) * 100
            print(f"    {band:<12s}: {count:>8,}  ({pct:>5.1f}%)")
        return df

    def step6_intervention(self, df):
        """Step 6: Intervention Triggering & Scenario Detection."""
        print(f"\n  STEP 6: Intervention & Scenario Detection")

        interventions = []
        scenarios_triggered = {}

        at_risk = df[df['risk_band'].isin(['MODERATE', 'HIGH RISK', 'CRITICAL'])]
        print(f"    Customers at risk: {len(at_risk):,}")

        for _, row in at_risk.iterrows():
            row_dict = row.to_dict()
            band = row_dict['risk_band']
            emi = row_dict.get('LoanAmount', 0) / max(row_dict.get('LoanTerm', 1), 1)

            # Check special scenarios
            scenario_result = None
            for handler in self.scenario_methods:
                result = handler(row_dict, row_dict.get('risk_score_v2', 50))
                if result:
                    scenario_result = result
                    scenario_name = result['scenario']
                    scenarios_triggered[scenario_name] = scenarios_triggered.get(scenario_name, 0) + 1
                    if result.get('risk_override'):
                        band = result['risk_override']
                    break

            # Get intervention template
            template_info = INTERVENTION_TEMPLATES.get(band)
            if template_info:
                msg = template_info['template'].format(
                    name=row_dict.get('name', f"Customer {row_dict.get('LoanID', '')}"),
                    emi=emi,
                    emi_3x=emi * 3,
                    due_date='upcoming',
                    days_to_emi=np.random.randint(3, 21),
                    emergency_loan=emi * 2
                )

                interventions.append({
                    'LoanID': row_dict.get('LoanID'),
                    'risk_band': band,
                    'risk_score': row_dict.get('risk_score_v2', 0),
                    'channel': template_info['channel'],
                    'timing': template_info['timing'],
                    'message': msg[:200] + '...' if len(msg) > 200 else msg,
                    'scenario': scenario_result['scenario'] if scenario_result else None,
                })

        print(f"    Interventions generated: {len(interventions):,}")
        if scenarios_triggered:
            print(f"    Special scenarios detected:")
            for s, c in sorted(scenarios_triggered.items(), key=lambda x: -x[1]):
                print(f"      {s}: {c:,}")

        return df, interventions

    def step7_monitoring(self, df, interventions):
        """Step 7: Monitoring & Feedback Summary."""
        print(f"\n  STEP 7: Monitoring Summary")
        total = len(df)
        at_risk = len([i for i in interventions])
        print(f"    Total customers: {total:,}")
        print(f"    Interventions: {at_risk:,}")
        print(f"    Intervention rate: {at_risk/total*100:.2f}%")

        # Simulated outcomes
        success_rate = 0.72  # 72% intervention success (spec target: 70-75%)
        prevented = int(at_risk * success_rate)
        cost_per_intervention = 50
        cost_per_default = 8000
        savings = prevented * cost_per_default - at_risk * cost_per_intervention
        roi = savings / max(at_risk * cost_per_intervention, 1) * 100

        print(f"\n    Projected Outcomes (based on 72% success rate):")
        print(f"    Prevented defaults: {prevented:,}")
        print(f"    Intervention cost: ₹{at_risk * cost_per_intervention:,}")
        print(f"    Potential savings: ₹{savings:,}")
        print(f"    ROI: {roi:,.0f}%")

        # Save interventions
        int_df = pd.DataFrame(interventions)
        os.makedirs('data/output', exist_ok=True)
        int_df.to_csv('data/output/interventions.csv', index=False)
        print(f"\n    ✓ Interventions saved to data/output/interventions.csv")
        return int_df

    def run(self, data_path=None):
        """Execute full 7-step workflow."""
        if data_path is None:
            data_path = 'data/processed/featured_loan_data.parquet'
            if not os.path.exists(data_path):
                data_path = 'data/processed/enriched_loan_data.parquet'

        print("=" * 65)
        print("  PRE-DELINQUENCY WORKFLOW ENGINE")
        print(f"  Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)

        df = self.step1_ingest(data_path)
        df = self.step2_preprocess(df)
        df = self.step3_feature_engineering(df)
        df = self.step4_model_inference(df)
        df = self.step5_risk_stratification(df)
        df, interventions = self.step6_intervention(df)
        int_df = self.step7_monitoring(df, interventions)

        # Save final scored dataset
        final_path = 'data/processed/final_scored_data.parquet'
        df.to_parquet(final_path, index=False)
        print(f"\n{'=' * 65}")
        print(f"  ✓ WORKFLOW COMPLETE")
        print(f"  Final dataset: {final_path} ({len(df):,} rows, {len(df.columns)} cols)")
        print(f"  Interventions: {len(int_df):,}")
        print(f"{'=' * 65}")

        return df, int_df


if __name__ == "__main__":
    engine = WorkflowEngine()
    df, interventions = engine.run()
