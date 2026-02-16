# src/risk_scoring/risk_scorer.py
"""
Section 5: Comprehensive Risk Scoring Model
- 15-component weighted risk score (0-100)
- 5-band classification: SAFE / LOW RISK / MODERATE / HIGH RISK / CRITICAL
"""

import pandas as pd
import numpy as np


# ============================================================
# 5.2 Component Weights (sum = 1.00)
# ============================================================
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
# Sum = 1.00 (verified)

# 5.4 Risk Bands
RISK_BANDS = [
    (0,  25, 'SAFE',      '< 5%',      'Monitor only'),
    (26, 45, 'LOW RISK',  '5-15%',     'Gentle nudge, wellness tips'),
    (46, 60, 'MODERATE',  '15-35%',    'Proactive outreach, options'),
    (61, 75, 'HIGH RISK', '35-60%',    'Urgent call, restructuring'),
    (76, 100, 'CRITICAL', '> 60%',     'Immediate intervention'),
]


class RiskScorer:
    """15-component risk scoring engine per Section 5 spec."""

    def __init__(self):
        self.weights = COMPONENT_WEIGHTS

    # ================================================================
    # Individual Component Scorers (each returns 0-100)
    # ================================================================

    def _score_income_stability(self, row):
        """ISI < 10%: 10, ISI 10-25: 30, ISI 25-50: 60, ISI > 50: 90"""
        isi = row.get('isi', row.get('income_stability_index', 15)) * 100 if row.get('isi', 0) < 2 else row.get('isi', 15)
        if isi < 10:   return 10
        elif isi < 25: return 30
        elif isi < 50: return 60
        else:          return 90

    def _score_debt_burden(self, row):
        """DTI < 30%: 15, 30-40: 40, 40-50: 70, >50: 95"""
        dti = row.get('computed_dti', row.get('DTIRatio', 30))
        if dti < 30:   return 15
        elif dti < 40: return 40
        elif dti < 50: return 70
        else:          return 95

    def _score_savings_adequacy(self, row):
        """100 - SR (inverted). Low savings = high score."""
        sr = row.get('savings_rate_pct', row.get('savings_rate', 20) * 100 if row.get('savings_rate', 0) < 2 else row.get('savings_rate', 20))
        return np.clip(100 - sr, 0, 100)

    def _score_payment_history(self, row):
        """DPD + consistency. Perfect history = 5, delays = higher."""
        on_time = row.get('ontime_payment_rate_12m', 0.95)
        avg_delay = row.get('avg_delay_days', 0)
        dpd_count = row.get('num_dpd_30_plus', 0)
        score = (1 - on_time) * 50 + min(avg_delay, 30) * 1.5 + dpd_count * 5
        return np.clip(score, 0, 100)

    def _score_emi_cushion(self, row):
        """ECI > 2: 10, 1-2: 30, 0.5-1: 60, <0.5: 90"""
        eci = row.get('eci', 1.0)
        return np.clip(100 * (1 - eci / 2), 0, 100)

    def _score_expenditure_pattern(self, row):
        """EVS volatility + DSR drop signal."""
        evs_ratio = row.get('evs_ratio', 0.2)
        dsr = row.get('dsr', 20)
        score = evs_ratio * 50 + max(0, 30 - dsr) * 2
        return np.clip(score, 0, 100)

    def _score_cash_flow(self, row):
        """For business: CDPS + CSR. For salaried: 0."""
        segment = row.get('segment_category', 'EMPLOYED')
        if segment != 'BUSINESS_OWNER':
            return 0  # N/A for non-business
        cdps = row.get('cdps', 70)
        csr = row.get('cash_sufficiency_ratio', 1.2)
        if cdps is None or np.isnan(cdps): return 30
        score = max(0, 100 - cdps) * 0.6 + max(0, 1.5 - csr) * 40
        return np.clip(score, 0, 100)

    def _score_credit_score(self, row):
        """(900 - CIBIL) / 6"""
        cs = row.get('CreditScore', 700)
        return np.clip((900 - cs) / 6, 0, 100)

    def _score_employer_health(self, row):
        """Employer health + peer defaults + industry stress."""
        segment = row.get('segment_category', 'EMPLOYED')
        if segment != 'EMPLOYED':
            return 10  # Minimal for non-salaried
        employer = row.get('employer_health_score', 70)
        peer = row.get('peer_default_rate', 0.05)
        return np.clip((100 - employer) * 0.5 + peer * 200, 0, 100)

    def _score_healthcare_costs(self, row):
        """Spike ratio. Normal (1x) = 10, Emergency (3x+) = 80."""
        spike = row.get('healthcare_spike_ratio', 1.0)
        if spike is None or np.isnan(spike): spike = 1.0
        if spike < 2:   return 10
        elif spike < 3: return 50
        else:           return 80

    def _score_network_risk(self, row):
        """Family + employer + merchant network effects."""
        employer_mult = row.get('employer_risk_multiplier', 1.0)
        merchant_deg = row.get('merchant_downgrade_score', 0)
        parent_stable = row.get('parent_income_stable', 1)
        score = (employer_mult - 1) * 30 + merchant_deg * 5
        if parent_stable == 0:
            score += 20
        return np.clip(score, 0, 100)

    def _score_behavioral_signals(self, row):
        """Subscription cascade + wallet velocity + merchant downgrade."""
        cascade = row.get('subscription_cascade_score', 0.25)
        velocity = row.get('wallet_velocity', 0.2)
        score = cascade * 40 + velocity * 30
        return np.clip(score, 0, 100)

    def _score_life_events(self, row):
        """Job loss, medical emergency, major expense detection."""
        healthcare_spike = row.get('healthcare_spike_ratio', 1.0) or 1.0
        sdt = row.get('sdt', 0) or 0
        score = 0
        if healthcare_spike > 3: score += 40
        if sdt > 5: score += 30  # salary delay = employer trouble
        dpd = row.get('max_dpd_last_12m', 0)
        if dpd > 60: score += 30
        return np.clip(score, 0, 100)

    def _score_age_vintage(self, row):
        """Young + new accounts = higher risk."""
        age = row.get('Age', 35)
        oldest_cl = row.get('oldest_credit_line_years', 5)
        score = 0
        if age < 25: score += 20
        elif age > 60: score += 15
        if oldest_cl < 2: score += 30
        return np.clip(score, 0, 100)

    def _score_external_factors(self, row):
        """Seasonal, industry, regulatory. Simulated."""
        industry = row.get('industry_stress_index', 30) or 30
        return np.clip(industry, 0, 100)

    # ================================================================
    # Main Scoring
    # ================================================================

    def score_customer(self, row):
        """Calculate final risk score for a single customer (0-100)."""
        components = {
            'income_stability':    self._score_income_stability(row),
            'debt_burden':         self._score_debt_burden(row),
            'savings_adequacy':    self._score_savings_adequacy(row),
            'payment_history':     self._score_payment_history(row),
            'emi_cushion':         self._score_emi_cushion(row),
            'expenditure_pattern': self._score_expenditure_pattern(row),
            'cash_flow':           self._score_cash_flow(row),
            'credit_score_adj':    self._score_credit_score(row),
            'employer_health':     self._score_employer_health(row),
            'healthcare_costs':    self._score_healthcare_costs(row),
            'network_risk':        self._score_network_risk(row),
            'behavioral_signals':  self._score_behavioral_signals(row),
            'life_events':         self._score_life_events(row),
            'age_vintage':         self._score_age_vintage(row),
            'external_factors':    self._score_external_factors(row),
        }

        final_score = sum(
            components[k] * self.weights[k] for k in components
        )
        return round(final_score, 2), components

    def classify_risk(self, score):
        """Classify score into 5-band risk level."""
        for lo, hi, band, prob, action in RISK_BANDS:
            if lo <= score <= hi:
                return band, prob, action
        return 'CRITICAL', '> 60%', 'Immediate intervention'

    def score_dataframe(self, df):
        """Score all customers in a DataFrame."""
        scores = []
        bands = []
        all_components = []

        for _, row in df.iterrows():
            score, components = self.score_customer(row)
            band, prob, action = self.classify_risk(score)
            scores.append(score)
            bands.append(band)
            all_components.append(components)

        df = df.copy()
        df['risk_score_v2'] = scores
        df['risk_band'] = bands

        # Add component scores as columns
        comp_df = pd.DataFrame(all_components)
        for col in comp_df.columns:
            df[f'comp_{col}'] = comp_df[col].values

        return df


def validate_worked_example():
    """Section 5.5: Validate with Rajesh Kumar example."""
    print("=" * 60)
    print("WORKED EXAMPLE: Rajesh Kumar (Salaried IT Professional)")
    print("=" * 60)

    rajesh = {
        'Age': 32,
        'segment_category': 'EMPLOYED',
        'Income': 85000,
        'LoanAmount': 540000,  # ₹15k EMI × 36 months
        'LoanTerm': 36,
        'CreditScore': 720,
        'DTIRatio': 23.5,
        'isi': 1.22,
        'computed_dti': 23.5,
        'savings_rate_pct': 33.0,
        'avg_monthly_balance_6m': 28000,
        'ontime_payment_rate_12m': 0.83,
        'avg_delay_days': 1.0,
        'num_dpd_30_plus': 0,
        'eci': 1.0,
        'evs_ratio': 0.15,
        'dsr': 15.0,
        'cdps': None,
        'cash_sufficiency_ratio': None,
        'employer_health_score': 85,
        'peer_default_rate': 0.02,
        'industry_stress_index': 15,
        'healthcare_spike_ratio': 3.0,  # 3x spike
        'employer_risk_multiplier': 1.0,
        'merchant_downgrade_score': 2,
        'parent_income_stable': 1,
        'subscription_cascade_score': 0.50,  # Phase 2
        'wallet_velocity': 0.3,
        'sdt': 0,
        'max_dpd_last_12m': 7,
        'oldest_credit_line_years': 8,
        'salary_delay_days': 0,
    }

    scorer = RiskScorer()
    score, components = scorer.score_customer(rajesh)
    band, prob, action = scorer.classify_risk(score)

    print(f"\n{'Component':<25s} {'Score':>8s} {'Weight':>8s} {'Weighted':>10s}")
    print("-" * 55)
    for k, v in components.items():
        w = COMPONENT_WEIGHTS[k]
        print(f"  {k:<23s} {v:>7.1f}  ×  {w:.2f}  =  {v*w:>7.2f}")

    print("-" * 55)
    print(f"  {'FINAL SCORE':<23s} {'':>7s}  {'':>7s}  {score:>8.2f}")
    print(f"\n  Risk Band: {band}")
    print(f"  Default Probability: {prob}")
    print(f"  Action: {action}")
    print(f"\n  Expected: ~38.5 (LOW RISK)")
    return score, band


if __name__ == "__main__":
    # Validate worked example
    score, band = validate_worked_example()

    # Score full dataset
    print(f"\n{'=' * 60}")
    print("SCORING FULL DATASET")
    print(f"{'=' * 60}")

    import os
    data_path = 'data/processed/featured_loan_data.parquet'
    if not os.path.exists(data_path):
        data_path = 'data/processed/enriched_loan_data.parquet'

    df = pd.read_parquet(data_path)
    print(f"Loaded: {len(df):,} rows")

    scorer = RiskScorer()
    df_scored = scorer.score_dataframe(df)

    print(f"\nRisk Band Distribution:")
    dist = df_scored['risk_band'].value_counts()
    for band in ['SAFE', 'LOW RISK', 'MODERATE', 'HIGH RISK', 'CRITICAL']:
        count = dist.get(band, 0)
        pct = count / len(df_scored) * 100
        print(f"  {band:<12s}: {count:>8,}  ({pct:>5.1f}%)")

    print(f"\nRisk Score Stats:")
    print(f"  Mean: {df_scored['risk_score_v2'].mean():.2f}")
    print(f"  Median: {df_scored['risk_score_v2'].median():.2f}")
    print(f"  Std: {df_scored['risk_score_v2'].std():.2f}")

    # Save scored data
    out = 'data/processed/scored_loan_data.parquet'
    df_scored.to_parquet(out, index=False)
    print(f"\n✓ Saved to: {out}")
