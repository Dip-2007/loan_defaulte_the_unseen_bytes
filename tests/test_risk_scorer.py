# tests/test_risk_scorer.py
"""
Tests for Section 5: Risk Scoring Engine.
Includes Rajesh Kumar worked example validation (Section 5.5).
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_scoring.risk_scorer import RiskScorer, validate_worked_example, RISK_BANDS, COMPONENT_WEIGHTS


class TestRiskBandClassification:
    """Test the 5-band classification boundaries (Section 5.4)."""

    def test_safe_band(self):
        scorer = RiskScorer()
        band, _, _ = scorer.classify_risk(0)
        assert band == 'SAFE'
        band, _, _ = scorer.classify_risk(10)
        assert band == 'SAFE'
        band, _, _ = scorer.classify_risk(25)
        assert band == 'SAFE'

    def test_low_risk_band(self):
        scorer = RiskScorer()
        band, _, _ = scorer.classify_risk(26)
        assert band == 'LOW RISK'
        band, _, _ = scorer.classify_risk(35)
        assert band == 'LOW RISK'
        band, _, _ = scorer.classify_risk(45)
        assert band == 'LOW RISK'

    def test_moderate_band(self):
        scorer = RiskScorer()
        band, _, _ = scorer.classify_risk(46)
        assert band == 'MODERATE'
        band, _, _ = scorer.classify_risk(55)
        assert band == 'MODERATE'
        band, _, _ = scorer.classify_risk(60)
        assert band == 'MODERATE'

    def test_high_risk_band(self):
        scorer = RiskScorer()
        band, _, _ = scorer.classify_risk(61)
        assert band == 'HIGH RISK'
        band, _, _ = scorer.classify_risk(70)
        assert band == 'HIGH RISK'
        band, _, _ = scorer.classify_risk(75)
        assert band == 'HIGH RISK'

    def test_critical_band(self):
        scorer = RiskScorer()
        band, _, _ = scorer.classify_risk(76)
        assert band == 'CRITICAL'
        band, _, _ = scorer.classify_risk(90)
        assert band == 'CRITICAL'
        band, _, _ = scorer.classify_risk(100)
        assert band == 'CRITICAL'


class TestComponentScores:
    """Test individual component scoring functions."""

    def setup_method(self):
        self.scorer = RiskScorer()

    def test_income_stability_low_volatility(self):
        """ISI < 10% should give low score (10)."""
        row = pd.Series({'isi': 5.0})
        score = self.scorer._score_income_stability(row)
        assert 0 <= score <= 100
        # ISI=5.0 is >= 2 so is used directly → 5 < 10 → returns 10
        assert score == 10

    def test_income_stability_high_volatility(self):
        """ISI > 50% should give high score (90)."""
        row = pd.Series({'isi': 60.0})
        score = self.scorer._score_income_stability(row)
        assert score == 90

    def test_debt_burden_low_dti(self):
        """DTI < 30% should give low score (15)."""
        row = pd.Series({'computed_dti': 20.0, 'DTIRatio': 20.0})
        score = self.scorer._score_debt_burden(row)
        assert score == 15

    def test_debt_burden_critical_dti(self):
        """DTI > 50% should give high score (95)."""
        row = pd.Series({'computed_dti': 65.0, 'DTIRatio': 65.0})
        score = self.scorer._score_debt_burden(row)
        assert score == 95

    def test_credit_score_formula(self):
        """Score = (900 - CIBIL) / 6"""
        row = pd.Series({'CreditScore': 720})
        score = self.scorer._score_credit_score(row)
        expected = (900 - 720) / 6  # = 30
        assert abs(score - expected) < 5

    def test_emi_cushion_safe(self):
        """ECI > 2 should give low score (continuous formula: 100*(1-eci/2))."""
        row = pd.Series({'eci': 2.5})
        score = self.scorer._score_emi_cushion(row)
        # 100*(1 - 2.5/2) = 100*(-0.25) = -25 → clipped to 0
        assert score == 0

    def test_emi_cushion_critical(self):
        """ECI < 0.5 should give high score."""
        row = pd.Series({'eci': 0.3})
        score = self.scorer._score_emi_cushion(row)
        # 100*(1 - 0.3/2) = 100*0.85 = 85
        assert score == 85


class TestRajeshKumarExample:
    """
    Section 5.5 Worked Example: Rajesh Kumar (Salaried IT Professional)
    Expected: Risk Score ≈ 38.5, Classification = LOW RISK
    """

    def test_worked_example_function(self):
        """Validate the built-in worked example produces expected results."""
        score, band = validate_worked_example()
        # Score should be approximately 38.5 (allow some tolerance)
        assert 20 <= score <= 50, f"Expected ~38.5, got {score}"
        assert band == 'LOW RISK', f"Expected LOW RISK, got {band}"


class TestBatchScoring:
    """Test scoring entire DataFrames."""

    def test_score_dataframe(self, sample_dataframe):
        scorer = RiskScorer()
        result = scorer.score_dataframe(sample_dataframe)

        assert 'risk_score_v2' in result.columns
        assert 'risk_band' in result.columns

        # All scores should be 0-100
        assert (result['risk_score_v2'] >= 0).all()
        assert (result['risk_score_v2'] <= 100).all()

        # All bands should be valid
        valid_bands = {b[2] for b in RISK_BANDS}
        assert set(result['risk_band'].unique()).issubset(valid_bands)

    def test_score_deterministic(self, sample_customer_row):
        """Same input should produce same score."""
        scorer = RiskScorer()
        score1, _ = scorer.score_customer(sample_customer_row)
        score2, _ = scorer.score_customer(sample_customer_row)
        assert score1 == score2


class TestWeights:
    """Verify component weights sum to 1.0 (Section 5.2)."""

    def test_weights_sum_to_one(self):
        total = sum(COMPONENT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_all_components_have_weights(self):
        scorer = RiskScorer()
        expected_components = [
            'income_stability', 'debt_burden', 'savings_adequacy',
            'payment_history', 'emi_cushion', 'expenditure_pattern',
            'cash_flow', 'credit_score_adj', 'employer_health',
            'healthcare_costs', 'network_risk', 'behavioral_signals',
            'life_events', 'age_vintage', 'external_factors',
        ]
        for comp in expected_components:
            assert comp in scorer.weights, f"Missing weight for {comp}"
