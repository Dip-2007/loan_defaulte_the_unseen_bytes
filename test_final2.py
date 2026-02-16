# Final Test Results v2
# ============================= test session starts =============================
# collecting ... collected 59 items
# 
# tests/test_api.py::TestHealthEndpoint::test_health_returns_200[asyncio] PASSED [  1%]
# tests/test_api.py::TestHealthEndpoint::test_health_reports_model_status[asyncio] PASSED [  3%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_returns_200[asyncio] FAILED [  5%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_has_required_fields[asyncio] FAILED [  6%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_valid_band[asyncio] FAILED [  8%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_range[asyncio] FAILED [ 10%]
# tests/test_api.py::TestRiskScoreEndpoint::test_high_dti_increases_risk[asyncio] FAILED [ 11%]
# tests/test_api.py::TestPredictEndpoint::test_predict_returns_200_or_503[asyncio] PASSED [ 13%]
# tests/test_api.py::TestPredictEndpoint::test_predict_response_format[asyncio] PASSED [ 15%]
# tests/test_api.py::TestExplainEndpoint::test_explain_returns_200_or_503[asyncio] PASSED [ 16%]
# tests/test_api.py::TestExplainEndpoint::test_explain_response_format[asyncio] PASSED [ 18%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_income_stability_index PASSED [ 20%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_debt_to_income PASSED [ 22%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_savings_rate PASSED [ 23%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_emi_cushion_index PASSED [ 25%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_expenditure_volatility PASSED [ 27%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_salary_delay_trend PASSED [ 28%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_cash_deposit_pattern_score PASSED [ 30%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_discretionary_spending_ratio PASSED [ 32%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_payment_timing_consistency PASSED [ 33%]
# tests/test_feature_engineering.py::TestCoreFormulas::test_account_balance_trajectory PASSED [ 35%]
# tests/test_feature_engineering.py::TestAdvancedBehavioral::test_subscription_cascade PASSED [ 37%]
# tests/test_feature_engineering.py::TestAdvancedBehavioral::test_digital_wallet_velocity PASSED [ 38%]
# tests/test_feature_engineering.py::TestAdvancedBehavioral::test_merchant_downgrade PASSED [ 40%]
# tests/test_feature_engineering.py::TestAdvancedBehavioral::test_healthcare_spike PASSED [ 42%]
# tests/test_feature_engineering.py::TestAdvancedBehavioral::test_employer_contagion PASSED [ 44%]
# tests/test_feature_engineering.py::TestAdvancedBehavioral::test_gig_income_variance PASSED [ 45%]
# tests/test_feature_engineering.py::TestFullPipeline::test_run_produces_features PASSED [ 47%]
# tests/test_feature_engineering.py::TestFullPipeline::test_no_all_nan_columns PASSED [ 49%]
# tests/test_risk_scorer.py::TestRiskBandClassification::test_safe_band PASSED [ 50%]
# tests/test_risk_scorer.py::TestRiskBandClassification::test_low_risk_band PASSED [ 52%]
# tests/test_risk_scorer.py::TestRiskBandClassification::test_moderate_band PASSED [ 54%]
# tests/test_risk_scorer.py::TestRiskBandClassification::test_high_risk_band PASSED [ 55%]
# tests/test_risk_scorer.py::TestRiskBandClassification::test_critical_band PASSED [ 57%]
# tests/test_risk_scorer.py::TestComponentScores::test_income_stability_low_volatility PASSED [ 59%]
# tests/test_risk_scorer.py::TestComponentScores::test_income_stability_high_volatility PASSED [ 61%]
# tests/test_risk_scorer.py::TestComponentScores::test_debt_burden_low_dti PASSED [ 62%]
# tests/test_risk_scorer.py::TestComponentScores::test_debt_burden_critical_dti PASSED [ 64%]
# tests/test_risk_scorer.py::TestComponentScores::test_credit_score_formula PASSED [ 66%]
# tests/test_risk_scorer.py::TestComponentScores::test_emi_cushion_safe PASSED [ 67%]
# tests/test_risk_scorer.py::TestComponentScores::test_emi_cushion_critical PASSED [ 69%]
# tests/test_risk_scorer.py::TestRajeshKumarExample::test_worked_example_function PASSED [ 71%]
# tests/test_risk_scorer.py::TestBatchScoring::test_score_dataframe PASSED [ 72%]
# tests/test_risk_scorer.py::TestBatchScoring::test_score_deterministic PASSED [ 74%]
# tests/test_risk_scorer.py::TestWeights::test_weights_sum_to_one PASSED   [ 76%]
# tests/test_risk_scorer.py::TestWeights::test_all_components_have_weights PASSED [ 77%]
# tests/test_workflow.py::TestInterventionTemplates::test_all_bands_have_templates PASSED [ 79%]
# tests/test_workflow.py::TestInterventionTemplates::test_safe_has_no_intervention PASSED [ 81%]
# tests/test_workflow.py::TestInterventionTemplates::test_critical_uses_phone PASSED [ 83%]
# tests/test_workflow.py::TestInterventionTemplates::test_moderate_uses_sms PASSED [ 84%]
# tests/test_workflow.py::TestScenarioHandlers::test_scenario_1_cash_businessman PASSED [ 86%]
# tests/test_workflow.py::TestScenarioHandlers::test_scenario_2_student PASSED [ 88%]
# tests/test_workflow.py::TestScenarioHandlers::test_scenario_3_multiple_loans PASSED [ 89%]
# tests/test_workflow.py::TestScenarioHandlers::test_scenario_4_retired_healthcare PASSED [ 91%]
# tests/test_workflow.py::TestScenarioHandlers::test_scenario_5_gig_suspension PASSED [ 93%]
# tests/test_workflow.py::TestScenarioHandlers::test_scenario_6_employer_layoff PASSED [ 94%]
# tests/test_workflow.py::TestWorkflowEngine::test_engine_init PASSED      [ 96%]
# tests/test_workflow.py::TestWorkflowEngine::test_step1_ingest PASSED     [ 98%]
# tests/test_workflow.py::TestWorkflowEngine::test_step2_preprocess PASSED [100%]
# 
# ================================== FAILURES ===================================
# _________ TestRiskScoreEndpoint.test_risk_score_returns_200[asyncio] __________
# tests\test_api.py:56: in test_risk_score_returns_200
#     assert response.status_code == 200
# E   assert 500 == 200
# E    +  where 500 = <Response [500 Internal Server Error]>.status_code
# _____ TestRiskScoreEndpoint.test_risk_score_has_required_fields[asyncio] ______
# tests\test_api.py:61: in test_risk_score_has_required_fields
#     assert "risk_score" in data
# E   AssertionError: assert 'risk_score' in {'detail': 'tuple indices must be integers or slices, not str'}
# __________ TestRiskScoreEndpoint.test_risk_score_valid_band[asyncio] __________
# tests\test_api.py:70: in test_risk_score_valid_band
#     assert data["risk_band"] in valid_bands
# E   KeyError: 'risk_band'
# ____________ TestRiskScoreEndpoint.test_risk_score_range[asyncio] _____________
# tests\test_api.py:75: in test_risk_score_range
#     assert 0 <= data["risk_score"] <= 100
# E   KeyError: 'risk_score'
# _________ TestRiskScoreEndpoint.test_high_dti_increases_risk[asyncio] _________
# tests\test_api.py:80: in test_high_dti_increases_risk
#     assert high["risk_score"] > low["risk_score"]
# E   KeyError: 'risk_score'
# =========================== short test summary info ===========================
# FAILED tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_returns_200[asyncio]
# FAILED tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_has_required_fields[asyncio]
# FAILED tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_valid_band[asyncio]
# FAILED tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_range[asyncio]
# FAILED tests/test_api.py::TestRiskScoreEndpoint::test_high_dti_increases_risk[asyncio]
# ======================== 5 failed, 54 passed in 2.43s =========================
# 