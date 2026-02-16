# Final Test Results
# ============================= test session starts =============================
# collecting ... collected 59 items
# 
# tests/test_api.py::TestHealthEndpoint::test_health_returns_200 ERROR     [  1%]
# tests/test_api.py::TestHealthEndpoint::test_health_reports_model_status ERROR [  3%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_returns_200 ERROR [  5%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_has_required_fields ERROR [  6%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_valid_band ERROR [  8%]
# tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_range ERROR    [ 10%]
# tests/test_api.py::TestRiskScoreEndpoint::test_high_dti_increases_risk ERROR [ 11%]
# tests/test_api.py::TestPredictEndpoint::test_predict_returns_200_or_503 ERROR [ 13%]
# tests/test_api.py::TestPredictEndpoint::test_predict_response_format ERROR [ 15%]
# tests/test_api.py::TestExplainEndpoint::test_explain_returns_200_or_503 ERROR [ 16%]
# tests/test_api.py::TestExplainEndpoint::test_explain_response_format ERROR [ 18%]
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
# tests/test_feature_engineering.py::TestFullPipeline::test_run_produces_features FAILED [ 47%]
# tests/test_feature_engineering.py::TestFullPipeline::test_no_all_nan_columns FAILED [ 49%]
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
# =================================== ERRORS ====================================
# ________ ERROR at setup of TestHealthEndpoint.test_health_returns_200 _________
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# ____ ERROR at setup of TestHealthEndpoint.test_health_reports_model_status ____
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# _____ ERROR at setup of TestRiskScoreEndpoint.test_risk_score_returns_200 _____
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# _ ERROR at setup of TestRiskScoreEndpoint.test_risk_score_has_required_fields _
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# _____ ERROR at setup of TestRiskScoreEndpoint.test_risk_score_valid_band ______
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# ________ ERROR at setup of TestRiskScoreEndpoint.test_risk_score_range ________
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# ____ ERROR at setup of TestRiskScoreEndpoint.test_high_dti_increases_risk _____
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# ____ ERROR at setup of TestPredictEndpoint.test_predict_returns_200_or_503 ____
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# _____ ERROR at setup of TestPredictEndpoint.test_predict_response_format ______
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# ____ ERROR at setup of TestExplainEndpoint.test_explain_returns_200_or_503 ____
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# _____ ERROR at setup of TestExplainEndpoint.test_explain_response_format ______
# tests\test_api.py:19: in client
#     with httpx.Client(transport=transport, base_url="http://test") as c:
# ..\venv\lib\site-packages\httpx\_client.py:1287: in __enter__
#     self._transport.__enter__()
# E   AttributeError: 'ASGITransport' object has no attribute '__enter__'. Did you mean: '__aenter__'?
# ================================== FAILURES ===================================
# _________________ TestFullPipeline.test_run_produces_features _________________
# ..\venv\lib\site-packages\pandas\core\indexes\base.py:3802: in get_loc
#     return self._engine.get_loc(casted_key)
# pandas\_libs\index.pyx:138: in pandas._libs.index.IndexEngine.get_loc
#     ???
# pandas\_libs\index.pyx:165: in pandas._libs.index.IndexEngine.get_loc
#     ???
# pandas\_libs\hashtable_class_helper.pxi:5745: in pandas._libs.hashtable.PyObjectHashTable.get_item
#     ???
# pandas\_libs\hashtable_class_helper.pxi:5753: in pandas._libs.hashtable.PyObjectHashTable.get_item
#     ???
# E   KeyError: 'num_dependents'
# 
# The above exception was the direct cause of the following exception:
# tests\test_feature_engineering.py:198: in test_run_produces_features
#     result = fe.run()
# src\feature_engineering\feature_engineer.py:416: in run
#     self.df = fn()
# src\feature_engineering\feature_engineer.py:319: in calc_cross_features
#     df['Income'] / np.maximum(df['num_dependents'] + 1, 1)
# ..\venv\lib\site-packages\pandas\core\frame.py:3807: in __getitem__
#     indexer = self.columns.get_loc(key)
# ..\venv\lib\site-packages\pandas\core\indexes\base.py:3804: in get_loc
#     raise KeyError(key) from err
# E   KeyError: 'num_dependents'
# ---------------------------- Captured stdout call -----------------------------
# ============================================================\nFEATURE ENGINEERING PIPELINE (Section 4)\n============================================================\nInput: 50 rows, 35 columns\n  \u2713 ISI (Income Stability Index)        \u2192 37 columns\n  \u2713 DTI (Debt-to-Income)                \u2192 39 columns\n  \u2713 SR (Savings Rate)                   \u2192 41 columns\n  \u2713 ECI (EMI Cushion Index)             \u2192 43 columns\n  \u2713 EVS (Expenditure Volatility)        \u2192 45 columns\n  \u2713 SDT (Salary Delay Trend)            \u2192 47 columns\n  \u2713 CDPS (Cash Deposit Pattern)         \u2192 50 columns\n  \u2713 DSR (Discretionary Spending)        \u2192 52 columns\n  \u2713 PTC (Payment Timing)                \u2192 54 columns\n  \u2713 ABT (Balance Trajectory)            \u2192 56 columns\n  \u2713 Subscription Cascade                \u2192 57 columns\n  \u2713 Wallet Velocity                     \u2192 59 columns\n  \u2713 Merchant Downgrade                  \u2192 61 columns\n  \u2713 Healthcare Spike                    \u2192 64 columns\n  \u2713 Employer Contagion                  \u2192 65 columns\n  \u2713 Gig Income Variance                 \u2192 66 columns
# __________________ TestFullPipeline.test_no_all_nan_columns ___________________
# ..\venv\lib\site-packages\pandas\core\indexes\base.py:3802: in get_loc
#     return self._engine.get_loc(casted_key)
# pandas\_libs\index.pyx:138: in pandas._libs.index.IndexEngine.get_loc
#     ???
# pandas\_libs\index.pyx:165: in pandas._libs.index.IndexEngine.get_loc
#     ???
# pandas\_libs\hashtable_class_helper.pxi:5745: in pandas._libs.hashtable.PyObjectHashTable.get_item
#     ???
# pandas\_libs\hashtable_class_helper.pxi:5753: in pandas._libs.hashtable.PyObjectHashTable.get_item
#     ???
# E   KeyError: 'num_dependents'
# 
# The above exception was the direct cause of the following exception:
# tests\test_feature_engineering.py:206: in test_no_all_nan_columns
#     result = fe.run()
# src\feature_engineering\feature_engineer.py:416: in run
#     self.df = fn()
# src\feature_engineering\feature_engineer.py:319: in calc_cross_features
#     df['Income'] / np.maximum(df['num_dependents'] + 1, 1)
# ..\venv\lib\site-packages\pandas\core\frame.py:3807: in __getitem__
#     indexer = self.columns.get_loc(key)
# ..\venv\lib\site-packages\pandas\core\indexes\base.py:3804: in get_loc
#     raise KeyError(key) from err
# E   KeyError: 'num_dependents'
# ---------------------------- Captured stdout call -----------------------------
# ============================================================\nFEATURE ENGINEERING PIPELINE (Section 4)\n============================================================\nInput: 50 rows, 35 columns\n  \u2713 ISI (Income Stability Index)        \u2192 37 columns\n  \u2713 DTI (Debt-to-Income)                \u2192 39 columns\n  \u2713 SR (Savings Rate)                   \u2192 41 columns\n  \u2713 ECI (EMI Cushion Index)             \u2192 43 columns\n  \u2713 EVS (Expenditure Volatility)        \u2192 45 columns\n  \u2713 SDT (Salary Delay Trend)            \u2192 47 columns\n  \u2713 CDPS (Cash Deposit Pattern)         \u2192 50 columns\n  \u2713 DSR (Discretionary Spending)        \u2192 52 columns\n  \u2713 PTC (Payment Timing)                \u2192 54 columns\n  \u2713 ABT (Balance Trajectory)            \u2192 56 columns\n  \u2713 Subscription Cascade                \u2192 57 columns\n  \u2713 Wallet Velocity                     \u2192 59 columns\n  \u2713 Merchant Downgrade                  \u2192 61 columns\n  \u2713 Healthcare Spike                    \u2192 64 columns\n  \u2713 Employer Contagion                  \u2192 65 columns\n  \u2713 Gig Income Variance                 \u2192 66 columns
# =========================== short test summary info ===========================
# FAILED tests/test_feature_engineering.py::TestFullPipeline::test_run_produces_features
# FAILED tests/test_feature_engineering.py::TestFullPipeline::test_no_all_nan_columns
# ERROR tests/test_api.py::TestHealthEndpoint::test_health_returns_200 - Attrib...
# ERROR tests/test_api.py::TestHealthEndpoint::test_health_reports_model_status
# ERROR tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_returns_200
# ERROR tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_has_required_fields
# ERROR tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_valid_band - ...
# ERROR tests/test_api.py::TestRiskScoreEndpoint::test_risk_score_range - Attri...
# ERROR tests/test_api.py::TestRiskScoreEndpoint::test_high_dti_increases_risk
# ERROR tests/test_api.py::TestPredictEndpoint::test_predict_returns_200_or_503
# ERROR tests/test_api.py::TestPredictEndpoint::test_predict_response_format - ...
# ERROR tests/test_api.py::TestExplainEndpoint::test_explain_returns_200_or_503
# ERROR tests/test_api.py::TestExplainEndpoint::test_explain_response_format - ...
# =================== 2 failed, 46 passed, 11 errors in 3.22s ===================
# 