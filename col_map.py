# Column mapping
# calc_income_stability_index: ['isi', 'isi_band']
# calc_debt_to_income: ['computed_dti', 'dti_band']
# calc_savings_rate: ['savings_band', 'savings_rate_pct']
# calc_emi_cushion_index: ['eci', 'eci_band']
# calc_expenditure_volatility: ['evs', 'evs_ratio']
# calc_salary_delay_trend: ['sdt', 'sdt_band']
# calc_cash_deposit_pattern_score: ['cash_sufficiency_ratio', 'cdps', 'cdps_band']
# calc_discretionary_spending_ratio: ['dsr', 'dsr_band']
# calc_payment_timing_consistency: ['ptc', 'ptc_band']
# calc_account_balance_trajectory: ['abt', 'abt_band']
# calc_subscription_cascade_index: ['subscription_cascade_score']
# calc_digital_wallet_velocity: ['wallet_velocity', 'wallet_velocity_band']
# calc_merchant_downgrade_score: ['merchant_downgrade_band', 'merchant_downgrade_score']
# calc_healthcare_spike_indicator: ['healthcare_current_spend', 'healthcare_spike_band', 'healthcare_spike_ratio']
# Traceback (most recent call last):
#   File "d:\loan\venv\lib\site-packages\pandas\core\indexes\base.py", line 3802, in get_loc
#     return self._engine.get_loc(casted_key)
#   File "pandas\_libs\index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
#   File "pandas\_libs\index.pyx", line 165, in pandas._libs.index.IndexEngine.get_loc
#   File "pandas\_libs\hashtable_class_helper.pxi", line 5745, in pandas._libs.hashtable.PyObjectHashTable.get_item
#   File "pandas\_libs\hashtable_class_helper.pxi", line 5753, in pandas._libs.hashtable.PyObjectHashTable.get_item
# KeyError: 'industry_stress_index'
# 
# The above exception was the direct cause of the following exception:
# 
# Traceback (most recent call last):
#   File "<string>", line 55, in <module>
#   File "d:\loan\predelinquency-engine\src\feature_engineering\feature_engineer.py", line 287, in calc_employer_contagion_risk
#     industry_stress = df['industry_stress_index'].fillna(30) / 100
#   File "d:\loan\venv\lib\site-packages\pandas\core\frame.py", line 3807, in __getitem__
#     indexer = self.columns.get_loc(key)
#   File "d:\loan\venv\lib\site-packages\pandas\core\indexes\base.py", line 3804, in get_loc
#     raise KeyError(key) from err
# KeyError: 'industry_stress_index'
# 