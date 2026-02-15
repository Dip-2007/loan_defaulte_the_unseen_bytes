import pandas as pd
import numpy as np
import json
from datetime import datetime
import random
import os


# ---------------------------------------------------
# JSON Serializer for NumPy + Pandas types
# ---------------------------------------------------
def json_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class AccountAggregatorSimulator:
    """Generate realistic AA data showing multi-bank financial profiles"""

    def __init__(self, base_customer_df):
        self.customers = base_customer_df
        self.banks = ['HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK', 'PNB', 'BOB']
        self.fintech_lenders = ['PaySense', 'MoneyTap', 'CASHe', 'EarlySalary']

    def generate_aa_response(self, customer_id, customer_profile):
        """Generate complete AA JSON response for a customer"""

        num_banks = int(np.random.choice([2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.1]))
        customer_banks = random.sample(self.banks, num_banks)

        aa_data = {
            "customer_id": str(customer_id),  # Keep ID as string
            "consent_id": f"CONSENT_{random.randint(100000, 999999)}",
            "timestamp": datetime.now().isoformat(),
            "accounts": [],
            "mutual_funds": [],
            "tax_records": {},
            "_metadata": {}
        }

        total_emi_other_banks = 0.0
        total_balance = 0.0

        # Generate accounts
        for i, bank in enumerate(customer_banks):
            is_primary = (i == 0)

            balance = (
                np.random.uniform(10000, 100000)
                if is_primary
                else np.random.uniform(2000, 30000)
            )
            total_balance += balance

            aa_data["accounts"].append({
                "fip_id": bank,
                "account_type": "SAVINGS",
                "masked_account": f"XXXX{random.randint(1000, 9999)}",
                "balance": round(float(balance), 2),
                "is_primary": is_primary
            })

            # Loans at non-primary banks
            if np.random.random() < 0.4 and not is_primary:
                loan_type = np.random.choice(['HOME_LOAN', 'CAR_LOAN', 'PERSONAL_LOAN'])
                emi = np.random.uniform(5000, 25000)
                total_emi_other_banks += emi

                aa_data["accounts"].append({
                    "fip_id": bank,
                    "account_type": "LOAN",
                    "loan_type": str(loan_type),
                    "emi": round(float(emi), 2),
                    "outstanding": round(float(np.random.uniform(50000, 500000)), 2)
                })

        # Add fintech loans for stressed customers
        if int(customer_profile['target']) == 1 and np.random.random() < 0.6:
            fintech = random.choice(self.fintech_lenders)
            emi = np.random.uniform(2000, 8000)
            total_emi_other_banks += emi

            aa_data["accounts"].append({
                "fip_id": fintech,
                "account_type": "LOAN",
                "loan_type": "PERSONAL_LOAN_DIGITAL",
                "emi": round(float(emi), 2),
                "outstanding": round(float(np.random.uniform(5000, 50000)), 2),
                "interest_rate": round(float(np.random.uniform(18, 36)), 2)
            })

        # Mutual Funds
        if np.random.random() < 0.3:
            aa_data["mutual_funds"].append({
                "fund_name": random.choice(['HDFC Balanced', 'ICICI Prudential']),
                "current_value": round(float(np.random.uniform(10000, 500000)), 2)
            })

        # Tax Records
        true_income = float(customer_profile['monthly_income']) * 12
        aa_data["tax_records"] = {
            "itr": {
                "2023-24": {
                    "gross_total_income": round(
                        float(true_income * np.random.uniform(0.95, 1.05)), 2
                    )
                }
            }
        }

        # Metadata
        aa_data["_metadata"] = {
            "total_emi_other_banks": round(float(total_emi_other_banks), 2),
            "num_banks": int(num_banks),
            "total_balance_all_banks": round(float(total_balance), 2),
            "has_fintech_loans": any(
                a["fip_id"] in self.fintech_lenders
                for a in aa_data["accounts"]
            )
        }

        return aa_data

    def generate_all_aa_data(self):
        """Generate AA data for all customers"""

        print("Generating Account Aggregator data for all customers...")
        aa_dataset = []

        for idx, row in self.customers.iterrows():
            if idx % 5000 == 0:
                print(f"  Processed {idx} customers...")

            aa_data = self.generate_aa_response(
                row['customer_id'],   # No int conversion
                row.to_dict()
            )
            aa_dataset.append(aa_data)

        print(f"\n✓ Generated AA data for {len(aa_dataset)} customers")
        return aa_dataset


# ---------------------------------------------------
# Main Execution
# ---------------------------------------------------
if __name__ == "__main__":
    df = pd.read_parquet('data/raw/synthetic_predelinquency_data.parquet')

    aa_sim = AccountAggregatorSimulator(df)
    aa_data = aa_sim.generate_all_aa_data()

    os.makedirs('data/raw', exist_ok=True)

    output_path = 'data/raw/aa_data.jsonl'
    with open(output_path, 'w') as f:
        for record in aa_data:
            f.write(json.dumps(record, default=json_serializer) + '\n')

    print(f"✓ Saved to: {output_path}")

    sample_path = 'data/raw/aa_sample.json'
    with open(sample_path, 'w') as f:
        json.dump(aa_data[0], f, indent=2, default=json_serializer)

    print(f"✓ Sample saved to: {sample_path}")
