# src/data_generation/synthetic_data_generator.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker('en_IN')  # Indian locale
np.random.seed(42)

class PreDelinquencyDataGenerator:
    """
    Generate synthetic pre-delinquency data with behavioral signals
    """
    
    def __init__(self, n_customers=50000):
        self.n_customers = n_customers
        
    def generate_customer_profile(self, customer_id):
        """Generate base customer profile"""
        
        # Customer demographics
        age = np.random.randint(21, 65)
        gender = np.random.choice(['M', 'F'], p=[0.6, 0.4])
        
        # Employment type
        employment_type = np.random.choice(
            ['SALARIED_PRIVATE', 'SALARIED_GOVT', 'SELF_EMPLOYED', 
             'BUSINESS_OWNER', 'GIG_WORKER', 'RETIRED', 'STUDENT'],
            p=[0.35, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05]
        )
        
        # Income based on employment type
        income_ranges = {
            'SALARIED_PRIVATE': (60000, 25000),
            'SALARIED_GOVT': (55000, 15000),
            'SELF_EMPLOYED': (50000, 30000),
            'BUSINESS_OWNER': (80000, 50000),
            'GIG_WORKER': (35000, 15000),
            'RETIRED': (30000, 10000),
            'STUDENT': (15000, 5000)
        }
        
        mean, std = income_ranges[employment_type]
        income = max(10000, np.random.normal(mean, std))
        
        # Loan details
        loan_amount = np.random.uniform(100000, 5000000)
        loan_tenure = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240])
        interest_rate = np.random.uniform(8.5, 18.5)
        
        # Calculate EMI
        r = interest_rate / 12 / 100
        n = loan_tenure
        emi = loan_amount * r * (1 + r)**n / ((1 + r)**n - 1)
        
        # DTI ratio
        dti = (emi / income) * 100
        
        # Credit score
        credit_score = int(np.clip(np.random.normal(720, 80), 300, 900))
        
        return {
            'customer_id': f'CUST_{customer_id:06d}',
            'age': age,
            'gender': gender,
            'employment_type': employment_type,
            'monthly_income': round(income, 2),
            'loan_amount': round(loan_amount, 2),
            'loan_tenure_months': loan_tenure,
            'interest_rate': round(interest_rate, 2),
            'emi': round(emi, 2),
            'dti_ratio': round(dti, 2),
            'credit_score': credit_score
        }
    
    def generate_behavioral_features(self, profile):
        """Generate features first, then probabilistic target (Causal approach)"""
        
        # 1. Generate Independent Behavioral Features (with some noise)
        
        # Digital Engagement
        app_login_duration_sec = np.random.lognormal(mean=3.5, sigma=0.8) # ~33 sec mean
        notification_dismissal_speed_sec = np.random.lognormal(mean=1.5, sigma=0.6) # ~4.5 sec mean
        statement_open_rate = np.clip(np.random.beta(a=2, b=2), 0, 1) # Uniform-ish
        password_resets_30d = np.random.poisson(lam=0.5)
        
        # Financial Stress Indicators
        discretionary_spending_change_pct = np.random.normal(loc=0, scale=20)
        param_cashouts = 0.5 if profile['credit_score'] > 700 else 2.0
        instant_cashouts_month = np.random.poisson(lam=param_cashouts)
        
        # Social/Network
        param_p2p = 0.2 if profile['monthly_income'] > 50000 else 1.5
        p2p_borrow_requests = np.random.poisson(lam=param_p2p)
        
        # Employment/Income Stability
        salary_delay_days = 0
        if profile['employment_type'].startswith('SALARIED'):
            if np.random.random() < 0.1: # 10% chance of delay
                salary_delay_days = np.random.randint(3, 15)
        
        # Account Health
        balance_trajectory_pct = np.random.normal(loc=2, scale=15)
        merchant_downgrade_score = 0
        if np.random.random() < 0.2:
            merchant_downgrade_score = -1 * np.random.randint(1, 10)
            
        # Medical
        healthcare_spending_multiplier = 1.0
        if np.random.random() < 0.05: # 5% medical emergency
            healthcare_spending_multiplier = np.random.choice([2.5, 5.0])
            
        # Segment specific
        cash_deposit_pattern_score = 0
        if profile['employment_type'] == 'BUSINESS_OWNER':
            cash_deposit_pattern_score = np.random.normal(loc=70, scale=20)
            
        employer_health_score = 85
        if profile['employment_type'].startswith('SALARIED'):
            employer_health_score = np.random.normal(loc=75, scale=15)
            
        subscription_cascade_phase = np.random.choice([1, 2, 3, 4], p=[0.7, 0.15, 0.1, 0.05])

        # 2. Calculate Risk Score (Logit)
        # Higher score = Higher probability of default
        
        risk_score = -4.0 # Base logit (intercept) -> low default prob
        
        # Add risk factors
        risk_score += 0.02 * (profile['dti_ratio'] - 40) # Higher DTI -> Higher risk
        risk_score -= 0.005 * (profile['credit_score'] - 650) # Higher Credit Score -> Lower risk
        
        # Behavioral impacts
        if app_login_duration_sec < 10: risk_score += 0.5 # Quick logins check anxiety
        if notification_dismissal_speed_sec < 2: risk_score += 0.4 # Ignoring fast
        if statement_open_rate < 0.2: risk_score += 0.6 # Ignoring statements
        risk_score += 0.3 * password_resets_30d # Frantic resets
        risk_score += 0.5 * (subscription_cascade_phase - 1)
        
        if discretionary_spending_change_pct < -20: risk_score += 0.8 # Sudden drop in spending
        risk_score += 0.2 * instant_cashouts_month
        risk_score += 0.3 * p2p_borrow_requests
        
        if salary_delay_days > 5: risk_score += 1.0
        if balance_trajectory_pct < -10: risk_score += 1.2
        if merchant_downgrade_score < -3: risk_score += 0.7
        if healthcare_spending_multiplier > 2: risk_score += 1.5
        
        # Add random noise to make it imperfect
        risk_score += np.random.normal(loc=0, scale=0.5)
        
        # 3. Convert to Probability (Sigmoid)
        default_prob = 1 / (1 + np.exp(-risk_score))
        
        # 4. Assign Target
        target = 1 if np.random.random() < default_prob else 0
        days_to_default = np.random.randint(7, 45) if target == 1 else None

        behavioral = {
            'app_login_duration_sec': round(app_login_duration_sec, 2),
            'notification_dismissal_speed_sec': round(notification_dismissal_speed_sec, 2),
            'statement_open_rate': round(statement_open_rate, 3),
            'password_resets_30d': int(password_resets_30d),
            'subscription_cascade_phase': int(subscription_cascade_phase),
            'discretionary_spending_change_pct': round(discretionary_spending_change_pct, 2),
            'instant_cashouts_month': int(instant_cashouts_month),
            'p2p_borrow_requests': int(p2p_borrow_requests),
            'salary_delay_days': int(salary_delay_days),
            'balance_trajectory_pct': round(balance_trajectory_pct, 2),
            'merchant_downgrade_score': int(merchant_downgrade_score),
            'healthcare_spending_multiplier': round(healthcare_spending_multiplier, 2),
            'cash_deposit_pattern_score': round(max(0, cash_deposit_pattern_score), 2),
            'employer_health_score': round(max(0, employer_health_score), 2),
            'target': target,
            'days_to_default': days_to_default
        }
        
        return behavioral
    
    def generate_dataset(self):
        """Generate complete dataset"""
        
        print(f"Generating {self.n_customers} synthetic customers...")
        
        data = []
        for i in range(self.n_customers):
            if i % 5000 == 0:
                print(f"  Generated {i} customers...")
            
            profile = self.generate_customer_profile(i)
            behavioral = self.generate_behavioral_features(profile)
            
            customer = {**profile, **behavioral}
            data.append(customer)
        
        df = pd.DataFrame(data)
        
        print(f"\n✓ Dataset created: {len(df)} rows, {len(df.columns)} columns")
        print(f"✓ Default rate: {df['target'].mean()*100:.2f}%")
        
        return df

# Main execution
if __name__ == "__main__":
    # Create data directory if not exists
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate dataset
    generator = PreDelinquencyDataGenerator(n_customers=50000)
    df = generator.generate_dataset()
    
    # Save to parquet
    output_path = 'data/raw/synthetic_predelinquency_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Save sample to CSV for inspection
    sample_path = 'data/raw/sample_data.csv'
    df.head(1000).to_csv(sample_path, index=False)
    print(f"✓ Sample saved to: {sample_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total customers: {len(df):,}")
    print(f"Will default: {(df['target']==1).sum():,} ({df['target'].mean()*100:.1f}%)")
    print(f"Won't default: {(df['target']==0).sum():,} ({(1-df['target'].mean())*100:.1f}%)")
    print(f"\nFeatures: {len(df.columns)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")