import pandas as pd
import numpy as np
from faker import Faker
import datetime
import os

# Initialize Faker
fake = Faker()

DATA_PATH = r"c:\Users\yashg\OneDrive\Desktop\Hack_O_Hire_Hackathon\loan_defaulte_the_unseen_bytes\data\raw\synthetic_predelinquency_data.parquet"

def load_data():
    """Loads the main customer dataset."""
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
    else:
        # Fallback if file doesn't exist (for testing)
        print(f"Warning: Data file not found at {DATA_PATH}. Creating mock data.")
        df = pd.DataFrame({
            'customer_id': [f'CUST_{i}' for i in range(100)],
            'risk_score': np.random.uniform(0, 100, 100),
            'monthly_income': np.random.uniform(3000, 10000, 100),
            'name': [fake.name() for _ in range(100)]
        })
    
    # Add mock names if not present (the parquet file seemed to lack names)
    if 'name' not in df.columns:
        df['name'] = [fake.name() for _ in range(len(df))]
        
    # Ensure risk_score exists (mocking it if not present based on target)
    if 'risk_score' not in df.columns:
        # Assuming 'target' or similar exists, or just random for now
         df['risk_score'] = np.random.randint(300, 850, size=len(df)) # Credit score like
         df['delinquency_probability'] = np.random.uniform(0, 1, size=len(df))

    # Calculate Risk Category once at load time
    if 'risk_category' not in df.columns or df['risk_category'].dtype == 'object':
         # Ensure risk_score is valid
         if 'risk_score' not in df.columns:
             df['risk_score'] = np.random.randint(300, 850, size=len(df))
         else:
             df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
             
         # Fill any remaining NaNs (from coercion)
         if df['risk_score'].isnull().any():
             df.loc[df['risk_score'].isnull(), 'risk_score'] = np.random.randint(300, 850, size=df['risk_score'].isnull().sum())
             
         df['risk_category'] = pd.cut(df['risk_score'], bins=[-1, 300, 600, 1000], labels=['Low', 'Medium', 'High'])
         df['risk_category'] = df['risk_category'].astype(str)

    # Add Employment Type
    if 'employment_type' not in df.columns:
        employment_categories = ['Salaried', 'Business', 'Retired', 'Student', 'Unemployed', 'Freelancer']
        # Weighted probabilities to be realistic
        probs = [0.45, 0.20, 0.15, 0.05, 0.05, 0.10]
        df['employment_type'] = np.random.choice(employment_categories, size=len(df), p=probs)

    return df.copy()

def get_customer_timeline(customer_id):
    """Generates synthetic timeline data for a specific customer."""
    # Generate dates for the last 6 months
    dates = pd.date_range(end=datetime.date.today(), periods=180).tolist()
    
    # Base trend
    base_balance = np.random.uniform(1000, 50000)
    
    # create some random fluctuations
    balance_history = []
    current_balance = base_balance
    
    for _ in dates:
        change = np.random.uniform(-500, 600)
        current_balance += change
        if current_balance < 0: current_balance = 0
        balance_history.append(current_balance)
        
    return pd.DataFrame({
        'date': dates,
        'balance': balance_history
    })

def get_spending_breakdown(customer_id):
    """Generates synthetic spending breakdown."""
    categories = ['Rent/Mortgage', 'Groceries', 'Utilities', 'Dining Out', 'Entertainment', 'Shopping']
    amounts = np.random.randint(50, 2000, size=len(categories))
    return pd.DataFrame({
        'category': categories,
        'amount': amounts
    })
