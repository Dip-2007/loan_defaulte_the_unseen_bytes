# src/data_generation/download_dataset.py
"""Download the Loan Default Prediction dataset from Kaggle"""

import kagglehub
import shutil
import os

def download_loan_default_data():
    """Download Loan_default.csv from Kaggle"""
    
    print("Downloading Loan Default Prediction dataset from Kaggle...")
    print("(This may prompt for Kaggle credentials on first run)\n")
    
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("nikhil1e9/loan-default")
    
    print(f"\n✓ Dataset downloaded to: {path}")
    
    # Copy to our data/raw directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Find CSV files in the downloaded path
    for f in os.listdir(path):
        if f.endswith('.csv'):
            src = os.path.join(path, f)
            dst = os.path.join('data', 'raw', f)
            shutil.copy2(src, dst)
            file_size = os.path.getsize(dst) / (1024 * 1024)
            print(f"✓ Copied {f} ({file_size:.1f} MB) -> {dst}")
    
    return path

if __name__ == "__main__":
    download_loan_default_data()
