# src/api/app.py
"""
FastAPI Prediction Service for Pre-Delinquency Engine.
Endpoints: /predict, /risk-score, /explain, /workflow/run
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from risk_scoring.risk_scorer import RiskScorer

app = FastAPI(
    title="Pre-Delinquency Detection API",
    description="Predict EMI/Credit Card payment defaults 2-3 weeks in advance",
    version="2.0.0"
)

# Load models at startup
scorer = RiskScorer()
ensemble_model = None
shap_explainer = None
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ENSEMBLE_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'ensemble_model.pkl')


@app.on_event("startup")
def load_models():
    global ensemble_model, shap_explainer
    if os.path.exists(ENSEMBLE_PATH):
        ensemble_model = joblib.load(ENSEMBLE_PATH)
        print(f"✓ Ensemble model loaded ({len(ensemble_model['feature_names'])} features)")

        # Load SHAP explainer
        try:
            from models.shap_explainer import SHAPExplainer
            shap_explainer = SHAPExplainer(ENSEMBLE_PATH)
            print("✓ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠ SHAP explainer failed: {e}")
    else:
        print(f"⚠ Ensemble model not found at {ENSEMBLE_PATH}")


# ============================================================
# Request/Response Models
# ============================================================

class CustomerData(BaseModel):
    """Input: customer data for prediction."""
    LoanID: str
    Age: int = 35
    Income: float = 50000
    LoanAmount: float = 500000
    CreditScore: int = 700
    MonthsEmployed: int = 36
    NumCreditLines: int = 3
    InterestRate: float = 12.0
    LoanTerm: int = 36
    DTIRatio: float = 30.0
    HasCoSigner: int = 0
    segment_category: str = "EMPLOYED"
    ontime_payment_rate_12m: float = 0.95
    avg_delay_days: float = 0
    avg_monthly_balance_6m: float = 30000
    total_outstanding_debt: float = 300000
    credit_utilization_ratio: float = 40
    num_dpd_30_plus: int = 0


class RiskResponse(BaseModel):
    """Output: risk score + band + components."""
    loan_id: str
    risk_score: float
    risk_band: str
    default_probability: str
    recommended_action: str
    component_scores: Dict[str, float]
    top_risk_factors: List[str]


class PredictionResponse(BaseModel):
    """Output: ensemble prediction."""
    loan_id: str
    default_probability: float
    risk_score_0_to_100: float
    risk_band: str


class ExplainResponse(BaseModel):
    """Output: SHAP explanation for a customer."""
    loan_id: str
    base_value: float
    risk_probability: float
    risk_score_0_to_100: float
    top_risk_up: List[Dict]
    top_risk_down: List[Dict]


class WorkflowResponse(BaseModel):
    """Output: workflow run summary."""
    total_customers: int
    risk_distribution: Dict[str, int]
    interventions_generated: int
    projected_prevented_defaults: int
    projected_roi_percent: float


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": ensemble_model is not None,
        "shap_available": shap_explainer is not None,
        "version": "2.0.0"
    }


@app.post("/risk-score", response_model=RiskResponse)
def get_risk_score(customer: CustomerData):
    """Get 15-component risk score for a customer."""
    try:
        row = pd.Series(customer.dict())
        score, component_scores = scorer.score_customer(row)
        band, prob, action = scorer.classify_risk(score)

        # Top risk factors
        sorted_components = sorted(component_scores.items(),
                                   key=lambda x: x[1], reverse=True)
        top_factors = [f"{name}: {s:.1f}" for name, s in sorted_components[:5]]

        return RiskResponse(
            loan_id=customer.LoanID,
            risk_score=round(score, 1),
            risk_band=band,
            default_probability=prob,
            recommended_action=action,
            component_scores={k: round(v, 2) for k, v in component_scores.items()},
            top_risk_factors=top_factors
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Get ensemble model prediction."""
    if ensemble_model is None:
        raise HTTPException(status_code=503,
                            detail="Ensemble model not loaded")
    try:
        row_dict = customer.dict()
        X = pd.DataFrame([row_dict])

        feature_names = ensemble_model['feature_names']
        models = ensemble_model['models']
        scaler = ensemble_model['scaler']
        weights = ensemble_model.get('weights', {
            'xgboost': 0.35, 'lightgbm': 0.30,
            'random_forest': 0.20, 'logistic': 0.15
        })

        # Prepare features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        X = X.fillna(-999).select_dtypes(include=[np.number])

        # Align to model features
        for f in feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[feature_names]
        X_scaled = scaler.transform(X)

        # Ensemble probability
        probas = {}
        for name, model in models.items():
            if name == 'logistic':
                probas[name] = model.predict_proba(X_scaled)[:, 1]
            else:
                probas[name] = model.predict_proba(X)[:, 1]

        ensemble_prob = float(sum(weights[n] * probas[n] for n in probas)[0])
        risk_score = round(ensemble_prob * 100, 1)

        # Classify
        from risk_scoring.risk_scorer import RISK_BANDS
        band = 'SAFE'
        for lo, hi, b, _, _ in RISK_BANDS:
            if lo <= risk_score <= hi:
                band = b
                break

        return PredictionResponse(
            loan_id=customer.LoanID,
            default_probability=round(ensemble_prob, 4),
            risk_score_0_to_100=risk_score,
            risk_band=band
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain(customer: CustomerData):
    """Get SHAP explanation for a customer's risk prediction."""
    if shap_explainer is None:
        raise HTTPException(status_code=503,
                            detail="SHAP explainer not loaded")
    try:
        row_dict = customer.dict()
        result = shap_explainer.explain_customer(row_dict)

        return ExplainResponse(
            loan_id=customer.LoanID,
            base_value=result['base_value'],
            risk_probability=result['risk_probability'],
            risk_score_0_to_100=result['risk_score_0_to_100'],
            top_risk_up=result['top_risk_up'],
            top_risk_down=result['top_risk_down']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/run", response_model=WorkflowResponse)
def run_workflow():
    """Run full 7-step workflow on the dataset."""
    try:
        from workflow.workflow_engine import WorkflowEngine
        engine = WorkflowEngine()
        df, interventions = engine.run()

        # Count risk bands
        risk_dist = {}
        if 'risk_band' in df.columns:
            risk_dist = df['risk_band'].value_counts().to_dict()
        elif 'risk_category' in df.columns:
            risk_dist = df['risk_category'].value_counts().to_dict()

        n_interventions = len(interventions)
        prevented = int(n_interventions * 0.70)  # 70% success rate estimate
        cost_intervention = n_interventions * 50  # ₹50 per intervention
        cost_without = n_interventions * 8000     # ₹8000 per default
        roi = ((cost_without - cost_intervention) / max(cost_intervention, 1)) * 100

        return WorkflowResponse(
            total_customers=len(df),
            risk_distribution=risk_dist,
            interventions_generated=n_interventions,
            projected_prevented_defaults=prevented,
            projected_roi_percent=round(roi, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
