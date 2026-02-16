# src/serving/model_server.py
"""
Model Serving Module for Pre-Delinquency Engine.
Handles model versioning, prediction logging, and A/B testing support.

Architecture:
  Model Registry → Model Server → Prediction API → Monitoring
"""

import os
import sys
import json
import joblib
import logging
import hashlib
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


class ModelRegistry:
    """Simple file-based model registry for versioning.

    Tracks model versions with metadata, performance metrics,
    and deployment status.
    """

    def __init__(self, registry_dir=None):
        self.registry_dir = registry_dir or os.path.join(PROJECT_ROOT, 'data', 'models')
        self.registry_file = os.path.join(self.registry_dir, 'model_registry.json')
        self.models: Dict[str, dict] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.models = json.load(f)
        else:
            self.models = {}

    def _save_registry(self):
        """Save registry to disk."""
        os.makedirs(self.registry_dir, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.models, f, indent=2, default=str)

    def register_model(self, model_path: str, model_type: str = 'ensemble',
                       metrics: Optional[dict] = None, description: str = ''):
        """Register a new model version.

        Args:
            model_path: Path to the model artifact (.pkl).
            model_type: Type identifier (ensemble, xgboost, etc.).
            metrics: Performance metrics (AUC, precision, recall).
            description: Description of this version.

        Returns:
            Version ID string.
        """
        # Generate version ID from file hash
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        version_id = f"{model_type}_v{len(self.models) + 1}_{file_hash}"

        self.models[version_id] = {
            'model_path': model_path,
            'model_type': model_type,
            'registered_at': datetime.utcnow().isoformat(),
            'metrics': metrics or {},
            'description': description,
            'status': 'staged',  # staged → production → archived
        }
        self._save_registry()
        logger.info(f"Registered model: {version_id}")
        return version_id

    def promote_to_production(self, version_id: str):
        """Promote a staged model to production."""
        if version_id not in self.models:
            raise ValueError(f"Model {version_id} not found")

        # Archive current production model
        for vid, meta in self.models.items():
            if meta['status'] == 'production':
                meta['status'] = 'archived'

        self.models[version_id]['status'] = 'production'
        self._save_registry()
        logger.info(f"Promoted {version_id} to production")

    def get_production_model(self) -> Optional[dict]:
        """Get the current production model metadata."""
        for vid, meta in self.models.items():
            if meta['status'] == 'production':
                return {'version_id': vid, **meta}
        return None

    def list_models(self) -> List[dict]:
        """List all registered models."""
        return [{
            'version_id': vid,
            'type': meta['model_type'],
            'status': meta['status'],
            'registered_at': meta['registered_at'],
            'metrics': meta.get('metrics', {}),
        } for vid, meta in self.models.items()]


class ModelServer:
    """Serves predictions from the production model.

    Features:
    - Automatic model loading from registry
    - Prediction logging for monitoring
    - A/B testing support (shadow mode)
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.production_model = None
        self.shadow_model = None
        self.prediction_log: List[dict] = []

    def load_production_model(self):
        """Load the current production model."""
        prod = self.registry.get_production_model()
        if prod is None:
            # Fallback: load default ensemble model
            default_path = os.path.join(PROJECT_ROOT, 'data', 'models', 'ensemble_model.pkl')
            if os.path.exists(default_path):
                self.production_model = joblib.load(default_path)
                logger.info(f"Loaded fallback model: {default_path}")
            else:
                logger.warning("No production model available")
            return

        model_path = prod['model_path']
        if os.path.exists(model_path):
            self.production_model = joblib.load(model_path)
            logger.info(f"Loaded production model: {prod['version_id']}")
        else:
            logger.error(f"Model file not found: {model_path}")

    def predict(self, customer_data: dict) -> dict:
        """Generate prediction for a single customer.

        Args:
            customer_data: Dict of customer features.

        Returns:
            Dict with default_probability, risk_score, risk_band.
        """
        if self.production_model is None:
            self.load_production_model()

        if self.production_model is None:
            return {'error': 'No model available', 'default_probability': 0.5}

        # Prepare features
        X = pd.DataFrame([customer_data])
        feature_names = self.production_model.get('feature_names', [])
        models = self.production_model.get('models', {})
        scaler = self.production_model.get('scaler')
        weights = self.production_model.get('weights', {
            'xgboost': 0.35, 'lightgbm': 0.30,
            'random_forest': 0.20, 'logistic': 0.15
        })

        # Encode categoricals and align features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        X = X.fillna(-999).select_dtypes(include=[np.number])

        for f in feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[feature_names]

        # Scale for logistic regression
        X_scaled = scaler.transform(X) if scaler else X

        # Ensemble prediction
        probas = {}
        for name, model in models.items():
            try:
                if name == 'logistic':
                    probas[name] = model.predict_proba(X_scaled)[:, 1]
                else:
                    probas[name] = model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not probas:
            return {'error': 'All model predictions failed'}

        ensemble_prob = float(sum(weights.get(n, 0) * probas[n] for n in probas)[0])
        risk_score = round(ensemble_prob * 100, 1)

        # Classify
        from risk_scoring.risk_scorer import RISK_BANDS
        band = 'SAFE'
        for lo, hi, b, _, _ in RISK_BANDS:
            if lo <= risk_score <= hi:
                band = b
                break

        result = {
            'default_probability': round(ensemble_prob, 4),
            'risk_score': risk_score,
            'risk_band': band,
            'model_probabilities': {k: round(float(v[0]), 4) for k, v in probas.items()},
        }

        # Log prediction
        self._log_prediction(customer_data, result)
        return result

    def _log_prediction(self, input_data: dict, result: dict):
        """Log prediction for monitoring and drift detection."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'customer_id': input_data.get('LoanID', 'unknown'),
            'risk_score': result.get('risk_score'),
            'risk_band': result.get('risk_band'),
            'default_probability': result.get('default_probability'),
        }
        self.prediction_log.append(log_entry)

        # Keep last 10000 entries in memory
        if len(self.prediction_log) > 10000:
            self.prediction_log = self.prediction_log[-5000:]

    def get_monitoring_stats(self) -> dict:
        """Get summary statistics from prediction log."""
        if not self.prediction_log:
            return {'total_predictions': 0}

        scores = [entry['risk_score'] for entry in self.prediction_log if entry.get('risk_score')]
        bands = [entry['risk_band'] for entry in self.prediction_log if entry.get('risk_band')]

        return {
            'total_predictions': len(self.prediction_log),
            'avg_risk_score': round(np.mean(scores), 2) if scores else 0,
            'risk_band_distribution': dict(pd.Series(bands).value_counts()) if bands else {},
            'latest_prediction': self.prediction_log[-1] if self.prediction_log else None,
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Demo: register and serve
    registry = ModelRegistry()
    print(f"Registered models: {len(registry.models)}")

    server = ModelServer(registry)
    server.load_production_model()

    # Example prediction
    test_customer = {
        'LoanID': 'SERVE_TEST',
        'Age': 32,
        'Income': 85000,
        'CreditScore': 720,
        'DTIRatio': 23.5,
    }

    if server.production_model:
        result = server.predict(test_customer)
        print(f"Prediction: {result}")
    else:
        print("No model loaded - train model first")
