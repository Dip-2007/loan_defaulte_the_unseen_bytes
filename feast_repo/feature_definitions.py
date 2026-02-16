# feast_repo/feature_definitions.py
"""
Feast Feature Store definitions for Pre-Delinquency Engine.
Defines entities, data sources, and feature views for online/offline serving.

This module provides feature definitions that can be registered with a Feast
feature store for consistent, versioned feature serving across training and
inference pipelines.
"""

from datetime import timedelta

try:
    from feast import Entity, Feature, FeatureView, FileSource, ValueType
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False


# ============================================================
# Data Sources
# ============================================================

if FEAST_AVAILABLE:
    # Enriched loan data (offline source)
    enriched_source = FileSource(
        path="data/processed/enriched_loan_data.parquet",
        event_timestamp_column="event_timestamp",
        created_timestamp_column="created_timestamp",
    )

    # Featured data (offline source)
    featured_source = FileSource(
        path="data/processed/featured_loan_data.parquet",
        event_timestamp_column="event_timestamp",
        created_timestamp_column="created_timestamp",
    )


    # ============================================================
    # Entities
    # ============================================================

    customer = Entity(
        name="customer_id",
        value_type=ValueType.STRING,
        description="Unique customer / loan ID",
    )


    # ============================================================
    # Feature Views
    # ============================================================

    # Core financial features
    core_financial_features = FeatureView(
        name="core_financial_features",
        entities=["customer_id"],
        ttl=timedelta(days=1),
        features=[
            Feature(name="Income", dtype=ValueType.FLOAT),
            Feature(name="LoanAmount", dtype=ValueType.FLOAT),
            Feature(name="CreditScore", dtype=ValueType.INT64),
            Feature(name="DTIRatio", dtype=ValueType.FLOAT),
            Feature(name="InterestRate", dtype=ValueType.FLOAT),
            Feature(name="LoanTerm", dtype=ValueType.INT64),
            Feature(name="avg_monthly_balance_6m", dtype=ValueType.FLOAT),
            Feature(name="total_monthly_expense", dtype=ValueType.FLOAT),
            Feature(name="savings_rate", dtype=ValueType.FLOAT),
        ],
        batch_source=enriched_source,
    )

    # Behavioral signal features
    behavioral_features = FeatureView(
        name="behavioral_features",
        entities=["customer_id"],
        ttl=timedelta(days=1),
        features=[
            Feature(name="ontime_payment_rate_12m", dtype=ValueType.FLOAT),
            Feature(name="payment_day_consistency", dtype=ValueType.FLOAT),
            Feature(name="max_dpd_last_12m", dtype=ValueType.INT64),
            Feature(name="subscription_cascade_phase", dtype=ValueType.INT64),
            Feature(name="p2p_borrow_requests_30d", dtype=ValueType.INT64),
            Feature(name="instant_cashouts_month", dtype=ValueType.INT64),
            Feature(name="salary_delay_days", dtype=ValueType.FLOAT),
        ],
        batch_source=enriched_source,
    )

    # Engineered formula features
    formula_features = FeatureView(
        name="formula_features",
        entities=["customer_id"],
        ttl=timedelta(days=1),
        features=[
            Feature(name="isi", dtype=ValueType.FLOAT),
            Feature(name="computed_dti", dtype=ValueType.FLOAT),
            Feature(name="savings_rate_pct", dtype=ValueType.FLOAT),
            Feature(name="eci", dtype=ValueType.FLOAT),
            Feature(name="evs", dtype=ValueType.FLOAT),
            Feature(name="evs_ratio", dtype=ValueType.FLOAT),
            Feature(name="sdt", dtype=ValueType.FLOAT),
            Feature(name="cdps", dtype=ValueType.FLOAT),
            Feature(name="dsr", dtype=ValueType.FLOAT),
            Feature(name="ptc", dtype=ValueType.FLOAT),
            Feature(name="abt", dtype=ValueType.FLOAT),
        ],
        batch_source=featured_source,
    )

    # Advanced behavioral features
    advanced_behavioral_features = FeatureView(
        name="advanced_behavioral_features",
        entities=["customer_id"],
        ttl=timedelta(days=1),
        features=[
            Feature(name="subscription_cascade_score", dtype=ValueType.FLOAT),
            Feature(name="wallet_velocity", dtype=ValueType.FLOAT),
            Feature(name="merchant_downgrade_score", dtype=ValueType.INT64),
            Feature(name="healthcare_spike_ratio", dtype=ValueType.FLOAT),
            Feature(name="employer_risk_multiplier", dtype=ValueType.FLOAT),
        ],
        batch_source=featured_source,
    )

    # Risk scoring output features
    risk_score_features = FeatureView(
        name="risk_score_features",
        entities=["customer_id"],
        ttl=timedelta(hours=6),  # Refreshed more frequently
        features=[
            Feature(name="risk_score_v2", dtype=ValueType.FLOAT),
            Feature(name="risk_band", dtype=ValueType.STRING),
        ],
        batch_source=FileSource(
            path="data/processed/scored_loan_data.parquet",
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        ),
    )
