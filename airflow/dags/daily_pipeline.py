# airflow/dags/daily_pipeline.py
"""
Airflow DAG: Daily Pre-Delinquency Detection Pipeline (Section 6)
Orchestrates the 7-step workflow on a daily schedule.

Steps:
  1. Data Ingestion (01:00 AM)
  2. Preprocessing & Cleaning (02:00 AM)
  3. Feature Engineering (02:30 AM)
  4. Model Inference (03:00 AM)
  5. Risk Stratification (04:00 AM)
  6. Intervention Triggering (05:00 AM)
  7. Monitoring & Feedback (continuous)
"""

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

import sys
import os

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


# ============================================================
# Task Functions
# ============================================================

def step_1_ingest(**kwargs):
    """Load raw data from source systems."""
    from workflow.workflow_engine import WorkflowEngine
    engine = WorkflowEngine()
    df = engine.step_1_ingest()
    kwargs['ti'].xcom_push(key='row_count', value=len(df))
    return f"Ingested {len(df):,} rows"


def step_2_preprocess(**kwargs):
    """Clean and validate data."""
    from workflow.workflow_engine import WorkflowEngine
    engine = WorkflowEngine()
    df = engine.step_1_ingest()
    df = engine.step_2_preprocess(df)
    return f"Preprocessed {len(df):,} rows, {len(df.columns)} columns"


def step_3_features(**kwargs):
    """Run feature engineering pipeline (300+ features)."""
    from workflow.workflow_engine import WorkflowEngine
    engine = WorkflowEngine()
    df = engine.step_1_ingest()
    df = engine.step_2_preprocess(df)
    df = engine.step_3_feature_engineering(df)
    return f"Engineered {len(df.columns)} features"


def step_4_inference(**kwargs):
    """Run ensemble model inference."""
    from workflow.workflow_engine import WorkflowEngine
    engine = WorkflowEngine()
    df = engine.step_1_ingest()
    df = engine.step_2_preprocess(df)
    df = engine.step_3_feature_engineering(df)
    df = engine.step_4_inference(df)
    return f"Inference complete on {len(df):,} customers"


def step_5_stratify(**kwargs):
    """Risk stratification into 5 bands."""
    from workflow.workflow_engine import WorkflowEngine
    engine = WorkflowEngine()
    df = engine.step_1_ingest()
    df = engine.step_2_preprocess(df)
    df = engine.step_3_feature_engineering(df)
    df = engine.step_4_inference(df)
    df = engine.step_5_stratify(df)
    dist = df['risk_band'].value_counts().to_dict() if 'risk_band' in df.columns else {}
    return f"Stratified: {dist}"


def step_6_intervene(**kwargs):
    """Trigger interventions based on risk bands."""
    from workflow.workflow_engine import WorkflowEngine
    engine = WorkflowEngine()
    df, interventions = engine.run()
    return f"Generated {len(interventions)} interventions"


def step_7_monitor(**kwargs):
    """Log metrics & performance monitoring."""
    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'pipeline_version': '2.0.0',
        'status': 'completed',
    }
    return f"Monitoring: {metrics}"


# ============================================================
# DAG Definition
# ============================================================

default_args = {
    'owner': 'pre-delinquency-engine',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
}

if AIRFLOW_AVAILABLE:
    dag = DAG(
        dag_id='predelinquency_daily_pipeline',
        default_args=default_args,
        description='Daily pre-delinquency detection workflow (Section 6)',
        schedule_interval='0 1 * * *',  # 1:00 AM daily
        catchup=False,
        max_active_runs=1,
        tags=['ml', 'risk', 'pre-delinquency'],
    )

    t1 = PythonOperator(task_id='step_1_ingest',    python_callable=step_1_ingest,    dag=dag)
    t2 = PythonOperator(task_id='step_2_preprocess', python_callable=step_2_preprocess, dag=dag)
    t3 = PythonOperator(task_id='step_3_features',   python_callable=step_3_features,   dag=dag)
    t4 = PythonOperator(task_id='step_4_inference',  python_callable=step_4_inference,  dag=dag)
    t5 = PythonOperator(task_id='step_5_stratify',   python_callable=step_5_stratify,   dag=dag)
    t6 = PythonOperator(task_id='step_6_intervene',  python_callable=step_6_intervene,  dag=dag)
    t7 = PythonOperator(task_id='step_7_monitor',    python_callable=step_7_monitor,    dag=dag)

    # Linear dependency chain
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
