📦 loan_default_predictor
 ┣ 🌐 Interfaces & Serving
 ┃ ┣ 📂 src/api/            # Real-time REST API for model inference
 ┃ ┣ 📂 src/dashboard/      # Interactive UI for Risk Analytics & Business Logic
 ┃ ┗ 📂 src/serving/        # Model server handling live prediction requests
 ┃
 ┣ 🧠 Core ML & Explainable AI (XAI)
 ┃ ┣ 📂 src/models/         # XGBoost & Ensemble Model implementations
 ┃ ┃ ┗ 📜 shap_explainer.py # 💡 Highlighting Model Interpretability for Compliance
 ┃ ┗ 📂 src/risk_scoring/   # Custom business logic translating ML to Risk Scores
 ┃
 ┣ ⚙️ Data Engineering & Feature Store
 ┃ ┣ 📂 feast_repo/         # Enterprise Feature Store (Feast) integration
 ┃ ┣ 📂 src/feature_eng/    # Scalable feature transformation pipelines
 ┃ ┗ 📂 src/data_gen/       # Synthetic data simulators & enrichment
 ┃
 ┣ 🔄 Orchestration & Streaming
 ┃ ┣ 📂 airflow/dags/       # Automated daily batch processing (DAGs)
 ┃ ┣ 📂 src/streaming/      # Live streaming consumer for real-time events
 ┃ ┗ 📂 src/workflow/       # Core workflow engine bridging components
 ┃
 ┗ 🐳 DevOps, Testing & MLOps
   ┣ 📂 docker/             # Fully containerized for instant deployment
   ┣ 📂 tests/              # Comprehensive Unit/Integration tests (CI/CD ready)
   ┗ 📜 docker-compose.yml  # One-click local infrastructure spin-up
