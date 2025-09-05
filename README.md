🏦 Bank Marketing MLOps Project

📌 Overview

This project implements a lean, production-grade MLOps system for predicting whether a bank client will subscribe to a term deposit based on direct marketing campaigns (phone calls).

The system is end-to-end:
ETL → Experimentation → Training → Deployment → Orchestration → Monitoring → Automated Retraining.

📊 Dataset

Source: Portuguese Bank Marketing dataset (Moro et al., 2011).

Records: 45,211

Features: 16 input variables + binary target (y).

Target imbalance: Only ~11.7% positive (subscribed).

Challenges: severe imbalance, categorical complexity, many “unknowns”, leakage risk (duration feature).

🛠️ Tech Stack

Experimentation: Jupyter, pandas, scikit-learn, CatBoost

Orchestration: Apache Airflow (Docker)

Tracking & Registry: MLflow

Deployment: FastAPI (API), Streamlit (sandbox UI)

Containerization: Docker

Infra (AWS): S3, ECS/Fargate, CloudWatch, SNS

CI/CD: GitHub Actions, SonarQube

Testing & Quality: pytest, pre-commit

Key Features

ETL: PySpark-based schema enforcement & Parquet storage.

Experimentation: Full EDA, imbalance handling, model comparison → CatBoost selected.

Pipeline: Modular phases with MLflow artifact logging.

Deployment: FastAPI (machine APIs) + Streamlit (business UI), Dockerized and deployed to ECS.

Orchestration: Airflow DAGs for training, scoring, and monitoring.

Monitoring: Drift detection (PSI, KS-test), alerts via SNS, automated retrain triggers.

CI/CD: GitHub Actions pipeline for test → scan → build → deploy → DAG sync.

Security: IAM least privilege, S3 SSE-KMS encryption, CloudTrail audit logging.

📈 Results

Champion Model: XGBoost, CatBoost

Performance (CV AUC ~0.94, PR AUC ~0.64)

Impact:

Improved targeting of term deposit subscribers.

Reduced wasted calls → lower operational cost.

Higher conversion → increased deposits.

Business Value:
A repeatable, auditable, scalable ML system instead of one-off models.

🧪 How to Run
1. Local Dev Setup
# create venv
python -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt

2. Run ETL
python -m bank_marketing.etl.bank_etl \
  --input data/bank.csv \
  --output data/clean/bank.parquet

3. Train Pipeline
python -m bank_marketing.pipelines.train_pipeline \
  --input data/clean/bank.parquet \
  --output artifacts/train_run \
  --run-name "catboost_default"

4. Run FastAPI Service
uvicorn bank_marketing.services.api:app --reload --port 8000


Check docs: http://localhost:8000/docs

5. Run Streamlit App
   
streamlit run app_streamlit.py

This concludes this End-to-end MLOps project implemented as a lean, production-grade system with strong business impact. Suitable for interviews, portfolio showcase, and production deployments.
