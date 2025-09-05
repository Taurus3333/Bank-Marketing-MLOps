from __future__ import annotations

import os
from pathlib import Path

# bank_marketing/config.py (append if not present)

ROOT_DIR = Path(__file__).resolve().parents[1]
from dotenv import load_dotenv

# Load environment once. Keep this side-effect minimal and predictable.
load_dotenv()

# ROOTS
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MLFLOW_DIR = BASE_DIR / "mlruns"  # local default; can be overridden

# APP
APP_NAME = "BANK_MARKETING"
ENV = os.getenv("ENV", "sandbox")  # sandbox | prod
SEED = 42

# LOGGING
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / f"{APP_NAME.lower()}.log"

# MLOPS / TRACKING
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{MLFLOW_DIR.as_posix()}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "bank_marketing_default")

# AWS (used later)
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET", "bmkt-artifacts-dev")

# PYSPARK (used starting next chunk)
PYSPARK_APP_NAME = os.getenv("PYSPARK_APP_NAME", "bank_marketing_etl")

# DATA PATHS
DATA_DIR = (BASE_DIR / "data")
RAW_BANK_CSV = (DATA_DIR / "bank.csv")
CLEAN_DIR = (DATA_DIR / "clean")
CLEAN_BANK_PARQUET = (CLEAN_DIR / "bank.parquet")

# MLflow
MLFLOW_EXPERIMENT = "bank_marketing_prod"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{MLFLOW_DIR.as_posix()}")


# Inference
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT_DIR / "artifacts" / "train_now" / "model" / "catboost_model.cbm"))
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.50"))

# MLflow defaults (already present earlier)
MLFLOW_DIR = ROOT_DIR / "mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{MLFLOW_DIR.as_posix()}")
