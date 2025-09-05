# Paste this block into bank_marketing/pipelines/trainer.py (replace previous content)
from __future__ import annotations
from pathlib import Path
import json
import mlflow
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool # type: ignore
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
recall_score, f1_score, average_precision_score, confusion_matrix)
from ..core.logging import get_logger
from .transform import feature_engineer
from .data_validation import validate_schema, simple_validation_report


log = get_logger(__name__)

# Production defaults (from CHUNK 3)
CATBOOST_DEFAULT = dict(
iterations=500,
depth=6,
learning_rate=0.05,
l2_leaf_reg=5,
bagging_temperature=0.5,
border_count=128,
random_seed=42,
loss_function="Logloss",
eval_metric="AUC",
verbose=False,
allow_writing_files=False,
)

def _compute_class_weights(y: pd.Series) -> list:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    if pos == 0:
        return [1.0, 1.0]
    return [1.0, float(neg / pos)]

def train_and_log(input_parquet: str, run_name: str, output_dir: str) -> dict:
    """
    Full training flow:
    - load parquet
    - validate schema
    - transform features
    - split train/test (stratified)
    - compute class weights
    - train CatBoost
    - evaluate
    - log everything to MLflow and save model
    """
    mlflow.set_experiment("bank_marketing_prod")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("input_parquet", input_parquet)
        df = pd.read_parquet(input_parquet)

    validate_schema(df)
    report = simple_validation_report(df, Path(output_dir) / "validation_report.json")
    mlflow.log_dict(report, "validation_report.json")


    df = feature_engineer(df)
    # split
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["y"])
    y = df["y"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.log_param("train_rows", int(len(X_train)))
    mlflow.log_param("test_rows", int(len(X_test)))


    # compute class weights
    class_weights = _compute_class_weights(y_train)
    mlflow.log_param("class_weights", class_weights)


    # Prepare CatBoost Pool (CatBoost natively handles categorical features)
    cat_feature_names = ["job","marital","education","contact","month","poutcome",
    "age_group","pdays_group","previous_group"]
    cat_indices = [X_train.columns.get_loc(c) for c in cat_feature_names if c in X_train.columns]


    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)

    # update default with weights
    params = dict(CATBOOST_DEFAULT)
    params["class_weights"] = class_weights


    mlflow.log_params({f"catboost_{k}": v for k, v in params.items() if k in ["iterations","depth","learning_rate","l2_leaf_reg","bagging_temperature","border_count"]})


    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)


    # Predict & evaluate
    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics = {
    "accuracy": float(accuracy_score(y_test, preds)),
    "precision": float(precision_score(y_test, preds, zero_division=0)),
    "recall": float(recall_score(y_test, preds)),
    "f1": float(f1_score(y_test, preds)),
    "roc_auc": float(roc_auc_score(y_test, probs)),
    "pr_auc": float(average_precision_score(y_test, probs)),
    # flattened confusion matrix entries (MLflow-friendly floats)
    "confusion_tn": float(tn),
    "confusion_fp": float(fp),
    "confusion_fn": float(fn),
    "confusion_tp": float(tp),
    }

    # Log scalar metrics (safe)
    try:
        mlflow.log_metrics(metrics)
    except Exception as e:
        log.error(f"Failed logging metrics to MLflow: {e}")
        # fallback: save metrics locally
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir)/"metrics_fallback.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

    # save model as artifact
    model_path = Path(output_dir) / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "catboost_model.cbm"
    model.save_model(str(model_file), format="cbm")
    try:
        mlflow.log_artifact(str(model_file))
    except Exception:
        log.warning("Could not log model artifact to MLflow; saved locally instead.")

    # log feature importance
    fi = model.get_feature_importance(type="FeatureImportance")
    fi_df = pd.DataFrame({"feature": X_train.columns, "importance": fi})
    fi_path = Path(output_dir) / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    mlflow.log_artifact(str(fi_path))


    # Save confusion matrix and run meta as artifacts
    cm_path = Path(output_dir) / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as fh:
        json.dump({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, fh, indent=2)
    mlflow.log_artifact(str(cm_path))


    run_data = {
    "run_id": run.info.run_id,
    "metrics": metrics,
    "params": params,
    "model_path": str(model_file)
    }
    run_meta = Path(output_dir) / "run_meta.json"
    with open(run_meta, "w", encoding="utf-8") as fh:
        json.dump(run_data, fh, indent=2)
    mlflow.log_artifact(str(run_meta))


    log.info("Training completed and artifacts logged to MLflow.")
    return run_data