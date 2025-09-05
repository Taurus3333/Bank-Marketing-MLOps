# bank_marketing/services/inference.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import os
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ..core.logging import get_logger
from ..config import MODEL_PATH, PRED_THRESHOLD
from ..pipelines.transform import feature_engineer

log = get_logger(__name__)

CAT_FEATURE_NAMES = [
    "job", "marital", "education", "contact", "month", "poutcome",
    "age_group", "pdays_group", "previous_group"
]

class ModelService:
    _instance = None

    def __init__(self, model_path: str | Path, threshold: float):
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self._model = None
        self._cat_indices_cache = None

    @classmethod
    def instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls(
                model_path=MODEL_PATH,
                threshold=PRED_THRESHOLD,
            )
        return cls._instance

    def _load(self):
        if self._model is None:
            p = Path(self.model_path)
            if not p.exists():
                raise FileNotFoundError(f"Model not found at {p}")
            log.info(f"Loading CatBoost model from {p}")
            m = CatBoostClassifier()
            m.load_model(str(p))
            self._model = m

    def _cat_indices(self, columns: list[str]) -> list[int]:
        if self._cat_indices_cache is None:
            self._cat_indices_cache = [
                columns.index(c) for c in CAT_FEATURE_NAMES if c in columns
            ]
        return self._cat_indices_cache

    def predict_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deterministic FE, run model, return probs + decisions.
        """
        self._load()

        # ensure deterministic FE used in training
        df = feature_engineer(df_raw)

        # separate target if present (ignore in inference)
        if "y" in df.columns:
            df = df.drop(columns=["y"])

        # CatBoost: create Pool with categorical indices
        cat_idx = self._cat_indices(df.columns.tolist())
        probs = self._model.predict_proba(Pool(df, cat_features=cat_idx))[:, 1]
        decision = (probs >= self.threshold).astype(int)

        out = df_raw.copy()
        out["_prob_yes"] = probs
        out["_decision"] = decision
        return out

    def predict_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        df_raw = pd.DataFrame.from_records(records)
        preds = self.predict_df(df_raw)
        return preds[["_prob_yes", "_decision"]].round(6).to_dict(orient="records")
