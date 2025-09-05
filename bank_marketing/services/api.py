# bank_marketing/services/api.py
from __future__ import annotations
from fastapi import FastAPI
from ..core.logging import get_logger
from ..config import MODEL_PATH, PRED_THRESHOLD
from .schemas import PredictRequest, PredictResponse, PredictResponseItem, HealthResponse
from .inference import ModelService
from .schemas import HealthResponse

log = get_logger(__name__)
app = FastAPI(title="Bank Marketing API", version="1.0.0")



@app.get("/health", response_model=HealthResponse)
def health():
    svc = ModelService.instance()
    # return using alias name to match schema
    return HealthResponse(status="ok", threshold=svc.threshold, modelFilePath=svc.model_path)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    svc = ModelService.instance()
    results = svc.predict_records([r.dict(exclude_none=True) for r in req.instances])
    return PredictResponse(results=[PredictResponseItem(prob_yes=r["_prob_yes"], decision=r["_decision"]) for r in results])
