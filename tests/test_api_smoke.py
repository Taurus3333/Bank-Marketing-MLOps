# tests/test_api_smoke.py
from bank_marketing.services.api import health, predict
from bank_marketing.services.schemas import PredictRequest, BankRecord

def test_health_direct():
    res = health()
    assert res.status == "ok"
    assert hasattr(res, "model_path")
    assert hasattr(res, "threshold")

def test_predict_direct_minimal():
    # build a minimal bank record (use only required fields for inference)
    record = BankRecord(
        age=40,
        balance=500,
        campaign=2,
        pdays=-1,
        previous=0,
        day=5,
        default=0,
        housing=0,
        loan=0,
        job="management",
        marital="married",
        education="tertiary",
        contact="cellular",
        month="may",
        poutcome="unknown",
    )
    req = PredictRequest(instances=[record])
    resp = predict(req)
    assert len(resp.results) == 1
    item = resp.results[0]
    assert 0.0 <= item.prob_yes <= 1.0
    assert item.decision in (0, 1)
