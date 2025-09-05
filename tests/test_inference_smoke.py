# tests/test_inference_smoke.py
from bank_marketing.services.inference import ModelService

def test_inference_direct():
    svc = ModelService.instance()
    rows = [{
        "age": 44, "balance": 29, "campaign": 1, "pdays": -1, "previous": 0, "day": 5,
        "default": 0, "housing": 1, "loan": 0,
        "job": "technician", "marital": "single", "education": "secondary",
        "contact": "unknown", "month": "may", "poutcome": "unknown"
    }]
    out = svc.predict_records(rows)[0]
    # keys are _prob_yes and _decision in the internal dict
    assert 0.0 <= out["_prob_yes"] <= 1.0
    assert out["_decision"] in (0, 1)
