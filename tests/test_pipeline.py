from bank_marketing.pipelines.transform import feature_engineer
import pandas as pd

def test_feature_engineer_minimal():
    sample = pd.DataFrame({
        "age":[30],
        "balance":[100],
        "campaign":[1],
        "default":[0],
        "housing":[1],
        "loan":[0],
        "job":["management"],
        "marital":["married"],
        "education":["tertiary"],
        "contact":["cellular"],
        "day":[5],
        "month":["may"],
        "duration":[100],
        "pdays":[-1],
        "previous":[0],
        "poutcome":["unknown"],
        "y":[0]
    })
    out = feature_engineer(sample)
    assert "balance_clipped" in out.columns
    assert "age_group" in out.columns
    assert out["balance_clipped"].iloc[0] == 100
