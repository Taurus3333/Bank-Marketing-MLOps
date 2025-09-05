from __future__ import annotations
import pandas as pd
import numpy as np
from ..core.logging import get_logger



log = get_logger(__name__)




def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic feature engineering matching experiments.
    - clip balance & campaign
    - create age_group, pdays_group, previous_group
    - drop leakage columns (duration)
    - ensure column types are stable

    """
    df = df.copy()


    # Drop leakage
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])


    # Safe binary mapping for common variants
    _binary_map = {"yes": 1, "no": 0, 1: 1, 0: 0, "1": 1, "0": 0}


    for b in ["default", "housing", "loan", "y"]:
        if b in df.columns:
            # Normalize strings first (strip/lower) then map
            df[b] = df[b].astype(str).str.strip().str.lower().map(lambda v: _binary_map.get(v, v))
            # Coerce to numeric safely, fill NA with 0, ensure int
            df[b] = pd.to_numeric(df[b], errors="coerce").fillna(0).astype(int)


    # Clip numeric
    if "balance" in df.columns:
        df["balance_clipped"] = df["balance"].clip(-2000, 5000).astype("int32")
    else:
        df["balance_clipped"] = 0


    if "campaign" in df.columns:
        df["campaign_clipped"] = df["campaign"].clip(0, 10).astype("int32")
    else:
        df["campaign_clipped"] = 0


    # Age groups
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[17,25,35,45,55,65,100],
    labels=["18-25","26-35","36-45","46-55","56-65","66+"])
    else:
        df["age_group"] = "unknown"


    # pdays group
    if "pdays" in df.columns:
        df["pdays_group"] = np.where(df["pdays"] == -1, "never_contacted",
        pd.cut(df["pdays"], bins=[0,30,90,180,999],
        labels=["<30","30-90","90-180","180+"]))
    else:
        df["pdays_group"] = "never_contacted"


    # previous group
    if "previous" in df.columns:
        df["previous_group"] = pd.cut(df["previous"], bins=[-1,0,2,10,999],
        labels=["0","1-2","3-10","10+"])
    else:
        df["previous_group"] = "0"
    
    # Standardize string/object columns (lowercase & trim)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().str.lower()


    # Keep deterministic column ordering (training contract)
    desired_cols = [
    "age", "age_group", "balance_clipped", "campaign_clipped",
    "default", "housing", "loan",
    "job", "marital", "education", "contact", "month", "poutcome",
    "pdays_group", "previous_group",
    "y"
    ]
    cols = [c for c in desired_cols if c in df.columns] + [c for c in df.columns if c not in desired_cols]


    produced = [c for c in cols if c in df.columns]
    log.info(f"Feature engineering produced columns: {produced}")


    return df[cols]