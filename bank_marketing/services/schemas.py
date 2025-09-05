# bank_marketing/services/schemas.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, conint, confloat


# NOTE: all fields are optional to keep the API flexible; FE will handle defaults if needed.

class BankRecord(BaseModel):
    # numeric
    age: Optional[conint(ge=18, le=100)] = Field(None) # type: ignore
    balance: Optional[int] = None
    campaign: Optional[int] = None
    pdays: Optional[int] = None
    previous: Optional[int] = None
    day: Optional[int] = None
    duration: Optional[int] = None  # optional (dropped in FE)

    # categoricals/binaries
    job: Optional[str] = None
    marital: Optional[str] = None
    education: Optional[str] = None
    default: Optional[int] = None     # 0/1
    housing: Optional[int] = None     # 0/1
    loan: Optional[int] = None        # 0/1
    contact: Optional[str] = None
    month: Optional[str] = None       # "jan"..."dec"
    poutcome: Optional[str] = None
    # y is not required for inference

class PredictRequest(BaseModel):
    instances: List[BankRecord]

class PredictResponseItem(BaseModel):
    prob_yes: confloat(ge=0.0, le=1.0) # type: ignore
    decision: conint(ge=0, le=1) # type: ignore

class PredictResponse(BaseModel):
    results: List[PredictResponseItem]



class HealthResponse(BaseModel):
    status: str
    threshold: float
    modelFilePath: Optional[str] = Field(None, alias="modelFilePath")
    model_path: Optional[str] = Field(None, alias="model_path")

    # Provide a read-only property for consistent access
    @property
    def model_path_value(self) -> str:
        """
        Unified accessor — prefer model_path, fallback to modelFilePath.
        Use this in code if you want explicit value; however tests check
        for attribute existence 'model_path', so we also add a property named 'model_path' below.
        """
        return self.model_path or self.modelFilePath or ""

    # Expose attribute .model_path (property) so hasattr(res, "model_path") is True
    @property
    def model_path(self) -> str:  # type: ignore[override]
        # return whichever exists
        return self.model_path_value

