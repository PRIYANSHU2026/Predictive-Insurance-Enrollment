# src/api.py
# ============================================================
# FastAPI Application — Real-Time Enrollment Prediction
# ============================================================
"""
A lightweight REST API that loads the trained champion model and
serves predictions over HTTP.

Endpoints
---------
GET  /              → Health check / welcome message.
GET  /health        → Liveness probe returning model load status.
POST /predict       → Accept employee JSON, return enrollment prediction.
POST /predict/batch → Accept a list of employees, return batch predictions.

Run locally:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import BEST_MODEL_PATH

logger = logging.getLogger(__name__)


# ── Pydantic Schemas ──────────────────────────────────────
class GenderEnum(str, Enum):
    """Accepted gender values."""
    male = "Male"
    female = "Female"


class MaritalStatusEnum(str, Enum):
    """Accepted marital status values."""
    single = "Single"
    married = "Married"
    divorced = "Divorced"


class EmploymentTypeEnum(str, Enum):
    """Accepted employment type values."""
    full_time = "Full-time"
    part_time = "Part-time"
    contract = "Contract"


class RegionEnum(str, Enum):
    """Accepted US region values."""
    west = "West"
    midwest = "Midwest"
    northeast = "Northeast"
    south = "South"


class HasDependentsEnum(str, Enum):
    """Accepted has_dependents values."""
    yes = "Yes"
    no = "No"


class EmployeeInput(BaseModel):
    """
    Schema for a single employee prediction request.
    Mirrors the raw feature columns of the training dataset.
    """
    age: int = Field(..., ge=18, le=100, description="Employee age in years")
    gender: GenderEnum
    marital_status: MaritalStatusEnum
    employment_type: EmploymentTypeEnum
    region: RegionEnum
    has_dependents: HasDependentsEnum
    salary: float = Field(..., gt=0, description="Annual salary in USD")
    tenure_years: float = Field(..., ge=0, description="Years at the company")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "gender": "Male",
                    "marital_status": "Married",
                    "employment_type": "Full-time",
                    "region": "West",
                    "has_dependents": "Yes",
                    "salary": 72000.0,
                    "tenure_years": 5.2,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    enrolled_prediction: int = Field(..., description="0 = not enrolled, 1 = enrolled")
    enrollment_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of enrollment"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str


# ── Application Lifespan (model loading) ──────────────────
# Using the modern lifespan approach instead of deprecated on_event.
model_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; release on shutdown."""
    try:
        pipeline = joblib.load(BEST_MODEL_PATH)
        model_store["pipeline"] = pipeline
        logger.info("✅ Model loaded from %s", BEST_MODEL_PATH)
    except FileNotFoundError:
        logger.error(
            "❌ No trained model found at %s. "
            "Run `python run_pipeline.py` first.",
            BEST_MODEL_PATH,
        )
        model_store["pipeline"] = None
    yield
    model_store.clear()


# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description=(
        "Predict whether an employee will opt in to a voluntary "
        "insurance product based on demographic and employment data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_pipeline():
    """Retrieve the loaded pipeline or raise 503."""
    pipeline = model_store.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with `python run_pipeline.py`.",
        )
    return pipeline


def _input_to_dataframe(employee: EmployeeInput) -> pd.DataFrame:
    """Convert a Pydantic model to the DataFrame format the pipeline expects."""
    return pd.DataFrame(
        [
            {
                "age": employee.age,
                "gender": employee.gender.value,
                "marital_status": employee.marital_status.value,
                "employment_type": employee.employment_type.value,
                "region": employee.region.value,
                "has_dependents": employee.has_dependents.value,
                "salary": employee.salary,
                "tenure_years": employee.tenure_years,
            }
        ]
    )


# ── Endpoints ─────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    """Welcome / discovery endpoint."""
    return {
        "message": "Insurance Enrollment Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Liveness probe — confirms the service and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=model_store.get("pipeline") is not None,
        model_path=str(BEST_MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(employee: EmployeeInput):
    """
    Predict enrollment for a single employee.

    Accepts a JSON body matching the ``EmployeeInput`` schema and
    returns the binary prediction along with the enrollment probability.
    """
    pipeline = _get_pipeline()
    df = _input_to_dataframe(employee)

    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0][1])

    return PredictionResponse(
        enrolled_prediction=prediction,
        enrollment_probability=round(probability, 4),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
)
def predict_batch(employees: List[EmployeeInput]):
    """
    Predict enrollment for a batch of employees.

    Accepts a JSON array of ``EmployeeInput`` objects and returns
    predictions for each.
    """
    if not employees:
        raise HTTPException(status_code=400, detail="Empty batch.")
    if len(employees) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000.",
        )

    pipeline = _get_pipeline()

    # Build a single DataFrame for vectorised prediction
    rows = [
        {
            "age": e.age,
            "gender": e.gender.value,
            "marital_status": e.marital_status.value,
            "employment_type": e.employment_type.value,
            "region": e.region.value,
            "has_dependents": e.has_dependents.value,
            "salary": e.salary,
            "tenure_years": e.tenure_years,
        }
        for e in employees
    ]
    df = pd.DataFrame(rows)

    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)[:, 1]

    results = [
        PredictionResponse(
            enrolled_prediction=int(p),
            enrollment_probability=round(float(prob), 4),
        )
        for p, prob in zip(preds, probas)
    ]

    return BatchPredictionResponse(predictions=results)
