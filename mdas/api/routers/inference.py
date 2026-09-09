"""mdas/api/routers/inference.py — Run the heuristic-rule engine over posted records."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter

from mdas.api.schemas import InferenceRunRequest
from mdas.inference import default_engine
import mdas.inference.rules  # noqa: F401 -- registers the example rules on import
from mdas.schemas import EpiMetricRecord

router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/rules", response_model=List[str])
def list_rules() -> List[str]:
    return default_engine.list_rules()


@router.post("/run", response_model=List[EpiMetricRecord])
def run_inference(body: InferenceRunRequest) -> List[EpiMetricRecord]:
    return default_engine.run(
        body.records, rule_names=body.rule_names, include_inputs=body.include_inputs
    )
