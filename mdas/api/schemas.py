"""
mdas/api/schemas.py — Request bodies for endpoints that take multiple inputs.

EpiMetricRecord and PopulationInitFractions (mdas/schemas.py, mdas/adapter.py)
are used directly as request/response bodies wherever a single one of those
is all an endpoint needs. The wrappers here exist only where an endpoint
needs one of those *plus* something else (a rule selection, a population
table) that doesn't belong on the domain model itself.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from mdas.schemas import EpiMetricRecord


class InferenceRunRequest(BaseModel):
    records: List[EpiMetricRecord]
    rule_names: Optional[List[str]] = None
    include_inputs: bool = True


class InitFractionsRequest(BaseModel):
    records: List[EpiMetricRecord]
    population_by_region: Optional[Dict[str, float]] = None
