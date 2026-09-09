"""mdas/api/routers/simulation.py — Records in, MAPS-ready init fractions out."""
from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

from mdas.adapter import (
    PopulationInitFractions,
    build_population_init_fractions_by_region,
    to_compartment_overrides,
)
from mdas.api.schemas import InitFractionsRequest

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/init-fractions", response_model=Dict[str, PopulationInitFractions])
def init_fractions(body: InitFractionsRequest) -> Dict[str, PopulationInitFractions]:
    return build_population_init_fractions_by_region(
        body.records, population_by_region=body.population_by_region
    )


@router.post("/compartment-overrides", response_model=Dict[str, Dict[str, float]])
def compartment_overrides(body: InitFractionsRequest) -> Dict[str, Dict[str, float]]:
    fractions_by_region = build_population_init_fractions_by_region(
        body.records, population_by_region=body.population_by_region
    )
    return {
        region: to_compartment_overrides(fractions)
        for region, fractions in fractions_by_region.items()
    }
