"""
mdas/adapter.py — Simulation bridge: EpiMetricRecords -> MAPS init fractions.

MAPS itself (build_maps_initial_conditions_with_history_full.py) does not
model individual agents — it is a grid/metapopulation model: each lat/lon
cell holds compartment population counts (S, I, A, R, ...) computed from
population *fractions* read out of its run config (see that script's
`s0_fraction`, `a_fraction`, `r_fraction` — e.g. lines ~822, ~861-862).
There is no S/I/R/V-per-agent-object layer anywhere in this codebase, and no
existing V (vaccinated) compartment either.

So rather than emit a fictional "list of agent objects" that nothing in this
repo consumes, this adapter emits one PopulationInitFractions per region:
the same shape of thing (S/I/R fractions, plus a new V overlay and an
immunity factor) that a run config already carries, ready to merge into that
config or into a per-FIPS override table. `to_compartment_overrides()`
below produces exactly the key names build_maps_initial_conditions expects
(s0_fraction, a_fraction, r_fraction), plus vaccination_fraction as a new
key that script does not yet read — wiring that in is a follow-up change to
that script, out of scope here, but the value is ready and named to match.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from mdas.schemas import DataProvenance, EpiMetricRecord, MetricType


class PopulationInitFractions(BaseModel):
    """Grid-cell-level (or region-level, pre-rasterization) agent init distribution."""

    region: str
    s_fraction: float = Field(..., ge=0.0, le=1.0, description="Susceptible share.")
    i_fraction: float = Field(..., ge=0.0, le=1.0, description="Currently-infectious share.")
    r_fraction: float = Field(..., ge=0.0, le=1.0, description="Recovered / prior-infection share.")
    v_fraction: float = Field(..., ge=0.0, le=1.0, description="Vaccinated share (overlay, not part of the S/I/R partition).")
    vaccination_tier_shares: Dict[str, float] = Field(
        default_factory=dict, description='e.g. {"unvaccinated": .., "partial": .., "full": ..}'
    )
    natural_immunity_factor: float = Field(
        ..., ge=0.0, le=1.0, description="Derived from past-infection estimates; feeds Ch/Cn-style immune memory."
    )
    provenance_summary: Dict[str, int] = Field(
        default_factory=dict, description="Count of input records used, by data_provenance value."
    )

    @model_validator(mode="after")
    def _check_sir_partition(self) -> "PopulationInitFractions":
        total = self.s_fraction + self.i_fraction + self.r_fraction
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"s_fraction + i_fraction + r_fraction = {total:.4f} exceeds 1.0 for region {self.region!r}"
            )
        return self


def _latest(records: List[EpiMetricRecord], metric_type: MetricType) -> Optional[EpiMetricRecord]:
    matches = [r for r in records if r.metric_type == metric_type]
    return max(matches, key=lambda r: r.observation_date) if matches else None


def build_population_init_fractions(
    records: List[EpiMetricRecord],
    *,
    region: str,
    population: Optional[float] = None,
) -> PopulationInitFractions:
    """
    Fold every record available for one region into a PopulationInitFractions.

    Expects `records` already filtered (or not — non-matching regions are
    ignored) to metrics for `region`; typically this is
    `default_engine.run(measured_and_census_records, include_inputs=True)`
    so both direct/census inputs and heuristic-imputed outputs are present.

    `daily_infections_estimated` / `cumulative_cases` values that are raw
    counts (rather than an already-normalized fraction) are converted to a
    population fraction when `population` is given; otherwise they are
    assumed to already be expressed as a 0-1 fraction of the region.
    """
    region_records = [r for r in records if r.region == region]

    sero = _latest(region_records, MetricType.SEROPREVALENCE_PCT)
    infections = _latest(region_records, MetricType.DAILY_INFECTIONS_ESTIMATED) or _latest(
        region_records, MetricType.CUMULATIVE_CASES
    )
    vaccine = _latest(region_records, MetricType.VACCINE_DOSES_TOTAL)

    def as_fraction(rec: Optional[EpiMetricRecord]) -> float:
        if rec is None:
            return 0.0
        if rec.unit == "fraction_of_population":
            return rec.value
        if rec.metric_type in (MetricType.SEROPREVALENCE_PCT,):
            return rec.value / 100.0
        if population:
            return min(1.0, rec.value / population)
        if rec.value > 1.0:
            raise ValueError(
                f"{rec.metric_type.value} for region {region!r} is a raw count ({rec.value}) "
                "with no unit='fraction_of_population' and no `population` argument given, so "
                "it cannot be normalized to a fraction. Pass `population=...` or ingest it "
                "already expressed as a 0-1 fraction."
            )
        return rec.value

    # Seroprevalence (cumulative past exposure) drives r_fraction; a separate
    # daily/cumulative case count drives i_fraction (currently-infectious) —
    # the two are independent signals, not a fallback chain.
    r_fraction = as_fraction(sero)
    i_fraction = as_fraction(infections)
    v_fraction = as_fraction(vaccine)
    s_fraction = max(0.0, 1.0 - i_fraction - r_fraction)

    vaccination_tier_shares = {
        "full": v_fraction,
        "unvaccinated": max(0.0, 1.0 - v_fraction),
    }

    provenance_summary: Dict[str, int] = {}
    for rec in region_records:
        key = rec.data_provenance.value
        provenance_summary[key] = provenance_summary.get(key, 0) + 1

    return PopulationInitFractions(
        region=region,
        s_fraction=s_fraction,
        i_fraction=i_fraction,
        r_fraction=r_fraction,
        v_fraction=v_fraction,
        vaccination_tier_shares=vaccination_tier_shares,
        natural_immunity_factor=r_fraction,
        provenance_summary=provenance_summary,
    )


def build_population_init_fractions_by_region(
    records: List[EpiMetricRecord], *, population_by_region: Optional[Dict[str, float]] = None
) -> Dict[str, PopulationInitFractions]:
    """Same as build_population_init_fractions, once per distinct region present in records."""
    population_by_region = population_by_region or {}
    regions = sorted({r.region for r in records})
    return {
        region: build_population_init_fractions(
            records, region=region, population=population_by_region.get(region)
        )
        for region in regions
    }


def to_compartment_overrides(fractions: PopulationInitFractions) -> Dict[str, float]:
    """
    Render as the run-config keys build_maps_initial_conditions_with_history_full.py
    reads (s0_fraction, a_fraction, r_fraction), plus vaccination_fraction —
    a new key for that script to pick up when it grows a V overlay.
    """
    return {
        "s0_fraction": fractions.s_fraction,
        "a_fraction": fractions.i_fraction,
        "r_fraction": fractions.r_fraction,
        "vaccination_fraction": fractions.v_fraction,
    }
