"""
mdas/schemas.py — Target schema for every epidemiological data point MDAS ingests.

A single record type, EpiMetricRecord, is shared by both ingestion pipelines
(continuous time-series and census/survey snapshots) and by the inference
engine's imputed output. What differs between them is `data_provenance`:

    MEASURED_DIRECT    — read straight off a public health API/registry.
    CENSUS_DERIVED     — read off a periodic survey/census table.
    HEURISTIC_IMPUTED  — produced by a rule in mdas/inference/ from other
                          records, never observed directly.

Every downstream consumer (mdas/adapter.py, MAPS itself) can branch on this
tag to decide how much to trust a value, instead of silently mixing measured
and assumed numbers.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class MetricType(str, Enum):
    DAILY_INFECTIONS_ESTIMATED = "daily_infections_estimated"
    CUMULATIVE_CASES = "cumulative_cases"
    VACCINE_DOSES_TOTAL = "vaccine_doses_total"
    SEROPREVALENCE_PCT = "seroprevalence_pct"
    TESTED_POPULATION_PCT = "tested_population_pct"


# Metrics expressed as a percentage of population, 0-100. Used to bound-check
# `value` — a percentage field with 130 in it is a unit-mixup, not a data point.
_PERCENT_METRICS = frozenset({MetricType.SEROPREVALENCE_PCT, MetricType.TESTED_POPULATION_PCT})


class DataProvenance(str, Enum):
    MEASURED_DIRECT = "MEASURED_DIRECT"
    CENSUS_DERIVED = "CENSUS_DERIVED"
    HEURISTIC_IMPUTED = "HEURISTIC_IMPUTED"


class DemographicStrata(BaseModel):
    """Optional breakdown dimensions a metric can be reported against."""

    age_group: Optional[str] = None
    region: Optional[str] = None
    census_tract: Optional[str] = None

    @model_validator(mode="after")
    def _require_at_least_one(self) -> "DemographicStrata":
        if self.age_group is None and self.region is None and self.census_tract is None:
            raise ValueError(
                "DemographicStrata was constructed with no dimensions set; "
                "omit demographic_strata entirely (leave it None) instead of "
                "passing an empty breakdown."
            )
        return self


class EpiMetricRecord(BaseModel):
    """One data point: a single metric, for a single region, on a single date."""

    pathogen: str = "SARS-CoV-2"
    metric_type: MetricType
    value: float
    unit: Optional[str] = None
    region: str = Field(..., description="Region identifier, e.g. a state/county FIPS code.")
    observation_date: date
    data_provenance: DataProvenance
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="[lower_bound, upper_bound], for survey/sample- or heuristic-derived figures.",
    )
    demographic_strata: Optional[DemographicStrata] = None
    source: Optional[str] = Field(
        default=None, description="Dataset/API/rule name this record came from, for traceability."
    )

    @field_validator("value")
    @classmethod
    def _value_is_finite_and_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"value must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _check_percent_bounds(self) -> "EpiMetricRecord":
        if self.metric_type in _PERCENT_METRICS and not (0.0 <= self.value <= 100.0):
            raise ValueError(
                f"{self.metric_type.value} is a percentage metric; value must be in "
                f"[0, 100], got {self.value}"
            )
        return self

    @model_validator(mode="after")
    def _check_confidence_interval_order(self) -> "EpiMetricRecord":
        if self.confidence_interval is not None:
            lower, upper = self.confidence_interval
            if lower > upper:
                raise ValueError(
                    f"confidence_interval lower_bound ({lower}) must be <= upper_bound ({upper})"
                )
        return self
