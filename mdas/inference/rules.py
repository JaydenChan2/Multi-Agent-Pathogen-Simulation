"""
mdas/inference/rules.py — Example heuristic rules for the inference engine.

Each rule below is built as `make_<rule>(config) -> HeuristicRule` so every
tunable assumption is a plain, overridable dataclass field rather than a
buried constant — the research team can register a variant with different
numbers (`default_engine.register("my_variant", make_vaccination_from_testing_rule(MyConfig(...)))`)
without touching this file. The module-level `vaccination_from_testing` and
`latent_infection_rate_from_testing` names are the default-config instances,
already registered against mdas.inference.default_engine on import.

WEIGHT — every constant that scales an assumption's magnitude
-----------------------------------------------------------------
WEIGHT 1 — VaccinationFromTestingConfig.p_vax_given_tested /
    p_vax_given_not_tested: the conditional-probability assumption from the
    spec — "if census indicates X% underwent voluntary COVID testing, model
    P(Vaccinated | Tested) = 0.85 and P(Vaccinated | Not Tested) = 0.45".
    These are a plausibility assumption, not a measurement; revisit them
    whenever a real paired testing/vaccination survey becomes available.
WEIGHT 2 — VaccinationFromTestingConfig.sensitivity_delta: how far the two
    conditional probabilities are perturbed (symmetrically) to build the
    output confidence_interval. Larger = wider, more conservative bounds.
WEIGHT 3 — LatentInfectionConfig.baseline_ascertainment /
    testing_coverage_slope: models the fraction of true infections that
    show up as a reported case as increasing with testing coverage
    (ascertainment_rate = baseline + slope * tested_fraction, clamped to
    [ascertainment_min, ascertainment_max]). true_infections = reported /
    ascertainment_rate. These are calibrated to "low testing -> heavy
    undercount, high testing -> modest undercount", not to a fitted curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from mdas.inference.engine import HeuristicRule, register_rule
from mdas.schemas import DataProvenance, EpiMetricRecord, MetricType


def _latest(records: List[EpiMetricRecord], metric_type: MetricType) -> Optional[EpiMetricRecord]:
    matches = [r for r in records if r.metric_type == metric_type]
    return max(matches, key=lambda r: r.observation_date) if matches else None


@dataclass(frozen=True)
class VaccinationFromTestingConfig:
    p_vax_given_tested: float = 0.85
    p_vax_given_not_tested: float = 0.45
    sensitivity_delta: float = 0.05


def make_vaccination_from_testing_rule(config: VaccinationFromTestingConfig) -> HeuristicRule:
    """
    Baseline vaccine-uptake estimate from voluntary-testing coverage.

    Requires a TESTED_POPULATION_PCT record for the region (typically
    CENSUS_DERIVED). Absent a direct vaccination-coverage figure, assumes
    the population splits into "tested" / "not tested" and each group has a
    different (configurable) probability of being vaccinated:

        uptake = tested_frac * P(Vaccinated | Tested)
               + (1 - tested_frac) * P(Vaccinated | Not Tested)

    Emits a VACCINE_DOSES_TOTAL record whose `value` is a population
    *fraction* (0-1, see unit="fraction_of_population") rather than a raw
    dose count, since no absolute population figure is required as input.
    """

    def rule(records: List[EpiMetricRecord], *, region: str, pathogen: str) -> List[EpiMetricRecord]:
        tested = _latest(records, MetricType.TESTED_POPULATION_PCT)
        if tested is None:
            return []

        tested_frac = tested.value / 100.0

        def uptake_at(p_tested: float, p_not_tested: float) -> float:
            return tested_frac * p_tested + (1 - tested_frac) * p_not_tested

        point = uptake_at(config.p_vax_given_tested, config.p_vax_given_not_tested)
        lower = uptake_at(
            config.p_vax_given_tested - config.sensitivity_delta,
            config.p_vax_given_not_tested - config.sensitivity_delta,
        )
        upper = uptake_at(
            config.p_vax_given_tested + config.sensitivity_delta,
            config.p_vax_given_not_tested + config.sensitivity_delta,
        )
        lower, upper = max(0.0, lower), min(1.0, upper)

        return [
            EpiMetricRecord(
                pathogen=pathogen,
                metric_type=MetricType.VACCINE_DOSES_TOTAL,
                value=point,
                unit="fraction_of_population",
                region=region,
                observation_date=tested.observation_date,
                data_provenance=DataProvenance.HEURISTIC_IMPUTED,
                confidence_interval=(lower, upper),
            )
        ]

    return rule


@dataclass(frozen=True)
class LatentInfectionConfig:
    baseline_ascertainment: float = 0.20
    testing_coverage_slope: float = 0.60
    ascertainment_min: float = 0.10
    ascertainment_max: float = 0.90
    ascertainment_uncertainty_delta: float = 0.10


def make_latent_infection_rate_rule(config: LatentInfectionConfig) -> HeuristicRule:
    """
    Estimate true (latent) infection count from reported cases + testing coverage.

    Requires a CUMULATIVE_CASES record (the reported count) and a
    TESTED_POPULATION_PCT record (testing coverage) for the region. Models
    the ascertainment rate — the fraction of true infections that surface
    as a reported case — as rising with testing coverage, then backs out
    true infections as reported / ascertainment_rate.
    """

    def rule(records: List[EpiMetricRecord], *, region: str, pathogen: str) -> List[EpiMetricRecord]:
        reported = _latest(records, MetricType.CUMULATIVE_CASES)
        tested = _latest(records, MetricType.TESTED_POPULATION_PCT)
        if reported is None or tested is None:
            return []

        tested_frac = tested.value / 100.0

        def ascertainment_at(delta: float) -> float:
            rate = config.baseline_ascertainment + config.testing_coverage_slope * tested_frac + delta
            return min(config.ascertainment_max, max(config.ascertainment_min, rate))

        point_rate = ascertainment_at(0.0)
        # Higher ascertainment -> smaller undercount -> lower true-infection estimate.
        lower = reported.value / ascertainment_at(config.ascertainment_uncertainty_delta)
        upper = reported.value / ascertainment_at(-config.ascertainment_uncertainty_delta)

        return [
            EpiMetricRecord(
                pathogen=pathogen,
                metric_type=MetricType.DAILY_INFECTIONS_ESTIMATED,
                value=reported.value / point_rate,
                unit=reported.unit,
                region=region,
                observation_date=reported.observation_date,
                data_provenance=DataProvenance.HEURISTIC_IMPUTED,
                confidence_interval=(lower, upper),
            )
        ]

    return rule


vaccination_from_testing = make_vaccination_from_testing_rule(VaccinationFromTestingConfig())
latent_infection_rate_from_testing = make_latent_infection_rate_rule(LatentInfectionConfig())

register_rule("vaccination_from_testing")(vaccination_from_testing)
register_rule("latent_infection_rate_from_testing")(latent_infection_rate_from_testing)
