"""
tests/test_mdas_inference.py — Assumption engine logic (requirement 5, bullet 2).

Run with: pytest tests/test_mdas_inference.py -v
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mdas.inference.engine import InferenceEngine
from mdas.inference.rules import (
    LatentInfectionConfig,
    VaccinationFromTestingConfig,
    make_latent_infection_rate_rule,
    make_vaccination_from_testing_rule,
)
from mdas.schemas import DataProvenance, EpiMetricRecord, MetricType


def _tested_pct(region: str, value: float) -> EpiMetricRecord:
    return EpiMetricRecord(
        metric_type=MetricType.TESTED_POPULATION_PCT,
        value=value,
        region=region,
        observation_date=date(2021, 6, 1),
        data_provenance=DataProvenance.CENSUS_DERIVED,
    )


def _cumulative_cases(region: str, value: float) -> EpiMetricRecord:
    return EpiMetricRecord(
        metric_type=MetricType.CUMULATIVE_CASES,
        value=value,
        region=region,
        observation_date=date(2021, 6, 1),
        data_provenance=DataProvenance.MEASURED_DIRECT,
    )


class TestVaccinationFromTestingRule(unittest.TestCase):
    def test_matches_expected_conditional_probability_formula(self):
        config = VaccinationFromTestingConfig(
            p_vax_given_tested=0.85, p_vax_given_not_tested=0.45, sensitivity_delta=0.05
        )
        rule = make_vaccination_from_testing_rule(config)
        tested_frac = 0.60
        records = [_tested_pct("US-CA", tested_frac * 100)]

        out = rule(records, region="US-CA", pathogen="SARS-CoV-2")

        self.assertEqual(len(out), 1)
        expected = tested_frac * 0.85 + (1 - tested_frac) * 0.45
        self.assertAlmostEqual(out[0].value, expected, places=6)
        self.assertEqual(out[0].data_provenance, DataProvenance.HEURISTIC_IMPUTED)
        self.assertEqual(out[0].metric_type, MetricType.VACCINE_DOSES_TOTAL)

    def test_confidence_interval_brackets_point_estimate(self):
        rule = make_vaccination_from_testing_rule(VaccinationFromTestingConfig())
        out = rule([_tested_pct("US-CA", 50.0)], region="US-CA", pathogen="SARS-CoV-2")
        lower, upper = out[0].confidence_interval
        self.assertLessEqual(lower, out[0].value)
        self.assertGreaterEqual(upper, out[0].value)

    def test_no_output_without_required_input(self):
        rule = make_vaccination_from_testing_rule(VaccinationFromTestingConfig())
        out = rule([], region="US-CA", pathogen="SARS-CoV-2")
        self.assertEqual(out, [])


class TestLatentInfectionRateRule(unittest.TestCase):
    def test_higher_testing_coverage_yields_smaller_undercount_multiplier(self):
        config = LatentInfectionConfig()
        rule = make_latent_infection_rate_rule(config)

        low_testing = rule(
            [_cumulative_cases("US-CA", 1000), _tested_pct("US-CA", 10.0)],
            region="US-CA", pathogen="SARS-CoV-2",
        )[0]
        high_testing = rule(
            [_cumulative_cases("US-CA", 1000), _tested_pct("US-CA", 90.0)],
            region="US-CA", pathogen="SARS-CoV-2",
        )[0]

        # Same reported count, but lower testing coverage implies a bigger
        # undercount, so the imputed true-infection estimate should be higher.
        self.assertGreater(low_testing.value, high_testing.value)
        self.assertEqual(low_testing.metric_type, MetricType.DAILY_INFECTIONS_ESTIMATED)

    def test_bounds_are_ordered(self):
        rule = make_latent_infection_rate_rule(LatentInfectionConfig())
        out = rule(
            [_cumulative_cases("US-CA", 1000), _tested_pct("US-CA", 50.0)],
            region="US-CA", pathogen="SARS-CoV-2",
        )[0]
        lower, upper = out.confidence_interval
        self.assertLessEqual(lower, upper)


class TestInferenceEngine(unittest.TestCase):
    def test_engine_runs_registered_rule_per_region_and_tags_provenance(self):
        engine = InferenceEngine()
        engine.register(
            "vaccination_from_testing", make_vaccination_from_testing_rule(VaccinationFromTestingConfig())
        )

        records = [_tested_pct("US-CA", 60.0), _tested_pct("US-TX", 20.0)]
        inferred = engine.run(records)

        self.assertEqual(len(inferred), 2)
        regions = {r.region for r in inferred}
        self.assertEqual(regions, {"US-CA", "US-TX"})
        self.assertTrue(all(r.data_provenance == DataProvenance.HEURISTIC_IMPUTED for r in inferred))

    def test_include_inputs_preserves_original_records(self):
        engine = InferenceEngine()
        engine.register(
            "vaccination_from_testing", make_vaccination_from_testing_rule(VaccinationFromTestingConfig())
        )
        records = [_tested_pct("US-CA", 60.0)]
        combined = engine.run(records, include_inputs=True)
        self.assertEqual(len(combined), 2)
        self.assertIn(records[0], combined)

    def test_duplicate_registration_raises(self):
        engine = InferenceEngine()
        engine.register("r1", make_vaccination_from_testing_rule(VaccinationFromTestingConfig()))
        with self.assertRaises(ValueError):
            engine.register("r1", make_vaccination_from_testing_rule(VaccinationFromTestingConfig()))


if __name__ == "__main__":
    unittest.main()
