"""
tests/test_mdas_adapter.py — Mocked end-to-end: records -> MAPS init fractions
(requirement 5, bullet 3).

Run with: pytest tests/test_mdas_adapter.py -v
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mdas.adapter import (
    build_population_init_fractions,
    build_population_init_fractions_by_region,
    to_compartment_overrides,
)
from mdas.schemas import DataProvenance, EpiMetricRecord, MetricType


def _record(metric_type, value, region="US-CA", provenance=DataProvenance.MEASURED_DIRECT, unit=None):
    return EpiMetricRecord(
        metric_type=metric_type,
        value=value,
        unit=unit,
        region=region,
        observation_date=date(2021, 6, 1),
        data_provenance=provenance,
    )


class TestBuildPopulationInitFractions(unittest.TestCase):
    def test_combines_measured_census_and_imputed_sources(self):
        records = [
            _record(MetricType.SEROPREVALENCE_PCT, 20.0, provenance=DataProvenance.CENSUS_DERIVED),
            _record(MetricType.DAILY_INFECTIONS_ESTIMATED, 0.02, unit="fraction_of_population"),
            _record(
                MetricType.VACCINE_DOSES_TOTAL,
                0.55,
                unit="fraction_of_population",
                provenance=DataProvenance.HEURISTIC_IMPUTED,
            ),
        ]

        fractions = build_population_init_fractions(records, region="US-CA")

        self.assertEqual(fractions.region, "US-CA")
        self.assertAlmostEqual(fractions.r_fraction, 0.20)
        self.assertAlmostEqual(fractions.i_fraction, 0.02)
        self.assertAlmostEqual(fractions.v_fraction, 0.55)
        self.assertAlmostEqual(fractions.s_fraction, 1.0 - 0.02 - 0.20)
        self.assertAlmostEqual(fractions.natural_immunity_factor, 0.20)
        self.assertEqual(
            fractions.provenance_summary,
            {"CENSUS_DERIVED": 1, "MEASURED_DIRECT": 1, "HEURISTIC_IMPUTED": 1},
        )

    def test_missing_metrics_default_to_zero_and_full_susceptibility(self):
        fractions = build_population_init_fractions([], region="US-NV")
        self.assertEqual(fractions.s_fraction, 1.0)
        self.assertEqual(fractions.i_fraction, 0.0)
        self.assertEqual(fractions.r_fraction, 0.0)
        self.assertEqual(fractions.v_fraction, 0.0)

    def test_raw_count_normalized_by_population(self):
        records = [_record(MetricType.CUMULATIVE_CASES, 5000, unit=None)]
        fractions = build_population_init_fractions(records, region="US-CA", population=100_000)
        self.assertAlmostEqual(fractions.i_fraction, 0.05)

    def test_raw_count_without_population_raises(self):
        records = [_record(MetricType.CUMULATIVE_CASES, 5000, unit=None)]
        with self.assertRaises(ValueError):
            build_population_init_fractions(records, region="US-CA")

    def test_by_region_covers_every_distinct_region(self):
        records = [
            _record(MetricType.SEROPREVALENCE_PCT, 10.0, region="US-CA"),
            _record(MetricType.SEROPREVALENCE_PCT, 30.0, region="US-TX"),
        ]
        by_region = build_population_init_fractions_by_region(records)
        self.assertEqual(set(by_region), {"US-CA", "US-TX"})
        self.assertAlmostEqual(by_region["US-TX"].r_fraction, 0.30)


class TestCompartmentOverrides(unittest.TestCase):
    """Verify the shape matches what build_maps_initial_conditions expects
    (s0_fraction / a_fraction / r_fraction keys, see that script's cfg.get() calls)."""

    def test_maps_expected_keys_present_and_in_unit_range(self):
        records = [
            _record(MetricType.SEROPREVALENCE_PCT, 15.0, provenance=DataProvenance.CENSUS_DERIVED),
            _record(MetricType.DAILY_INFECTIONS_ESTIMATED, 0.01, unit="fraction_of_population"),
            _record(MetricType.VACCINE_DOSES_TOTAL, 0.40, unit="fraction_of_population"),
        ]
        fractions = build_population_init_fractions(records, region="US-CA")
        overrides = to_compartment_overrides(fractions)

        self.assertEqual(
            set(overrides), {"s0_fraction", "a_fraction", "r_fraction", "vaccination_fraction"}
        )
        for key, value in overrides.items():
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)


if __name__ == "__main__":
    unittest.main()
