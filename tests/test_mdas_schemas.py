"""
tests/test_mdas_schemas.py — EpiMetricRecord validation.

Run with: pytest tests/test_mdas_schemas.py -v
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError

from mdas.schemas import DataProvenance, DemographicStrata, EpiMetricRecord, MetricType


class TestEpiMetricRecordDefaults(unittest.TestCase):
    def test_pathogen_defaults_to_sars_cov_2(self):
        rec = EpiMetricRecord(
            metric_type=MetricType.CUMULATIVE_CASES,
            value=100.0,
            region="US-CA",
            observation_date=date(2021, 3, 1),
            data_provenance=DataProvenance.MEASURED_DIRECT,
        )
        self.assertEqual(rec.pathogen, "SARS-CoV-2")

    def test_confidence_interval_and_strata_default_none(self):
        rec = EpiMetricRecord(
            metric_type=MetricType.CUMULATIVE_CASES,
            value=100.0,
            region="US-CA",
            observation_date=date(2021, 3, 1),
            data_provenance=DataProvenance.MEASURED_DIRECT,
        )
        self.assertIsNone(rec.confidence_interval)
        self.assertIsNone(rec.demographic_strata)


class TestEpiMetricRecordValidation(unittest.TestCase):
    def test_rejects_negative_value(self):
        with self.assertRaises(ValidationError):
            EpiMetricRecord(
                metric_type=MetricType.CUMULATIVE_CASES,
                value=-5.0,
                region="US-CA",
                observation_date=date(2021, 3, 1),
                data_provenance=DataProvenance.MEASURED_DIRECT,
            )

    def test_percent_metric_out_of_bounds_rejected(self):
        with self.assertRaises(ValidationError):
            EpiMetricRecord(
                metric_type=MetricType.SEROPREVALENCE_PCT,
                value=130.0,
                region="US-CA",
                observation_date=date(2021, 3, 1),
                data_provenance=DataProvenance.CENSUS_DERIVED,
            )

    def test_percent_metric_in_bounds_accepted(self):
        rec = EpiMetricRecord(
            metric_type=MetricType.SEROPREVALENCE_PCT,
            value=42.5,
            region="US-CA",
            observation_date=date(2021, 3, 1),
            data_provenance=DataProvenance.CENSUS_DERIVED,
        )
        self.assertEqual(rec.value, 42.5)

    def test_confidence_interval_order_enforced(self):
        with self.assertRaises(ValidationError):
            EpiMetricRecord(
                metric_type=MetricType.SEROPREVALENCE_PCT,
                value=42.5,
                region="US-CA",
                observation_date=date(2021, 3, 1),
                data_provenance=DataProvenance.CENSUS_DERIVED,
                confidence_interval=(0.5, 0.2),
            )

    def test_confidence_interval_valid_order_accepted(self):
        rec = EpiMetricRecord(
            metric_type=MetricType.SEROPREVALENCE_PCT,
            value=42.5,
            region="US-CA",
            observation_date=date(2021, 3, 1),
            data_provenance=DataProvenance.CENSUS_DERIVED,
            confidence_interval=(0.35, 0.50),
        )
        self.assertEqual(rec.confidence_interval, (0.35, 0.50))

    def test_demographic_strata_requires_a_dimension(self):
        with self.assertRaises(ValidationError):
            DemographicStrata()

    def test_demographic_strata_accepts_partial_breakdown(self):
        strata = DemographicStrata(age_group="18-29")
        self.assertEqual(strata.age_group, "18-29")
        self.assertIsNone(strata.region)


if __name__ == "__main__":
    unittest.main()
