"""
tests/test_mdas_ingestion.py — Continuous-feed and census-tabular ingestion.

Run with: pytest tests/test_mdas_ingestion.py -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mdas.ingestion.census import ingest_census_snapshot
from mdas.ingestion.timeseries import ingest_timeseries
from mdas.schemas import DataProvenance, MetricType


class TestTimeseriesIngestionFromCSV(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmpdir.name) / "daily_cases.csv"
        self.csv_path.write_text(
            "region,date,value\n"
            "US-CA,2021-03-01,120\n"
            "US-CA,2021-03-02,150\n"
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parses_rows_as_measured_direct(self):
        records = ingest_timeseries(self.csv_path, metric_type=MetricType.CUMULATIVE_CASES)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r.data_provenance == DataProvenance.MEASURED_DIRECT for r in records))
        self.assertEqual(records[0].region, "US-CA")
        self.assertEqual(records[1].value, 150.0)


class TestCensusIngestionFromCSV(unittest.TestCase):
    """Parser handling of census-style tabular data (requirement 5, bullet 1)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmpdir.name) / "vaccination_survey.csv"
        self.csv_path.write_text(
            "region,date,value,ci_lower,ci_upper,age_group\n"
            "US-CA,2021-06-01,62.5,58.0,67.0,18-29\n"
            "US-TX,2021-06-01,45.0,40.0,50.0,18-29\n"
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parses_census_rows_with_ci_and_strata(self):
        records = ingest_census_snapshot(
            self.csv_path,
            metric_type=MetricType.SEROPREVALENCE_PCT,
            ci_lower_field="ci_lower",
            ci_upper_field="ci_upper",
            age_group_field="age_group",
        )
        self.assertEqual(len(records), 2)
        ca = records[0]
        self.assertEqual(ca.data_provenance, DataProvenance.CENSUS_DERIVED)
        self.assertEqual(ca.confidence_interval, (58.0, 67.0))
        self.assertIsNotNone(ca.demographic_strata)
        self.assertEqual(ca.demographic_strata.age_group, "18-29")

    def test_missing_ci_columns_yield_none_interval(self):
        records = ingest_census_snapshot(
            self.csv_path, metric_type=MetricType.SEROPREVALENCE_PCT
        )
        self.assertIsNone(records[0].confidence_interval)
        self.assertIsNone(records[0].demographic_strata)


class TestJSONIngestion(unittest.TestCase):
    """Ensure raw JSON responses (dict-wrapped or bare list) are also supported."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_bare_list_json(self):
        path = Path(self.tmpdir.name) / "doses.json"
        path.write_text(
            '[{"region": "US-CA", "date": "2021-04-01", "value": 500000}]'
        )
        records = ingest_timeseries(path, metric_type=MetricType.VACCINE_DOSES_TOTAL)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].value, 500000.0)

    def test_dict_wrapped_json(self):
        path = Path(self.tmpdir.name) / "doses_wrapped.json"
        path.write_text(
            '{"data": [{"region": "US-CA", "date": "2021-04-01", "value": 500000}]}'
        )
        records = ingest_timeseries(path, metric_type=MetricType.VACCINE_DOSES_TOTAL)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].region, "US-CA")


if __name__ == "__main__":
    unittest.main()
