"""
mdas/ingestion/timeseries.py — Continuous-feed pipeline (daily APIs/registries).

Handles data that arrives as a dense time series: daily case counts,
hospitalizations, vaccine doses administered, from public health registry
APIs or their CSV/JSON exports. Every record produced here is tagged
MEASURED_DIRECT — this pipeline never guesses, it only reshapes.

For periodic survey/census snapshots (vaccination coverage surveys,
seroprevalence studies, self-reported infection history) use
mdas/ingestion/census.py instead, which tags CENSUS_DERIVED and understands
confidence intervals and demographic strata.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from mdas.ingestion.base import load_tabular_records
from mdas.schemas import DataProvenance, EpiMetricRecord, MetricType


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def records_from_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    metric_type: MetricType,
    pathogen: str = "SARS-CoV-2",
    region_field: str = "region",
    date_field: str = "date",
    value_field: str = "value",
    unit: Optional[str] = None,
    source: Optional[str] = None,
) -> List[EpiMetricRecord]:
    """Map already-loaded rows (dicts) onto MEASURED_DIRECT EpiMetricRecords."""
    out: List[EpiMetricRecord] = []
    for row in rows:
        out.append(
            EpiMetricRecord(
                pathogen=pathogen,
                metric_type=metric_type,
                value=float(row[value_field]),
                unit=unit,
                region=str(row[region_field]),
                observation_date=_parse_date(row[date_field]),
                data_provenance=DataProvenance.MEASURED_DIRECT,
                source=source,
            )
        )
    return out


def ingest_timeseries(
    source: Union[str, Path],
    *,
    metric_type: MetricType,
    pathogen: str = "SARS-CoV-2",
    region_field: str = "region",
    date_field: str = "date",
    value_field: str = "value",
    unit: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    verbose: bool = False,
) -> List[EpiMetricRecord]:
    """
    Ingest a continuous time-series feed from a CSV/Parquet/JSON file or API URL.

    `source` is resolved by mdas.ingestion.base.load_tabular_records — a
    local path is read directly; an http(s):// URL is fetched and cached.
    """
    rows = load_tabular_records(
        source, cache_dir=cache_dir, force_refresh=force_refresh, verbose=verbose
    )
    return records_from_rows(
        rows,
        metric_type=metric_type,
        pathogen=pathogen,
        region_field=region_field,
        date_field=date_field,
        value_field=value_field,
        unit=unit,
        source=str(source),
    )
