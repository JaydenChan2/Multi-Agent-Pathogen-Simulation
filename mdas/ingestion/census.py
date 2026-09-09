"""
mdas/ingestion/census.py — Census/survey snapshot pipeline (periodic, low-frequency).

Handles data that arrives as an infrequent snapshot table rather than a daily
feed: vaccination-coverage surveys, seroprevalence studies, self-reported
prior-infection surveys, demographic census extracts. These sources
routinely carry sampling uncertainty and strata breakdowns that a daily API
feed does not, so — unlike mdas/ingestion/timeseries.py — this pipeline
understands confidence_interval and demographic_strata columns.

Every record produced here is tagged CENSUS_DERIVED: it was measured, but
via a sample, not a census-of-one registry count.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from mdas.ingestion.base import load_tabular_records
from mdas.schemas import DataProvenance, DemographicStrata, EpiMetricRecord, MetricType


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def _extract_strata(row: Dict[str, Any], *, age_group_field, region_detail_field, tract_field) -> Optional[DemographicStrata]:
    kwargs = {}
    if age_group_field and row.get(age_group_field) not in (None, ""):
        kwargs["age_group"] = str(row[age_group_field])
    if region_detail_field and row.get(region_detail_field) not in (None, ""):
        kwargs["region"] = str(row[region_detail_field])
    if tract_field and row.get(tract_field) not in (None, ""):
        kwargs["census_tract"] = str(row[tract_field])
    return DemographicStrata(**kwargs) if kwargs else None


def _extract_ci(
    row: Dict[str, Any], *, ci_lower_field: Optional[str], ci_upper_field: Optional[str]
) -> Optional[Tuple[float, float]]:
    if not ci_lower_field or not ci_upper_field:
        return None
    lower, upper = row.get(ci_lower_field), row.get(ci_upper_field)
    if lower is None or upper is None:
        return None
    return (float(lower), float(upper))


def records_from_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    metric_type: MetricType,
    pathogen: str = "SARS-CoV-2",
    region_field: str = "region",
    date_field: str = "date",
    value_field: str = "value",
    unit: Optional[str] = None,
    ci_lower_field: Optional[str] = None,
    ci_upper_field: Optional[str] = None,
    age_group_field: Optional[str] = None,
    region_detail_field: Optional[str] = None,
    census_tract_field: Optional[str] = None,
    source: Optional[str] = None,
) -> List[EpiMetricRecord]:
    """Map already-loaded census/survey rows onto CENSUS_DERIVED EpiMetricRecords."""
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
                data_provenance=DataProvenance.CENSUS_DERIVED,
                confidence_interval=_extract_ci(
                    row, ci_lower_field=ci_lower_field, ci_upper_field=ci_upper_field
                ),
                demographic_strata=_extract_strata(
                    row,
                    age_group_field=age_group_field,
                    region_detail_field=region_detail_field,
                    tract_field=census_tract_field,
                ),
                source=source,
            )
        )
    return out


def ingest_census_snapshot(
    source: Union[str, Path],
    *,
    metric_type: MetricType,
    pathogen: str = "SARS-CoV-2",
    region_field: str = "region",
    date_field: str = "date",
    value_field: str = "value",
    unit: Optional[str] = None,
    ci_lower_field: Optional[str] = None,
    ci_upper_field: Optional[str] = None,
    age_group_field: Optional[str] = None,
    region_detail_field: Optional[str] = None,
    census_tract_field: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    verbose: bool = False,
) -> List[EpiMetricRecord]:
    """
    Ingest a census/survey snapshot from a CSV/Parquet/JSON file or API URL.

    Column-mapping parameters are all optional except the required value/
    region/date triple; leave ci_*/age_group_field/etc. as None when the
    source table doesn't carry that column.
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
        ci_lower_field=ci_lower_field,
        ci_upper_field=ci_upper_field,
        age_group_field=age_group_field,
        region_detail_field=region_detail_field,
        census_tract_field=census_tract_field,
        source=str(source),
    )
