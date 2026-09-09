"""
mdas/api/routers/ingestion.py — HTTP front door for the two ingestion pipelines.

Each endpoint accepts an uploaded file (CSV/Parquet/JSON — whatever
load_tabular_records already understands) plus the same column-mapping
parameters the ingestion pipelines' records_from_rows() functions take as
keyword arguments. The file is spooled to a temp path, read via
load_tabular_records (the same loader ingest_timeseries()/
ingest_census_snapshot() use for local files), then handed to
records_from_rows() with `source` set to the original upload filename rather
than the throwaway temp path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile

from mdas.ingestion.base import load_tabular_records
from mdas.ingestion.census import records_from_rows as census_records_from_rows
from mdas.ingestion.timeseries import records_from_rows as timeseries_records_from_rows
from mdas.schemas import EpiMetricRecord, MetricType

router = APIRouter(prefix="/ingest", tags=["ingestion"])


async def _spool_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await file.read())
    tmp.close()
    return Path(tmp.name)


@router.post("/timeseries", response_model=List[EpiMetricRecord])
async def ingest_timeseries_file(
    file: UploadFile = File(...),
    metric_type: MetricType = Form(...),
    pathogen: str = Form("SARS-CoV-2"),
    region_field: str = Form("region"),
    date_field: str = Form("date"),
    value_field: str = Form("value"),
    unit: Optional[str] = Form(None),
) -> List[EpiMetricRecord]:
    """Continuous-feed upload (daily case counts, doses administered, ...)."""
    path = await _spool_upload(file)
    try:
        rows = load_tabular_records(path)
        return timeseries_records_from_rows(
            rows,
            metric_type=metric_type,
            pathogen=pathogen,
            region_field=region_field,
            date_field=date_field,
            value_field=value_field,
            unit=unit,
            source=file.filename,
        )
    finally:
        path.unlink(missing_ok=True)


@router.post("/census", response_model=List[EpiMetricRecord])
async def ingest_census_file(
    file: UploadFile = File(...),
    metric_type: MetricType = Form(...),
    pathogen: str = Form("SARS-CoV-2"),
    region_field: str = Form("region"),
    date_field: str = Form("date"),
    value_field: str = Form("value"),
    unit: Optional[str] = Form(None),
    ci_lower_field: Optional[str] = Form(None),
    ci_upper_field: Optional[str] = Form(None),
    age_group_field: Optional[str] = Form(None),
    region_detail_field: Optional[str] = Form(None),
    census_tract_field: Optional[str] = Form(None),
) -> List[EpiMetricRecord]:
    """Census/survey snapshot upload (seroprevalence, testing coverage, ...)."""
    path = await _spool_upload(file)
    try:
        rows = load_tabular_records(path)
        return census_records_from_rows(
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
            source=file.filename,
        )
    finally:
        path.unlink(missing_ok=True)
