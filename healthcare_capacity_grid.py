#!/usr/bin/env python3
"""
healthcare_capacity_grid.py — Regional healthcare-adequacy grid for MAPS.

Standalone module. Does not import from, or get imported by, any other
script in this repository — call build_healthcare_capacity_grid() directly
wherever a healthcare-capacity multiplier/covariate grid is needed.

What it does
------------
1. Ingests a facility-level dataset (CMS Hospital General Information,
   HIFLD Hospitals, or any similarly-shaped table) from either a local
   CSV/Parquet file or a Socrata/CMS-style JSON API endpoint. API responses
   are cached to disk so repeat runs never re-hit the network.
2. For each facility, computes a capacity-weighted quality score:
       score = rating × log10(beds + 1)
   so a 5-star, 40-bed clinic does not outweigh a 3-star, 800-bed trauma
   center — bed count acts as a scale/resilience multiplier on the rating.
3. Imputes missing/suppressed ratings and bed counts (common for small and
   critical-access hospitals) using the state median, falling back to the
   national median, falling back to a neutral constant — so unrated
   facilities still contribute to the map instead of being dropped.
4. Rasterizes facility scores onto the reference lat/lon grid and applies a
   Gaussian distance-decay kernel (scipy.ndimage.gaussian_filter) sized in
   real kilometers via catchment_radius_km, so a regional hospital's
   influence spreads smoothly into surrounding rural/suburban cells instead
   of only lighting up the single cell it happens to sit in.

WEIGHT — every point where a factor's magnitude is scaled
-----------------------------------------------------------------
WEIGHT 1 — bed-scale weighting of the quality score, inside
    _prepare_facility_scores(): score = rating × log10(beds + 1). Using
    log10 rather than a raw bed count keeps a 2000-bed hub from swamping the
    grid by three orders of magnitude relative to a 20-bed clinic — it
    still wins, just proportionally rather than absolutely.
WEIGHT 2 — catchment_radius_km controls the Gaussian sigma (in grid cells)
    used to spread each facility's score to nearby cells; larger values
    model broader regional catchments (e.g. a Level I trauma center)
    at the cost of blurring finer local detail.

Requires numpy always; pandas/scipy/requests are imported lazily and only
required for the code paths that actually use them (mirrors the optional-
import pattern used elsewhere in this codebase, e.g. meteorological_factor.py).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KM_PER_DEG_LAT: float = 111.32  # WGS84 mean meridional degree length
_DEFAULT_CACHE_DIR = Path(".healthcare_capacity_cache")

# Percentile used to rescale the smoothed capacity field to [0, 1]. Using a
# high percentile rather than the raw max prevents a single mega-cluster
# (e.g. a dense hospital row in a major metro) from crushing every other
# region's normalized value toward zero.
_CAPACITY_NORM_PERCENTILE: float = 99.0


# ---------------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------------

def _ensure_monotonic(axis: np.ndarray, name: str) -> None:
    """Raise ValueError unless axis is strictly monotonic (increasing or decreasing)."""
    if axis.size < 2:
        return
    diffs = np.diff(axis)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            f"{name} must be strictly monotonic (increasing or decreasing) "
            "for nearest-cell rasterization and km/degree spacing to be well-defined."
        )


def _normalize_grid(
    lat_grid: np.ndarray, lon_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Reduce lat_grid/lon_grid (1-D or 2-D) to 1-D axes plus the target output shape.

    2-D inputs are assumed to be a rectilinear meshgrid built with
    indexing="ij" (lat varies along axis 0, lon along axis 1) — the same
    convention used elsewhere in this pipeline (see meteorological_factor.py's
    interpolate_to_maps_grid). Axis values are taken from the first row/column.
    """
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)

    if lat_grid.ndim == 1 and lon_grid.ndim == 1:
        lat_axis, lon_axis = lat_grid, lon_grid
        out_shape = (lat_axis.size, lon_axis.size)
    elif lat_grid.ndim == 2 and lon_grid.ndim == 2:
        if lat_grid.shape != lon_grid.shape:
            raise ValueError(
                f"lat_grid shape {lat_grid.shape} and lon_grid shape "
                f"{lon_grid.shape} must match when both are 2-D."
            )
        lat_axis, lon_axis = lat_grid[:, 0], lon_grid[0, :]
        out_shape = lat_grid.shape
    else:
        raise ValueError("lat_grid and lon_grid must both be 1-D or both be 2-D.")

    _ensure_monotonic(lat_axis, "lat_grid")
    _ensure_monotonic(lon_axis, "lon_grid")
    return lat_axis, lon_axis, out_shape


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map each value to the index of its nearest entry in a monotonic axis."""
    if axis.size == 1:
        return np.zeros(values.shape, dtype=np.int64)

    ascending = axis[1] > axis[0]
    a = axis if ascending else axis[::-1]
    pos = np.searchsorted(a, values)
    pos = np.clip(pos, 1, a.size - 1)
    left_dist = np.abs(values - a[pos - 1])
    right_dist = np.abs(values - a[pos])
    nearest = np.where(left_dist <= right_dist, pos - 1, pos)
    if not ascending:
        nearest = axis.size - 1 - nearest
    return nearest


def _km_per_cell(lat_axis: np.ndarray, lon_axis: np.ndarray) -> Tuple[float, float]:
    """
    Convert grid spacing (degrees) to approximate real-world spacing (km),
    evaluated at the grid's mean latitude for the longitude component.
    """
    dlat_deg = float(np.median(np.abs(np.diff(lat_axis)))) if lat_axis.size > 1 else 0.0
    dlon_deg = float(np.median(np.abs(np.diff(lon_axis)))) if lon_axis.size > 1 else 0.0

    mean_lat = float(np.mean(lat_axis))
    km_per_lat_cell = dlat_deg * _KM_PER_DEG_LAT
    km_per_lon_cell = dlon_deg * _KM_PER_DEG_LAT * max(np.cos(np.radians(mean_lat)), 1e-6)
    return km_per_lat_cell, km_per_lon_cell


def _rasterize_points(
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    fac_lat: np.ndarray,
    fac_lon: np.ndarray,
    fac_score: np.ndarray,
) -> np.ndarray:
    """Accumulate facility scores into their nearest grid cell (co-located facilities sum)."""
    raster = np.zeros((lat_axis.size, lon_axis.size), dtype=np.float64)
    iy = _nearest_index(lat_axis, fac_lat)
    ix = _nearest_index(lon_axis, fac_lon)
    np.add.at(raster, (iy, ix), fac_score)
    return raster


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def _hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _load_from_api(
    url: str,
    cache_dir: Path,
    force_refresh: bool,
    timeout_s: float,
    verbose: bool,
) -> "pd.DataFrame":
    """
    Fetch a Socrata/CMS-style JSON payload, caching the raw response to disk.

    A cached payload is reused whenever present (unless force_refresh=True),
    which keeps repeated runs fast and offline-safe. A network failure with
    a stale cache present falls back to that cache rather than raising.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_hash_key(url)}.json"

    payload = None
    if cache_path.exists() and not force_refresh:
        if verbose:
            print(f"[healthcare_capacity_grid] using cached payload: {cache_path}", file=sys.stderr)
        payload = json.loads(cache_path.read_text())
    else:
        if not _REQUESTS_AVAILABLE:
            if cache_path.exists():
                if verbose:
                    print(
                        "[healthcare_capacity_grid] 'requests' not installed; "
                        f"falling back to stale cache at {cache_path}",
                        file=sys.stderr,
                    )
                payload = json.loads(cache_path.read_text())
            else:
                raise RuntimeError(
                    "'requests' is required to fetch data_source from an API endpoint, "
                    f"and no local cache exists at {cache_path}.\n"
                    "Install with: pip install requests"
                )
        else:
            try:
                resp = _requests.get(url, timeout=timeout_s)
                resp.raise_for_status()
                payload = resp.json()
                cache_path.write_text(json.dumps(payload))
                if verbose:
                    print(f"[healthcare_capacity_grid] fetched and cached: {cache_path}", file=sys.stderr)
            except Exception as exc:
                if cache_path.exists():
                    if verbose:
                        print(
                            f"[healthcare_capacity_grid] API request failed ({exc}); "
                            "falling back to stale cache",
                            file=sys.stderr,
                        )
                    payload = json.loads(cache_path.read_text())
                else:
                    raise RuntimeError(
                        f"Failed to fetch data_source from '{url}' and no local cache "
                        f"is available: {exc}"
                    ) from exc

    if not _PANDAS_AVAILABLE:
        raise RuntimeError(
            "pandas is required to parse tabular facility data.\nInstall with: pip install pandas"
        )
    return pd.json_normalize(payload)


def _load_from_file(path_str: str) -> "pd.DataFrame":
    if not _PANDAS_AVAILABLE:
        raise RuntimeError(
            "pandas is required to load facility data from CSV/Parquet.\n"
            "Install with: pip install pandas"
        )
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"data_source file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _load_facility_data(
    data_source: str,
    cache_dir: Optional[Union[str, Path]],
    force_refresh: bool,
    api_timeout_s: float,
    verbose: bool,
) -> "pd.DataFrame":
    if data_source.startswith("http://") or data_source.startswith("https://"):
        resolved_cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        return _load_from_api(data_source, resolved_cache_dir, force_refresh, api_timeout_s, verbose)
    return _load_from_file(data_source)


# ---------------------------------------------------------------------------
# Rating/bed cleaning, imputation, and capacity-weighted scoring
# ---------------------------------------------------------------------------

def _impute_grouped_median(
    values: "pd.Series", groups: Optional["pd.Series"], fallback: float
) -> "pd.Series":
    """Fill NaNs with the group (e.g. state) median, then the global median, then a fixed fallback."""
    result = values.copy()
    if groups is not None:
        group_median = values.groupby(groups).transform("median")
        result = result.fillna(group_median)
    global_median = values.median()
    if pd.notna(global_median):
        result = result.fillna(global_median)
    return result.fillna(fallback)


def _prepare_facility_scores(
    df: "pd.DataFrame",
    lat_col: str,
    lon_col: str,
    rating_col: str,
    beds_col: str,
    state_col: Optional[str],
    default_rating_fallback: float,
    default_beds_fallback: float,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = [lat_col, lon_col, rating_col, beds_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"data_source is missing required column(s) {missing}. "
            f"Available columns: {list(df.columns)[:40]}"
        )

    work = df.copy()
    for col in (lat_col, lon_col, rating_col, beds_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    before = len(work)
    work = work[np.isfinite(work[lat_col]) & np.isfinite(work[lon_col])]
    dropped = before - len(work)
    if dropped and verbose:
        print(
            f"[healthcare_capacity_grid] dropped {dropped} facilities with missing/invalid coordinates",
            file=sys.stderr,
        )
    if work.empty:
        raise ValueError("No facilities with valid coordinates remain after cleaning data_source.")

    groups = work[state_col] if (state_col and state_col in work.columns) else None
    if state_col and groups is None and verbose:
        print(
            f"[healthcare_capacity_grid] state_col='{state_col}' not found; "
            "imputing ratings/beds from the national median only",
            file=sys.stderr,
        )

    n_rating_missing = int(work[rating_col].isna().sum())
    n_beds_missing = int(work[beds_col].isna().sum())
    work[rating_col] = _impute_grouped_median(work[rating_col], groups, default_rating_fallback)
    work[beds_col] = _impute_grouped_median(work[beds_col], groups, default_beds_fallback).clip(lower=0.0)
    if verbose and (n_rating_missing or n_beds_missing):
        print(
            f"[healthcare_capacity_grid] imputed {n_rating_missing} missing ratings and "
            f"{n_beds_missing} missing bed counts",
            file=sys.stderr,
        )

    # Capacity-weighted quality: a rating alone favors small elective clinics
    # over large trauma centers, so scale it by facility size (WEIGHT 1 above).
    score = work[rating_col].to_numpy(dtype=np.float64) * np.log10(
        work[beds_col].to_numpy(dtype=np.float64) + 1.0
    )
    score = np.clip(score, 0.0, None)

    return (
        work[lat_col].to_numpy(dtype=np.float64),
        work[lon_col].to_numpy(dtype=np.float64),
        score,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_healthcare_capacity_grid(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    data_source: str,
    catchment_radius_km: float = 35.0,
    rating_col: str = "hospital_overall_rating",
    beds_col: str = "number_of_licensed_beds",
    default_rating_fallback: float = 3.0,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    state_col: Optional[str] = "state",
    default_beds_fallback: float = 25.0,
    normalize_output: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    force_refresh: bool = False,
    api_timeout_s: float = 30.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build a 2-D healthcare-capacity/adequacy grid from facility-level data.

    Parameters
    ----------
    lat_grid, lon_grid : reference grid coordinates, either both 1-D
        (independent axes) or both 2-D (a rectilinear meshgrid built with
        indexing="ij", i.e. lat varies along axis 0 and lon along axis 1 —
        the convention used elsewhere in this pipeline).
    data_source : local CSV/Parquet path, or an HTTP(S) API endpoint
        (e.g. a CMS Socrata dataset URL). API responses are cached to
        `cache_dir` so repeat runs don't re-hit the network.
    catchment_radius_km : approximate radius (km) of a facility's regional
        service area; controls the Gaussian smoothing kernel that spreads
        each facility's capacity-weighted score into nearby grid cells
        (WEIGHT 2 above).
    rating_col, beds_col : column names for the official quality rating
        (e.g. CMS 1-5 star "hospital_overall_rating") and licensed bed
        count. Both are coerced to numeric; non-numeric entries (e.g.
        "Not Available") become NaN and are imputed.
    default_rating_fallback : neutral rating used only when a facility's
        rating is missing AND no state/national median is computable.
    lat_col, lon_col : facility coordinate column names in data_source.
    state_col : optional column used to group median-imputation of missing
        ratings/beds regionally before falling back to the national median.
        Pass None to always use the national median.
    default_beds_fallback : neutral bed count used only when a facility's
        bed count is missing AND no state/national median is computable.
    normalize_output : rescale the smoothed grid to [0, 1] using a high
        percentile (not the max) of the positive cells, so a single dense
        metro cluster doesn't crush every other region toward zero. If
        False, the raw smoothed capacity-weighted score density is returned.
    cache_dir : directory for caching API responses (only used when
        data_source is a URL). Defaults to ./.healthcare_capacity_cache.
    force_refresh : re-fetch from the API even if a cache entry exists.
    api_timeout_s : timeout (seconds) for the API request.
    verbose : print progress/diagnostics to stderr.

    Returns
    -------
    np.ndarray, float32, shape matching lat_grid/lon_grid — higher values
    indicate greater nearby capacity-weighted healthcare adequacy.

    Raises
    ------
    RuntimeError  if scipy (always), or pandas/requests (only on the code
        paths that need them), are not installed.
    ValueError    if lat_grid/lon_grid are malformed, required columns are
        missing from data_source, or no facility has valid coordinates.
    FileNotFoundError  if data_source is a local path that does not exist.
    """
    if not _SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for catchment-area Gaussian smoothing.\nInstall with: pip install scipy"
        )

    lat_axis, lon_axis, out_shape = _normalize_grid(lat_grid, lon_grid)

    raw_df = _load_facility_data(data_source, cache_dir, force_refresh, api_timeout_s, verbose)
    if raw_df.empty:
        raise ValueError(f"data_source '{data_source}' loaded zero facility records.")
    if verbose:
        print(
            f"[healthcare_capacity_grid] loaded {len(raw_df)} facility records from {data_source}",
            file=sys.stderr,
        )

    fac_lat, fac_lon, fac_score = _prepare_facility_scores(
        raw_df, lat_col, lon_col, rating_col, beds_col, state_col,
        default_rating_fallback, default_beds_fallback, verbose,
    )

    raster = _rasterize_points(lat_axis, lon_axis, fac_lat, fac_lon, fac_score)

    km_per_lat_cell, km_per_lon_cell = _km_per_cell(lat_axis, lon_axis)
    sigma_lat = catchment_radius_km / km_per_lat_cell if km_per_lat_cell > 0 else 0.0
    sigma_lon = catchment_radius_km / km_per_lon_cell if km_per_lon_cell > 0 else 0.0

    smoothed = _gaussian_filter(raster, sigma=(sigma_lat, sigma_lon), mode="nearest")

    if normalize_output:
        positive = smoothed[smoothed > 0]
        if positive.size > 0:
            scale = float(np.percentile(positive, _CAPACITY_NORM_PERCENTILE))
            if scale <= 0:
                scale = float(smoothed.max())
        else:
            scale = 0.0
        if scale > 0:
            smoothed = np.clip(smoothed / scale, 0.0, 1.0)

    result = smoothed.astype(np.float32)
    assert result.shape == out_shape, (
        f"internal error: output shape {result.shape} does not match reference grid shape {out_shape}"
    )
    return result


# ---------------------------------------------------------------------------
# Offline self-test (no network, no pre-existing files required)
# ---------------------------------------------------------------------------

def _self_test_demo() -> None:
    """Exercise the full pipeline end-to-end on a small synthetic facility table."""
    if not (_PANDAS_AVAILABLE and _SCIPY_AVAILABLE):
        print(
            "[healthcare_capacity_grid] self-test requires pandas and scipy; "
            "install with: pip install pandas scipy",
            file=sys.stderr,
        )
        return

    import tempfile

    synthetic = pd.DataFrame({
        "facility_name": [
            "Big Metro Trauma Center", "Rural Critical Access",
            "Suburban Community", "Unrated Clinic",
        ],
        "latitude": [40.71, 41.05, 40.85, 40.20],
        "longitude": [-74.00, -73.40, -73.95, -74.50],
        "state": ["NY", "NY", "NY", "NJ"],
        "hospital_overall_rating": [5, np.nan, 3, np.nan],
        "number_of_licensed_beds": [1200, 25, 180, np.nan],
    })

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "synthetic_facilities.csv"
        synthetic.to_csv(csv_path, index=False)

        lat_grid = np.linspace(39.8, 41.3, 16)
        lon_grid = np.linspace(-74.8, -73.2, 16)

        grid = build_healthcare_capacity_grid(
            lat_grid, lon_grid, str(csv_path),
            catchment_radius_km=35.0,
            verbose=True,
        )

    print(
        f"\n[healthcare_capacity_grid] self-test grid shape={grid.shape} "
        f"min={grid.min():.3f} max={grid.max():.3f} mean={grid.mean():.3f}"
    )


if __name__ == "__main__":
    _self_test_demo()
