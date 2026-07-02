#!/usr/bin/env python3
"""
meteorological_factor.py — Weather-derived beta multiplier grid for MAPS.

Pipeline position
-----------------
Run BEFORE the IC pre-processor for each forecast cycle:

    meteorological_factor.py          ← THIS SCRIPT  →  maps_meteo_<date>.nc
    build_maps_initial_conditions_with_history_full.py  (IC pre-processor)
    edit_namelist.py                                    (per-member editor)
    Fortran MAPS model

What it does
------------
1. Reads the lat/lon reference grid from a MAPS population NetCDF (or derives
   it from the namelist) so the output grid is pixel-aligned with the IC file.
2. Samples daily temperature (°C), relative humidity (%), and UV index from the
   Open-Meteo API at a coarser resolution (default 1°), then bilinearly
   interpolates to the full MAPS grid via scipy.
3. Converts T + RH to absolute humidity (g m⁻³) with the Magnus formula, then
   combines all three variables into a dimensionless meteo_beta_factor:
       factor = w_AH*f(AH) + w_UV*f(UV) + w_T*f(T)   (normalised sum)
   Values > 1 favour transmission; < 1 suppress it.
4. Writes a NetCDF file with the exact structure expected by read_grid_and_var()
   in the IC pre-processor:
       Dimensions : lat, lon
       Variables  : lat [f8,(lat,)], lon [f8,(lon,)],
                    meteo_beta_factor [f4,(lat,lon), zlib]
   NaN cells are pre-filled with 1.0 (neutral) so the reader's NaN→0 fill
   does not accidentally zero out beta.

Pass --dry-run for format-only testing (no network calls, no scipy required).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from netCDF4 import Dataset

# Optional imports needed only for real (non-dry-run) mode.
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from scipy.interpolate import RegularGridInterpolator as _RGI
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Physical constants and reference values
# ---------------------------------------------------------------------------

# Reference conditions at which each sub-factor evaluates to exactly 1.0.
_AH_REF_G_M3: float = 7.0    # annual US mean absolute humidity
_UV_REF: float = 3.0          # moderate UV (spring/autumn)
_T_REF_C: float = 10.0        # cool temperate baseline

# Linear sensitivities (factor change per unit of variable above reference).
_AH_SLOPE: float = 0.040      # per g m⁻³
_UV_SLOPE: float = 0.025      # per UV index unit
_T_SLOPE: float = 0.010       # per °C

# Per-cell clamp applied before writing.
_FACTOR_MIN: float = 0.50
_FACTOR_MAX: float = 1.50

# US continental bounding box used when the MAPS grid is larger than the US.
_US_LAT_MIN, _US_LAT_MAX = 24.0, 50.0 
_US_LON_MIN, _US_LON_MAX = -126.0, -65.0


# Switch from archive to forecast endpoint when init_date is this many days
# in the past or fewer (Open-Meteo archive has a ~5-day processing lag).
_ARCHIVE_LAG_DAYS: int = 5


# ---------------------------------------------------------------------------
# Utility: namelist parser (mirrors parse_simple_namelist in the IC builder)
# ---------------------------------------------------------------------------

def parse_simple_namelist(path: Path) -> dict:
    """
    Parse the &init_nml block from a Fortran-style namelist file.

    Returns a dict of lowercase keys mapped to Python booleans, ints, floats,
    or strings. Identical logic to the IC pre-processor so the same .nml
    file drives both scripts without modification.
    """
    entries: dict = {}
    inside = False
    for raw in path.read_text().splitlines():
        line = raw.split("!")[0].strip()
        if not line:
            continue
        if line.lower().startswith("&init_nml"):
            inside = True
            continue
        if line.strip() == "/":
            break
        if not inside or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().lower()
        val = val.strip().rstrip(",")
        if val.lower() in (".true.", "true"):
            entries[key] = True
        elif val.lower() in (".false.", "false"):
            entries[key] = False
        elif (val.startswith("'") and val.endswith("'")) or (
            val.startswith('"') and val.endswith('"')
        ):
            entries[key] = val[1:-1]
        else:
            try:
                entries[key] = float(val) if ("." in val or "e" in val.lower()) else int(val)
            except Exception:
                entries[key] = val
    return entries


def read_reference_latlon(nc_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the 1-D lat and lon coordinate arrays from any MAPS grid NetCDF.

    The file need not be the population grid specifically; any file produced
    by the IC pipeline (population, state-ID, or a prior meteo file) will
    work as long as it has lat and lon variables.
    """
    with Dataset(nc_path, "r") as ds:
        lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        lon = np.array(ds.variables["lon"][:], dtype=np.float64)
    return lat, lon


# ---------------------------------------------------------------------------
# Meteorological physics
# ---------------------------------------------------------------------------

def compute_absolute_humidity(T_celsius: float, RH_pct: float) -> float:
    """
    Convert temperature (°C) and relative humidity (%) to absolute humidity (g m⁻³).

    Uses the Magnus saturation vapour-pressure formula and the ideal gas
    approximation for water vapour density. Returns the result in g m⁻³.
    """
    es_hpa = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))
    e_hpa = (RH_pct / 100.0) * es_hpa
    return float(216.7 * e_hpa / (273.15 + T_celsius))


def _clamp(x: float) -> float:
    """Clamp x to the [_FACTOR_MIN, _FACTOR_MAX] range."""
    return max(_FACTOR_MIN, min(_FACTOR_MAX, x))


def f_humidity(AH: float) -> float:
    """
    Sub-factor for absolute humidity.

    Decreases linearly above the reference AH (low humidity → longer aerosol
    survival → higher transmission). Centred at 1.0 when AH = _AH_REF_G_M3.
    """
    return _clamp(1.0 - _AH_SLOPE * (AH - _AH_REF_G_M3))


def f_uv(UV: float) -> float:
    """
    Sub-factor for UV index.

    Higher UV suppresses transmission via virucidal photodegradation.
    Centred at 1.0 when UV = _UV_REF.
    """
    return _clamp(1.0 - _UV_SLOPE * (UV - _UV_REF))


def f_temp(T: float) -> float:
    """
    Sub-factor for temperature (°C).

    Warmer temperatures reduce indoor crowding and accelerate viral decay on
    surfaces. Centred at 1.0 when T = _T_REF_C.
    """
    return _clamp(1.0 - _T_SLOPE * (T - _T_REF_C))


def compute_meteo_factor(
    AH: float,
    UV: float,
    T: float,
    w_ah: float = 0.50,
    w_uv: float = 0.30,
    w_t: float = 0.20,
) -> float:
    """
    Combine the three meteorological sub-factors into a single dimensionless
    meteo_beta_factor for one grid cell.

    Parameters
    ----------
    AH    : absolute humidity in g m⁻³
    UV    : daily-maximum UV index
    T     : daily-mean temperature in °C
    w_ah, w_uv, w_t : relative weights (need not sum to 1; normalised internally)

    Returns a float in [_FACTOR_MIN, _FACTOR_MAX].
    """
    w_total = w_ah + w_uv + w_t
    if w_total <= 0.0:
        return 1.0
    raw = (w_ah * f_humidity(AH) + w_uv * f_uv(UV) + w_t * f_temp(T)) / w_total
    return _clamp(raw)


# ---------------------------------------------------------------------------
# Open-Meteo API
# ---------------------------------------------------------------------------

def _open_meteo_url(init_date: datetime) -> str:
    """
    Return the correct Open-Meteo base URL for the given date.

    Dates more than _ARCHIVE_LAG_DAYS in the past use the archive endpoint
    (ERA5-based reanalysis); more recent dates use the forecast endpoint.
    """
    lag = (datetime.utcnow().date() - init_date.date()).days
    if lag >= _ARCHIVE_LAG_DAYS:
        return "https://archive-api.open-meteo.com/v1/archive"
    return "https://api.open-meteo.com/v1/forecast"


def fetch_meteo_point(
    lat: float,
    lon: float,
    date_str: str,
    session: "requests.Session",  # type: ignore[name-defined]
) -> Optional[dict]:
    """
    Fetch daily-mean temperature, relative humidity, and UV index for one point.

    Parameters
    ----------
    lat, lon  : coordinates of the grid point
    date_str  : 'YYYY-MM-DD' string for the target date
    session   : an open requests.Session (caller manages lifecycle)

    Returns a dict {'T': float, 'RH': float, 'UV': float} or None on failure.
    The UV field falls back to _UV_REF when the API does not return it.
    """
    init_date = datetime.strptime(date_str, "%Y-%m-%d")
    base_url = _open_meteo_url(init_date)
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "daily": "temperature_2m_mean,relativehumidity_2m_mean,uv_index_max",
        "timezone": "UTC",
        "start_date": date_str,
        "end_date": date_str,
    }
    try:
        resp = session.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        T_list = daily.get("temperature_2m_mean", [None])
        RH_list = daily.get("relativehumidity_2m_mean", [None])
        UV_list = daily.get("uv_index_max", [None])
        T = T_list[0] if T_list else None
        RH = RH_list[0] if RH_list else None
        UV = UV_list[0] if UV_list else _UV_REF
        if T is None or RH is None:
            return None
        return {"T": float(T), "RH": float(RH), "UV": float(UV) if UV is not None else _UV_REF}
    except Exception:
        return None


def build_sample_latlons(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a coarse regular sampling grid covering the given bounding box.

    Returns 1-D lat and lon arrays separated by resolution_deg. Used as the
    sparse observation grid before bilinear interpolation to the MAPS grid.
    """
    sample_lats = np.arange(lat_min, lat_max + resolution_deg * 0.5, resolution_deg)
    sample_lons = np.arange(lon_min, lon_max + resolution_deg * 0.5, resolution_deg)
    return sample_lats, sample_lons


def fetch_factor_on_sample_grid(
    sample_lats: np.ndarray,
    sample_lons: np.ndarray,
    date_str: str,
    w_ah: float,
    w_uv: float,
    w_t: float,
    request_delay: float = 0.06,
) -> np.ndarray:
    """
    Fetch Open-Meteo data for every (lat, lon) in the sampling grid and compute
    meteo_beta_factor at each point.

    Returns a 2-D float64 array of shape (len(sample_lats), len(sample_lons)).
    Cells where the API returns no data are filled with 1.0 (neutral factor).
    """
    import requests  # deferred: only needed in non-dry-run mode

    ny, nx = len(sample_lats), len(sample_lons)
    factor_grid = np.ones((ny, nx), dtype=np.float64)
    session = requests.Session()
    total = ny * nx

    for count, (iy, ix) in enumerate(
        ((r, c) for r in range(ny) for c in range(nx)), start=1
    ):
        result = fetch_meteo_point(sample_lats[iy], sample_lons[ix], date_str, session)
        if result is not None:
            AH = compute_absolute_humidity(result["T"], result["RH"])
            factor_grid[iy, ix] = compute_meteo_factor(AH, result["UV"], result["T"],
                                                        w_ah, w_uv, w_t)
        if count % 50 == 0:
            print(f"  fetched {count}/{total} sample points …", file=sys.stderr)
        time.sleep(request_delay)

    session.close()
    return factor_grid


def interpolate_to_maps_grid(
    factor_coarse: np.ndarray,
    sample_lats: np.ndarray,
    sample_lons: np.ndarray,
    maps_lat: np.ndarray,
    maps_lon: np.ndarray,
) -> np.ndarray:
    """
    Bilinearly interpolate the coarse-resolution factor grid to the full MAPS grid.

    Points that fall outside the sample domain are extrapolated from the nearest
    boundary (fill_value=None). Since the sample grid always covers the US bounding
    box, only marginal ocean cells will be extrapolated; these are clamped to the
    valid range afterward.

    Requires scipy. Raises RuntimeError if scipy is not installed.
    """
    if not _SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for grid interpolation in non-dry-run mode.\n"
            "Install with: pip install scipy"
        )

    interpolator = _RGI(
        (sample_lats, sample_lons),
        factor_coarse,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    grid_lat, grid_lon = np.meshgrid(maps_lat, maps_lon, indexing="ij")
    points = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)
    factor_fine = interpolator(points).reshape(len(maps_lat), len(maps_lon))
    return np.clip(factor_fine, _FACTOR_MIN, _FACTOR_MAX).astype(np.float32)


# ---------------------------------------------------------------------------
# Dry-run synthetic grid
# ---------------------------------------------------------------------------

def make_dry_run_factor_grid(
    maps_lat: np.ndarray,
    maps_lon: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic meteo_beta_factor grid for format-verification purposes.

    No network calls are made. The result mimics a realistic winter-like
    pattern: higher latitudes have slightly elevated factors (colder, drier)
    with small random perturbations. Values are clipped to [0.75, 1.25].
    """
    rng = np.random.default_rng(seed)
    ny, nx = len(maps_lat), len(maps_lon)
    lat_std = max(maps_lat.std(), 1.0)
    lat_norm = (maps_lat - maps_lat.mean()) / lat_std
    # Higher latitude → colder → transmission slightly elevated.
    lat_effect = 1.0 + 0.10 * lat_norm
    noise = rng.normal(loc=0.0, scale=0.05, size=(ny, nx))
    grid = lat_effect[:, np.newaxis] + noise
    return np.clip(grid, 0.75, 1.25).astype(np.float32)


# ---------------------------------------------------------------------------
# NetCDF output — must satisfy read_grid_and_var() in the IC pre-processor
# ---------------------------------------------------------------------------

def write_meteo_nc(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    factor_grid: np.ndarray,
    init_date_str: str,
) -> None:
    """
    Write the meteo_beta_factor grid to a NetCDF file.

    Output schema (matches the structure that read_grid_and_var() in
    build_maps_initial_conditions_with_history_full.py expects):

        Dimensions : lat  (size = len(lat))
                     lon  (size = len(lon))
        Variables  :
            lat              — f8, (lat,), units="degrees_north"
            lon              — f8, (lon,), units="degrees_east"
            meteo_beta_factor — f4, (lat, lon), zlib=True

    Exactly ONE data variable besides lat/lon is written so read_grid_and_var()
    passes its single-variable guard. No 'valid_range' attribute is added —
    that CF attribute can trigger automatic masking in netCDF4-python, which
    would corrupt multiplier cells when the reader calls np.ma.filled(..., 0.0).

    NaN/inf values are replaced with 1.0 (neutral multiplier) before writing
    so the reader's NaN→0.0 fill never accidentally zeros out beta.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_grid = np.where(np.isfinite(factor_grid), factor_grid, 1.0).astype(np.float32)

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("lat", int(lat.size))
        ds.createDimension("lon", int(lon.size))

        latv = ds.createVariable("lat", "f8", ("lat",))
        lonv = ds.createVariable("lon", "f8", ("lon",))
        latv[:] = lat
        lonv[:] = lon
        latv.units = "degrees_north"
        lonv.units = "degrees_east"

        fv = ds.createVariable("meteo_beta_factor", "f4", ("lat", "lon"), zlib=True)
        fv[:, :] = safe_grid
        fv.long_name = "dimensionless transmission-rate multiplier from meteorological conditions"
        fv.units = "1"
        fv.comment = f"init_date={init_date_str}; range=[{_FACTOR_MIN},{_FACTOR_MAX}]"

        ds.title = f"MAPS meteorological beta factor — {init_date_str}"
        ds.source = "Open-Meteo API (https://open-meteo.com)"
        ds.conventions = "CF-1.8"

    print(f"Wrote: {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Drive the meteorological factor grid build from CLI arguments."""
    ap = argparse.ArgumentParser(
        description=(
            "Build a meteo_beta_factor NetCDF grid for MAPS from Open-Meteo data. "
            "Pass --dry-run for format verification without network calls."
        )
    )
    ap.add_argument("--init-date", required=True,
                    help="Initialisation date in YYYY-MM-DD format")
    ap.add_argument("--output", required=True,
                    help="Output NetCDF path (e.g. maps_meteo_2026-05-17.nc)")

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--namelist",
                     help="Path to init_template.nml — reads population_mapsgrid_file for lat/lon")
    src.add_argument("--pop-grid",
                     help="Path to MAPS population NetCDF (to extract lat/lon reference grid)")

    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Coarse API sampling resolution in degrees (default 1.0)")
    ap.add_argument("--humidity-weight", type=float, default=0.50,
                    help="Weight for absolute humidity sub-factor (default 0.50)")
    ap.add_argument("--uv-weight", type=float, default=0.30,
                    help="Weight for UV index sub-factor (default 0.30)")
    ap.add_argument("--temp-weight", type=float, default=0.20,
                    help="Weight for temperature sub-factor (default 0.20)")
    ap.add_argument("--request-delay", type=float, default=0.06,
                    help="Seconds between Open-Meteo API calls (default 0.06)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate synthetic data without any API calls (format testing)")
    args = ap.parse_args()

    # ── Resolve the reference lat/lon grid ───────────────────────────────────
    if args.dry_run and args.namelist is None and args.pop_grid is None:
        # Minimal synthetic grid for format testing — no file needed.
        maps_lat = np.array([32.0, 36.0, 40.0, 44.0], dtype=np.float64)
        maps_lon = np.array([-120.0, -115.0, -110.0, -105.0, -100.0], dtype=np.float64)
        print("[dry-run] Using built-in synthetic 4×5 reference grid.")
    elif args.namelist:
        cfg = parse_simple_namelist(Path(args.namelist))
        pop_nc = Path(cfg["population_mapsgrid_file"])
        maps_lat, maps_lon = read_reference_latlon(pop_nc)
        print(f"Reference grid from namelist → {pop_nc}")
    elif args.pop_grid:
        maps_lat, maps_lon = read_reference_latlon(Path(args.pop_grid))
        print(f"Reference grid from: {args.pop_grid}")
    else:
        ap.error("Supply --namelist or --pop-grid, or use --dry-run without a grid source.")

    print(f"MAPS grid : {len(maps_lat)} lat × {len(maps_lon)} lon")

    # ── Build the factor grid ─────────────────────────────────────────────────
    if args.dry_run:
        print(f"[dry-run] Generating synthetic meteo_beta_factor for {args.init_date}")
        factor_grid = make_dry_run_factor_grid(maps_lat, maps_lon)
    else:
        if not _REQUESTS_AVAILABLE:
            print("ERROR: 'requests' is required. Install: pip install requests", file=sys.stderr)
            sys.exit(1)
        if not _SCIPY_AVAILABLE:
            print("ERROR: 'scipy' is required. Install: pip install scipy", file=sys.stderr)
            sys.exit(1)

        lat_min = max(float(maps_lat.min()), _US_LAT_MIN)
        lat_max = min(float(maps_lat.max()), _US_LAT_MAX)
        lon_min = max(float(maps_lon.min()), _US_LON_MIN)
        lon_max = min(float(maps_lon.max()), _US_LON_MAX)

        sample_lats, sample_lons = build_sample_latlons(
            lat_min, lat_max, lon_min, lon_max, args.resolution
        )
        n_calls = len(sample_lats) * len(sample_lons)
        print(
            f"Sampling at {args.resolution}° resolution: "
            f"{len(sample_lats)} lat × {len(sample_lons)} lon = {n_calls} API calls"
        )

        factor_coarse = fetch_factor_on_sample_grid(
            sample_lats, sample_lons, args.init_date,
            args.humidity_weight, args.uv_weight, args.temp_weight,
            args.request_delay,
        )
        factor_grid = interpolate_to_maps_grid(
            factor_coarse, sample_lats, sample_lons, maps_lat, maps_lon
        )

    # ── Write output ──────────────────────────────────────────────────────────
    write_meteo_nc(Path(args.output), maps_lat, maps_lon, factor_grid, args.init_date)

    print(f"\nmeteo_beta_factor summary")
    print(f"  shape : {factor_grid.shape}")
    print(f"  min   : {float(factor_grid.min()):.4f}")
    print(f"  max   : {float(factor_grid.max()):.4f}")
    print(f"  mean  : {float(factor_grid.mean()):.4f}")
    print(f"  NaN   : {int(np.sum(~np.isfinite(factor_grid)))}")


if __name__ == "__main__":
    main()
