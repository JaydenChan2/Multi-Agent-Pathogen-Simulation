#!/usr/bin/env python3
"""
infrastructure_grid.py — Community-infrastructure contact multiplier grid for MAPS.

Pipeline position
-----------------
Run ONCE (or whenever infrastructure data needs refreshing) before the IC
pre-processor:

    infrastructure_grid.py            ← THIS SCRIPT  →  maps_infrastructure.nc
    build_maps_initial_conditions_with_history_full.py  (IC pre-processor)
    edit_namelist.py                                    (per-member editor)
    Fortran MAPS model

This is a static pre-computation step: infrastructure changes slowly, so the
output file is reused across many forecast cycles without re-querying.

What it does
------------
1. Reads the lat/lon reference grid from a MAPS population NetCDF (or namelist)
   so the output is pixel-aligned with the IC file.
2. Queries the OpenStreetMap Overpass API for community infrastructure POIs
   (schools, libraries, transit, supermarkets, community centres, etc.) across
   the continental US, tiling the bounding box into _TILE_SIZE_DEG × _TILE_SIZE_DEG
   sub-queries to avoid server timeouts.
3. Optionally supplements school data with an NCES Common Core of Data CSV,
   which provides per-school enrollment counts for capacity-weighted scoring.
4. Rasterises collected POIs onto the MAPS grid (nearest-cell assignment),
   accumulating infrastructure weights per cell.
5. Normalises the raw per-capita density to a dimensionless infra_multiplier
   centred at 1.0 via z-score scaling controlled by the --alpha parameter.
   Urban cells score > 1 (higher contact rates); rural cells score < 1.
6. Writes a NetCDF file with the exact structure expected by read_grid_and_var()
   in the IC pre-processor:
       Dimensions : lat, lon
       Variables  : lat [f8,(lat,)], lon [f8,(lon,)],
                    infra_multiplier [f4,(lat,lon), zlib]
   NaN cells are pre-filled with 1.0 before writing.

Pass --dry-run for format-only testing (no network calls).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from netCDF4 import Dataset

# Deferred: only needed in non-dry-run mode.
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Infrastructure type definitions and weights
# ---------------------------------------------------------------------------

# OSM amenity tag → relative transmission-weight, calibrated from published
# contact-pattern matrices (Mossong et al. 2008; Hoang et al. 2019).
# A weight of 1.0 represents baseline community contact rate; higher values
# indicate settings that generate proportionally more transmission-relevant
# contacts per unit time.
INFRA_WEIGHTS: Dict[str, float] = {
    "school":           1.50,   # dense, sustained child / young-adult contacts
    "university":       1.30,   # high mixing across age groups
    "library":          0.80,   # indoor, cross-demographic, lower density
    "bus_station":      1.20,   # high-volume, enclosed, short-duration
    "subway_entrance":  1.20,   # same dynamics as bus station
    "community_centre": 1.00,   # sustained indoor gatherings
    "place_of_worship": 0.70,   # weekly high-density, typically brief
    "restaurant":       0.60,   # indoor dining, moderate dwell time
    "fast_food":        0.40,   # shorter dwell than full-service dining
    "supermarket":      0.90,   # recurring high-volume indoor visits
    "gym":              0.80,   # sustained exertion, elevated aerosol risk
    "hospital":         0.50,   # partially covered by the existing baseline
}

# Additional OSM key=value pairs beyond amenity=, with their weights.
EXTRA_OSM_TAGS: Dict[Tuple[str, str], float] = {
    ("shop", "supermarket"):          0.90,
    ("leisure", "fitness_centre"):    0.80,
    ("public_transport", "stop_position"): 1.00,
}

# US continental bounding box.
_US_LAT_MIN, _US_LAT_MAX = 24.0, 50.0
_US_LON_MIN, _US_LON_MAX = -126.0, -65.0

# Overpass API endpoint.
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Each tile is queried as a separate Overpass request.
_TILE_SIZE_DEG: float = 5.0

# Multiplier clamp applied to every cell before writing.
_MULTIPLIER_MIN: float = 0.70
_MULTIPLIER_MAX: float = 1.50

# Default z-score scale factor: 1 SD above mean → multiplier of 1.15.
_DEFAULT_ALPHA: float = 0.15

# WEIGHT — how strongly the infrastructure factor modulates beta.
# Set lower than the meteo weight because infrastructure-based contact patterns
# are less directly validated against observed transmission data.
INFRA_BETA_WEIGHT: float = 0.40  # WEIGHT


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


def read_population_grid(
    nc_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a MAPS population NetCDF and return (lat, lon, pop_grid).

    Uses the same single-variable convention expected by read_grid_and_var() in
    the IC pre-processor: the file must contain exactly one data variable besides
    lat and lon. Masked values and NaN/inf are replaced with 0.0.
    """
    with Dataset(nc_path, "r") as ds:
        lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        candidates = [n for n in ds.variables if n not in ("lat", "lon")]
        if len(candidates) != 1:
            raise ValueError(
                f"Population grid must have exactly one data variable; "
                f"found: {candidates} in {nc_path}"
            )
        raw = ds.variables[candidates[0]][:]
        pop = np.ma.filled(raw, 0.0) if np.ma.isMaskedArray(raw) else np.array(raw, dtype=np.float64)
        pop = np.where(np.isfinite(pop), np.maximum(pop, 0.0), 0.0)
    return lat, lon, pop


# ---------------------------------------------------------------------------
# OpenStreetMap Overpass querying
# ---------------------------------------------------------------------------

def _build_overpass_query(
    south: float, west: float, north: float, east: float
) -> str:
    """
    Build an Overpass QL query that returns all infrastructure POI nodes within
    a geographic bounding box.

    The query uses [out:json] for easy parsing and a 60-second server timeout.
    It unions all amenity types in INFRA_WEIGHTS and all key=value pairs in
    EXTRA_OSM_TAGS so a single tile request captures every infrastructure type.
    """
    bbox = f"{south:.4f},{west:.4f},{north:.4f},{east:.4f}"
    amenity_lines = "\n  ".join(
        f'node["amenity"="{tag}"]({bbox});'
        for tag in INFRA_WEIGHTS
    )
    extra_lines = "\n  ".join(
        f'node["{k}"="{v}"]({bbox});'
        for k, v in EXTRA_OSM_TAGS
    )
    return (
        "[out:json][timeout:60];\n"
        "(\n"
        f"  {amenity_lines}\n"
        f"  {extra_lines}\n"
        ");\n"
        "out body;\n"
    )


def query_osm_tile(
    south: float,
    west: float,
    north: float,
    east: float,
    session: "_requests.Session",  # type: ignore[name-defined]
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> List[Tuple[float, float, float]]:
    """
    Query the Overpass API for infrastructure POIs in one geographic tile.

    Returns a list of (lat, lon, weight) tuples representing each POI found.
    On repeated failure the tile is skipped and an empty list is returned so
    the overall grid build is not aborted by a single flaky request.
    """
    query = _build_overpass_query(south, west, north, east)
    pois: List[Tuple[float, float, float]] = []

    for attempt in range(max_retries):
        try:
            resp = session.post(_OVERPASS_URL, data={"data": query}, timeout=90)
            resp.raise_for_status()
            for el in resp.json().get("elements", []):
                if el.get("type") != "node":
                    continue
                lat = el.get("lat")
                lon = el.get("lon")
                if lat is None or lon is None:
                    continue
                tags = el.get("tags", {})
                weight = INFRA_WEIGHTS.get(tags.get("amenity", ""), 0.0)
                if weight == 0.0:
                    for (k, v), w in EXTRA_OSM_TAGS.items():
                        if tags.get(k) == v:
                            weight = w
                            break
                if weight > 0.0:
                    pois.append((float(lat), float(lon), weight))
            return pois

        except Exception as exc:
            if attempt < max_retries - 1:
                print(
                    f"  Overpass query failed ({exc}); retrying in {retry_delay}s …",
                    file=sys.stderr,
                )
                time.sleep(retry_delay)
            else:
                print(
                    f"  Tile [{south:.1f},{west:.1f}]→[{north:.1f},{east:.1f}] "
                    f"failed after {max_retries} attempts; skipping.",
                    file=sys.stderr,
                )

    return pois


def fetch_all_osm_pois(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    tile_size: float = _TILE_SIZE_DEG,
    inter_tile_delay: float = 2.0,
) -> List[Tuple[float, float, float]]:
    """
    Tile the US bounding box and query OpenStreetMap for all tiles.

    Tiles are queried sequentially with inter_tile_delay seconds between
    requests to avoid hammering the public Overpass instance. Returns the
    aggregated list of (lat, lon, weight) tuples across all tiles.
    """
    import requests  # deferred: only needed in non-dry-run mode

    session = requests.Session()
    all_pois: List[Tuple[float, float, float]] = []
    lat_edges = np.arange(lat_min, lat_max, tile_size)
    lon_edges = np.arange(lon_min, lon_max, tile_size)
    total_tiles = len(lat_edges) * len(lon_edges)
    tile_num = 0

    for s in lat_edges:
        for w in lon_edges:
            n = min(s + tile_size, lat_max)
            e = min(w + tile_size, lon_max)
            tile_num += 1
            print(
                f"  tile {tile_num}/{total_tiles}: "
                f"[{s:.1f},{w:.1f}] → [{n:.1f},{e:.1f}]"
            )
            pois = query_osm_tile(s, w, n, e, session)
            all_pois.extend(pois)
            print(f"    → {len(pois)} POIs (running total: {len(all_pois)})")
            time.sleep(inter_tile_delay)

    session.close()
    return all_pois


# ---------------------------------------------------------------------------
# Optional NCES school supplement
# ---------------------------------------------------------------------------

def load_nces_schools(csv_path: Path) -> List[Tuple[float, float, float]]:
    """
    Load K-12 school locations and enrollment counts from an NCES Common Core CSV.

    Expected columns (NCES CCD format, available from nces.ed.gov/ccd/):
        LATCOD  — latitude in decimal degrees
        LONCOD  — longitude in decimal degrees
        MEMBER  — total enrollment

    The weight assigned to each school is scaled by enrollment so that a 1 000-
    student school receives the same weight as a single OSM 'school' node.
    Rows with missing or invalid coordinates, or zero enrollment, are skipped.

    Returning enrollment-weighted entries allows this data to replace OSM school
    nodes (which carry no capacity information) in high-accuracy use cases.
    """
    pois: List[Tuple[float, float, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("LATCOD") or row.get("lat") or "")
                lon = float(row.get("LONCOD") or row.get("lon") or "")
                enrollment = float(row.get("MEMBER") or row.get("enrollment") or "0")
            except (ValueError, TypeError):
                continue
            if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
                continue
            if enrollment <= 0.0:
                continue
            weight = INFRA_WEIGHTS["school"] * enrollment / 1000.0
            pois.append((lat, lon, weight))
    print(f"Loaded {len(pois)} schools from NCES CSV: {csv_path}")
    return pois


# ---------------------------------------------------------------------------
# Rasterisation and normalisation
# ---------------------------------------------------------------------------

def rasterize_pois(
    pois: List[Tuple[float, float, float]],
    maps_lat: np.ndarray,
    maps_lon: np.ndarray,
) -> np.ndarray:
    """
    Assign each POI to its nearest MAPS grid cell and accumulate weighted counts.

    Nearest-cell lookup uses the grid spacing assumed to be uniform (the MAPS
    grid is regular). POIs that fall outside the grid extent are silently
    discarded. Returns a float64 array of shape (len(maps_lat), len(maps_lon)).
    """
    ny, nx = len(maps_lat), len(maps_lon)
    raw_density = np.zeros((ny, nx), dtype=np.float64)
    if not pois:
        return raw_density

    dlat = float(maps_lat[1] - maps_lat[0]) if ny > 1 else 1.0
    dlon = float(maps_lon[1] - maps_lon[0]) if nx > 1 else 1.0

    for lat, lon, weight in pois:
        iy = int(round((lat - float(maps_lat[0])) / dlat))
        ix = int(round((lon - float(maps_lon[0])) / dlon))
        if 0 <= iy < ny and 0 <= ix < nx:
            raw_density[iy, ix] += weight

    return raw_density


def compute_infra_multiplier(
    raw_density: np.ndarray,
    pop_grid: Optional[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """
    Convert raw POI density to a dimensionless infra_multiplier centred at 1.0.

    Algorithm
    ---------
    1. Divide by population to get per-capita density (if pop_grid is supplied).
       Cells with zero population are excluded from statistics and set to 1.0.
    2. Z-score standardise across all populated cells.
    3. Map: multiplier = 1.0 + alpha * z_score.
    4. Clamp to [_MULTIPLIER_MIN, _MULTIPLIER_MAX].

    If all valid cells have identical density (zero variance), returns a flat
    1.0 grid rather than dividing by zero.
    """
    ny, nx = raw_density.shape

    if pop_grid is not None:
        pop_safe = np.where(pop_grid > 0.0, pop_grid, np.nan)
        per_capita = raw_density / pop_safe
        mask = np.isfinite(per_capita) & (pop_grid > 0.0)
    else:
        per_capita = raw_density.astype(np.float64)
        mask = np.ones((ny, nx), dtype=bool)

    valid_vals = per_capita[mask]
    if valid_vals.size == 0 or float(valid_vals.std()) < 1e-12:
        return np.ones((ny, nx), dtype=np.float32)

    mu = float(valid_vals.mean())
    sigma = float(valid_vals.std())
    z = (per_capita - mu) / sigma

    multiplier = 1.0 + alpha * z
    multiplier = np.where(mask, multiplier, 1.0)
    return np.clip(multiplier, _MULTIPLIER_MIN, _MULTIPLIER_MAX).astype(np.float32)


# ---------------------------------------------------------------------------
# Dry-run synthetic grid
# ---------------------------------------------------------------------------

def make_dry_run_multiplier_grid(
    maps_lat: np.ndarray,
    maps_lon: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic infra_multiplier grid for format-verification purposes.

    No network calls are made. The result mimics a plausible urban–rural
    gradient: cells in the eastern US (higher population density historically)
    receive slightly elevated multipliers. Random noise is added to avoid a
    perfectly smooth gradient. Values are clipped to [_MULTIPLIER_MIN, _MULTIPLIER_MAX].
    """
    rng = np.random.default_rng(seed)
    ny, nx = len(maps_lat), len(maps_lon)
    lon_std = max(float(maps_lon.std()), 1.0)
    # More negative longitude = further west = less dense → lower multiplier.
    lon_norm = (maps_lon - float(maps_lon.mean())) / lon_std
    lon_effect = 1.0 + 0.08 * (-lon_norm)
    noise = rng.normal(loc=0.0, scale=0.05, size=(ny, nx))
    grid = lon_effect[np.newaxis, :] + noise
    return np.clip(grid, _MULTIPLIER_MIN, _MULTIPLIER_MAX).astype(np.float32)


# ---------------------------------------------------------------------------
# Beta contribution — weighted application of the infra factor to beta
# ---------------------------------------------------------------------------

def apply_infra_to_beta(
    base_beta: float,
    infra_factor: float,
    weight: float = INFRA_BETA_WEIGHT,  # WEIGHT
) -> float:
    """
    Apply the infrastructure multiplier to a base beta value with an explicit weight.

    Formula
    -------
    new_beta = base_beta × (1.0 + weight × (infra_factor − 1.0))

    At weight=0.40 (default), a 10% infrastructure-driven increase in the
    factor raises beta by 4% rather than the full 10%. This conservative weight
    reflects that infrastructure density is a proxy for contact rate, not a
    direct measurement.

    Parameters
    ----------
    base_beta    : the beta value before environmental adjustment
    infra_factor : spatial mean of the infra_multiplier grid (1.0 = neutral)
    weight       : how strongly infrastructure density modulates beta
                   (default INFRA_BETA_WEIGHT)

    Returns
    -------
    Adjusted beta value. Guaranteed >= 0.
    """
    # WEIGHT — attenuates the infra factor's deviation from neutral before applying
    scaled_factor = 1.0 + weight * (infra_factor - 1.0)  # WEIGHT
    return max(0.0, base_beta * scaled_factor)


# ---------------------------------------------------------------------------
# NetCDF output — must satisfy read_grid_and_var() in the IC pre-processor
# ---------------------------------------------------------------------------

def write_infra_nc(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    multiplier_grid: np.ndarray,
    alpha: float,
) -> None:
    """
    Write the infra_multiplier grid to a NetCDF file.

    Output schema (matches the structure that read_grid_and_var() in
    build_maps_initial_conditions_with_history_full.py expects):

        Dimensions : lat  (size = len(lat))
                     lon  (size = len(lon))
        Variables  :
            lat              — f8, (lat,), units="degrees_north"
            lon              — f8, (lon,), units="degrees_east"
            infra_multiplier — f4, (lat, lon), zlib=True

    Exactly ONE data variable besides lat/lon is written so read_grid_and_var()
    passes its single-variable guard. No 'valid_range' attribute is written —
    that CF attribute can trigger automatic masking in netCDF4-python, which
    would corrupt multiplier cells when the reader calls np.ma.filled(..., 0.0).

    NaN/inf values are replaced with 1.0 (neutral multiplier) before writing
    so the reader's NaN→0.0 fill never accidentally zeros out contact rates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_grid = np.where(np.isfinite(multiplier_grid), multiplier_grid, 1.0).astype(np.float32)

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("lat", int(lat.size))
        ds.createDimension("lon", int(lon.size))

        latv = ds.createVariable("lat", "f8", ("lat",))
        lonv = ds.createVariable("lon", "f8", ("lon",))
        latv[:] = lat
        lonv[:] = lon
        latv.units = "degrees_north"
        lonv.units = "degrees_east"

        mv = ds.createVariable("infra_multiplier", "f4", ("lat", "lon"), zlib=True)
        mv[:, :] = safe_grid
        mv.long_name = (
            "dimensionless contact-rate multiplier from community infrastructure density"
        )
        mv.units = "1"
        mv.comment = f"alpha={alpha}; range=[{_MULTIPLIER_MIN},{_MULTIPLIER_MAX}]"

        ds.title = "MAPS infrastructure contact multiplier grid"
        ds.source = "OpenStreetMap Overpass API; NCES Common Core of Data"
        ds.conventions = "CF-1.8"

    print(f"Wrote: {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Drive the infrastructure multiplier grid build from CLI arguments."""
    ap = argparse.ArgumentParser(
        description=(
            "Build an infra_multiplier NetCDF grid for MAPS from OSM data. "
            "Pass --dry-run for format verification without network calls."
        )
    )
    ap.add_argument("--output", required=True,
                    help="Output NetCDF path (e.g. maps_infrastructure.nc)")

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--namelist",
                     help="Path to init_template.nml — reads population_mapsgrid_file for lat/lon")
    src.add_argument("--pop-grid",
                     help="Path to MAPS population NetCDF (for lat/lon and population weighting)")

    ap.add_argument("--nces-csv",
                    help="Path to NCES CCD school CSV (optional; adds enrollment-weighted schools)")
    ap.add_argument("--alpha", type=float, default=_DEFAULT_ALPHA,
                    help=f"Z-score scaling factor for the multiplier (default {_DEFAULT_ALPHA})")
    ap.add_argument("--tile-size", type=float, default=_TILE_SIZE_DEG,
                    help=f"Overpass tile size in degrees (default {_TILE_SIZE_DEG})")
    ap.add_argument("--tile-delay", type=float, default=2.0,
                    help="Seconds between Overpass tile queries (default 2.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate synthetic data without any API calls (format testing)")
    args = ap.parse_args()

    # ── Resolve the reference lat/lon grid and optional population ────────────
    pop_grid: Optional[np.ndarray] = None

    if args.dry_run and args.namelist is None and args.pop_grid is None:
        maps_lat = np.array([32.0, 36.0, 40.0, 44.0], dtype=np.float64)
        maps_lon = np.array([-120.0, -115.0, -110.0, -105.0, -100.0], dtype=np.float64)
        print("[dry-run] Using built-in synthetic 4×5 reference grid.")
    elif args.namelist:
        cfg = parse_simple_namelist(Path(args.namelist))
        pop_nc = Path(cfg["population_mapsgrid_file"])
        maps_lat, maps_lon, pop_grid = read_population_grid(pop_nc)
        print(f"Reference grid and population from namelist → {pop_nc}")
    elif args.pop_grid:
        maps_lat, maps_lon, pop_grid = read_population_grid(Path(args.pop_grid))
        print(f"Reference grid and population from: {args.pop_grid}")
    else:
        ap.error(
            "Supply --namelist or --pop-grid, or use --dry-run without a grid source."
        )

    print(f"MAPS grid : {len(maps_lat)} lat × {len(maps_lon)} lon")

    # ── Build the multiplier grid ─────────────────────────────────────────────
    if args.dry_run:
        print("[dry-run] Generating synthetic infra_multiplier grid")
        multiplier_grid = make_dry_run_multiplier_grid(maps_lat, maps_lon)
    else:
        if not _REQUESTS_AVAILABLE:
            print("ERROR: 'requests' is required. Install: pip install requests", file=sys.stderr)
            sys.exit(1)

        print("Querying OpenStreetMap Overpass API …")
        all_pois = fetch_all_osm_pois(
            lat_min=max(float(maps_lat.min()), _US_LAT_MIN),
            lat_max=min(float(maps_lat.max()), _US_LAT_MAX),
            lon_min=max(float(maps_lon.min()), _US_LON_MIN),
            lon_max=min(float(maps_lon.max()), _US_LON_MAX),
            tile_size=args.tile_size,
            inter_tile_delay=args.tile_delay,
        )

        if args.nces_csv:
            all_pois.extend(load_nces_schools(Path(args.nces_csv)))

        print(f"Total POIs collected: {len(all_pois)}")
        raw_density = rasterize_pois(all_pois, maps_lat, maps_lon)
        multiplier_grid = compute_infra_multiplier(raw_density, pop_grid, args.alpha)

    # ── Write output ──────────────────────────────────────────────────────────
    write_infra_nc(Path(args.output), maps_lat, maps_lon, multiplier_grid, args.alpha)

    print(f"\ninfra_multiplier summary")
    print(f"  shape : {multiplier_grid.shape}")
    print(f"  min   : {float(multiplier_grid.min()):.4f}")
    print(f"  max   : {float(multiplier_grid.max()):.4f}")
    print(f"  mean  : {float(multiplier_grid.mean()):.4f}")
    print(f"  NaN   : {int(np.sum(~np.isfinite(multiplier_grid)))}")


if __name__ == "__main__":
    main()
