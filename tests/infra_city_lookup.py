#!/usr/bin/env python3
"""
infra_city_lookup.py — Interactive city lookup for infrastructure_grid.py.

Type any city name (e.g. "Austin", "Paris", "Springfield") and this geocodes
it via Open-Meteo's free geocoding API, queries the OpenStreetMap Overpass API
for community-infrastructure POIs around it, then runs the REAL pipeline
functions (rasterize_pois -> compute_infra_multiplier -> apply_infra_to_beta)
to show the infra_multiplier for that location. This is a manual/ad-hoc
exploration tool (hits the live network), not a pytest suite — see
test_factor_accuracy.py for the automated unit tests.

Point vs. area — why not "day vs. month"
-----------------------------------------
The meteorological tool (meteo_city_lookup.py) toggles between a single DAY and
a whole MONTH because weather is time-varying. Infrastructure is *static*: it
comes from OpenStreetMap POI density and has no date dimension, so the temporal
toggle is meaningless here. The structural analog is spatial instead:

    meteo  single day   ->  infra  POINT : the one grid cell the city sits in
    meteo  whole month  ->  infra  AREA  : a grid of cells around the city, with
                                           a per-cell table plus mean/median/
                                           min/max/std and anomalous-cell flags

Important — the multiplier is relative, not absolute
----------------------------------------------------
compute_infra_multiplier() z-score-standardises POI density *across the cells it
is given*. A single isolated cell has zero variance, so it would always collapse
to a flat 1.0. To produce a meaningful number, POINT mode still samples the
surrounding region (the same grid AREA mode uses) to build the reference
distribution, then reports only your city's cell. Both modes are therefore
"relative to the sampled neighbourhood", NOT relative to the whole US the way a
full infrastructure_grid.py run is. Widen --radius for a broader reference.

Usage
-----
    python3 tests/infra_city_lookup.py                         # fully interactive
    python3 tests/infra_city_lookup.py --city Austin --point
    python3 tests/infra_city_lookup.py --city Paris --area
    python3 tests/infra_city_lookup.py --city Austin --area --radius 3 --cell-size 0.5
    python3 tests/infra_city_lookup.py --city Springfield --point --base-beta 0.25

Run with the project venv (has netCDF4/requests):
    /Users/jaydenchan/Multi-Agent-Pathogen-Simulation/.venv/bin/python3 tests/infra_city_lookup.py
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
from infrastructure_grid import (
    INFRA_WEIGHTS,
    EXTRA_OSM_TAGS,
    INFRA_BETA_WEIGHT,
    _DEFAULT_ALPHA,
    _MULTIPLIER_MIN,
    _MULTIPLIER_MAX,
    _OVERPASS_URL,
    _build_overpass_query,
    rasterize_pois,
    compute_infra_multiplier,
    apply_infra_to_beta,
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Overpass returns 406 Not Acceptable to requests without a User-Agent, so send
# one. (infrastructure_grid.query_osm_tile omits this and can 406 on the public
# endpoint; set the same header there if you hit it.)
_OVERPASS_HEADERS = {"User-Agent": "maps-infra-city-lookup/1.0 (MAPS exploration tool)"}

# A typed POI is (lat, lon, weight, label); the label names the OSM tag that
# matched (e.g. "school" or "shop=supermarket") for the per-type breakdown.
TypedPOI = Tuple[float, float, float, str]


# ---------------------------------------------------------------------------
# Geocoding (shared shape with meteo_city_lookup.py)
# ---------------------------------------------------------------------------

def geocode_city(query: str, session: requests.Session) -> list[dict]:
    """Look up a city name via Open-Meteo's geocoding API. Returns a list of candidates."""
    resp = session.get(
        GEOCODE_URL,
        params={"name": query, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def choose_candidate(candidates: list[dict]) -> dict:
    """If there's more than one match, ask the user to pick one; otherwise auto-select."""
    if len(candidates) == 1:
        return candidates[0]

    print(f"\nFound {len(candidates)} matches — pick one:")
    for i, c in enumerate(candidates, start=1):
        admin = c.get("admin1", "")
        country = c.get("country", "")
        location = ", ".join(p for p in (admin, country) if p)
        print(f"  [{i}] {c['name']}"
              f"{' — ' + location if location else ''}"
              f"  ({c['latitude']:.4f}, {c['longitude']:.4f})")

    while True:
        choice = input(f"Enter 1-{len(candidates)}: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("Invalid choice, try again.")


def resolve_city(city_query: str, session: requests.Session) -> tuple[str, float, float]:
    """Geocode a city name to (label, lat, lon), prompting for disambiguation if needed."""
    print(f"\nSearching for '{city_query}' …")
    try:
        candidates = geocode_city(city_query, session)
    except Exception as exc:
        print(f"Geocoding request failed: {exc}")
        sys.exit(1)

    if not candidates:
        print(f"No matches found for '{city_query}'. Try a different spelling or a nearby larger city.")
        sys.exit(1)

    place = choose_candidate(candidates)
    label_parts = [place["name"], place.get("admin1", ""), place.get("country", "")]
    label = ", ".join(p for p in label_parts if p)
    return label, place["latitude"], place["longitude"]


# ---------------------------------------------------------------------------
# OpenStreetMap querying — reuses infrastructure_grid's query builder and
# weight tables so POIs are scored exactly as a real grid build would score them.
# ---------------------------------------------------------------------------

def query_osm_pois(
    south: float,
    west: float,
    north: float,
    east: float,
    session: requests.Session,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> List[TypedPOI]:
    """
    Query the Overpass API for infrastructure POIs in one bounding box.

    Mirrors infrastructure_grid.query_osm_tile's request and weight-assignment
    logic exactly (same _build_overpass_query, same INFRA_WEIGHTS/EXTRA_OSM_TAGS
    precedence), but additionally keeps a human-readable type label per POI so
    this tool can show a per-type breakdown. Returns (lat, lon, weight, label)
    tuples; on repeated failure returns whatever was collected so far.
    """
    query = _build_overpass_query(south, west, north, east)
    pois: List[TypedPOI] = []

    for attempt in range(max_retries):
        try:
            resp = session.post(_OVERPASS_URL, data={"data": query},
                                headers=_OVERPASS_HEADERS, timeout=90)
            resp.raise_for_status()
            for el in resp.json().get("elements", []):
                if el.get("type") != "node":
                    continue
                lat = el.get("lat")
                lon = el.get("lon")
                if lat is None or lon is None:
                    continue
                tags = el.get("tags", {})
                label = tags.get("amenity", "")
                weight = INFRA_WEIGHTS.get(label, 0.0)
                if weight == 0.0:
                    for (k, v), w in EXTRA_OSM_TAGS.items():
                        if tags.get(k) == v:
                            weight = w
                            label = f"{k}={v}"
                            break
                if weight > 0.0:
                    pois.append((float(lat), float(lon), weight, label))
            return pois

        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"  Overpass query failed ({exc}); retrying in {retry_delay}s …",
                      file=sys.stderr)
                time.sleep(retry_delay)
            else:
                print(f"  Overpass query failed after {max_retries} attempts; "
                      f"returning partial results.", file=sys.stderr)

    return pois


# ---------------------------------------------------------------------------
# Local grid helpers
# ---------------------------------------------------------------------------

def build_local_grid(
    center_lat: float,
    center_lon: float,
    cell_size: float,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a (2*radius+1) x (2*radius+1) regular grid centred on the city.

    The city sits at the centre of the middle cell (index `radius` on each
    axis). Spacing is `cell_size` degrees, matching the uniform-grid assumption
    rasterize_pois() relies on.
    """
    offsets = np.arange(-radius, radius + 1) * cell_size
    lats = center_lat + offsets
    lons = center_lon + offsets
    return lats, lons


def bbox_for_grid(
    lats: np.ndarray, lons: np.ndarray, cell_size: float
) -> Tuple[float, float, float, float]:
    """Return (south, west, north, east) covering every cell, including cell margins."""
    half = cell_size / 2.0
    return (
        float(lats.min()) - half,
        float(lons.min()) - half,
        float(lats.max()) + half,
        float(lons.max()) + half,
    )


def assign_cell(
    lat: float, lon: float, lats: np.ndarray, lons: np.ndarray, cell_size: float
) -> Optional[Tuple[int, int]]:
    """Nearest-cell index for a POI, or None if it falls outside the grid (same rule as rasterize_pois)."""
    iy = int(round((lat - float(lats[0])) / cell_size))
    ix = int(round((lon - float(lons[0])) / cell_size))
    if 0 <= iy < len(lats) and 0 <= ix < len(lons):
        return iy, ix
    return None


def count_pois_per_cell(
    pois: List[TypedPOI], lats: np.ndarray, lons: np.ndarray, cell_size: float
) -> np.ndarray:
    """Integer POI counts per cell (parallels rasterize_pois' weighted accumulation)."""
    counts = np.zeros((len(lats), len(lons)), dtype=int)
    for lat, lon, _w, _label in pois:
        cell = assign_cell(lat, lon, lats, lons, cell_size)
        if cell is not None:
            counts[cell] += 1
    return counts


def compute_region(
    lats: np.ndarray,
    lons: np.ndarray,
    cell_size: float,
    alpha: float,
    session: requests.Session,
) -> Tuple[List[TypedPOI], np.ndarray, np.ndarray, np.ndarray]:
    """
    Query OSM for the whole grid, then run the real rasterisation + multiplier math.

    Returns (typed_pois, raw_density, counts, multiplier_grid). pop_grid is None
    (no population weighting), so the multiplier is a pure z-score of raw POI
    density across the sampled cells.
    """
    south, west, north, east = bbox_for_grid(lats, lons, cell_size)
    print(f"Querying OpenStreetMap Overpass for "
          f"[{south:.2f},{west:.2f}] → [{north:.2f},{east:.2f}] …")
    typed = query_osm_pois(south, west, north, east, session)
    print(f"  → {len(typed)} infrastructure POIs found in the sampled region.")

    raw_density = rasterize_pois([(la, lo, w) for la, lo, w, _ in typed], lats, lons)
    counts = count_pois_per_cell(typed, lats, lons, cell_size)
    multiplier = compute_infra_multiplier(raw_density, None, alpha)
    return typed, raw_density, counts, multiplier


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _beta_report(base_beta: float, factor: float, label: str) -> None:
    """Shared beta-adjustment footer."""
    new_beta = apply_infra_to_beta(base_beta, factor)
    print(f"\nExample beta adjustment using {label} (INFRA_BETA_WEIGHT={INFRA_BETA_WEIGHT}):")
    print(f"  base_beta  = {base_beta:.4f}")
    print(f"  new_beta   = {new_beta:.4f}  "
          f"({'+' if new_beta >= base_beta else ''}{(new_beta / base_beta - 1) * 100:.2f}% vs base)")


def report_point(
    lat: float, lon: float, cell_size: float, radius: int,
    alpha: float, base_beta: float, session: requests.Session,
) -> None:
    """Report the single grid cell the city falls in (using the region as its reference distribution)."""
    lats, lons = build_local_grid(lat, lon, cell_size, radius)
    ci = radius  # centre index on each axis — the city's own cell

    print(f"Mode       : POINT — the {cell_size}° cell containing the city")
    print(f"Reference  : z-scored against the surrounding "
          f"{len(lats)}×{len(lons)} cell region (radius {radius})")
    print("-" * 68)

    typed, raw_density, counts, multiplier = compute_region(lats, lons, cell_size, alpha, session)

    cell_lat, cell_lon = float(lats[ci]), float(lons[ci])
    half = cell_size / 2.0
    print(f"\nCity's cell : centre ({cell_lat:.3f}, {cell_lon:.3f}), "
          f"bounds [{cell_lat - half:.3f}→{cell_lat + half:.3f}, "
          f"{cell_lon - half:.3f}→{cell_lon + half:.3f}]")

    in_cell = [(la, lo, w, lb) for la, lo, w, lb in typed
               if assign_cell(la, lo, lats, lons, cell_size) == (ci, ci)]
    print(f"POIs in cell: {counts[ci, ci]}")
    if in_cell:
        by_type = Counter(lb for _la, _lo, _w, lb in in_cell)
        weight_by_type = {lb: w for _la, _lo, w, lb in in_cell}
        print("  breakdown by type (count × per-POI weight):")
        for lb, n in by_type.most_common():
            print(f"    {lb:<22} {n:>3} × {weight_by_type[lb]:.2f}")

    print(f"\nRaw weighted POI density (this cell) : {raw_density[ci, ci]:.3f}")
    print(f"Region mean raw density              : {float(raw_density.mean()):.3f}  "
          f"(std {float(raw_density.std()):.3f})")

    factor = float(multiplier[ci, ci])
    print(f"\ninfra_multiplier for the city's cell : {factor:.4f}  "
          f"(clamped to [{_MULTIPLIER_MIN}, {_MULTIPLIER_MAX}], alpha={alpha})")
    if float(raw_density.std()) < 1e-12:
        print("  NOTE: every sampled cell had identical density, so the z-score is\n"
              "        undefined and the multiplier defaults to a flat 1.0. Widen\n"
              "        --radius to include cells with contrasting density.")

    _beta_report(base_beta, factor, "the city's cell multiplier")


def report_area(
    lat: float, lon: float, cell_size: float, radius: int,
    alpha: float, base_beta: float, session: requests.Session,
) -> None:
    """Report every cell in the region, plus spread statistics and anomalous cells."""
    lats, lons = build_local_grid(lat, lon, cell_size, radius)

    print(f"Mode       : AREA — {2 * radius + 1}×{2 * radius + 1} grid of "
          f"{cell_size}° cells around the city")
    print("-" * 68)

    _typed, raw_density, counts, multiplier = compute_region(lats, lons, cell_size, alpha, session)

    print(f"\n{'lat':>8} {'lon':>9} {'POIs':>5} {'rawWt':>8} {'mult':>7}")
    for iy in range(len(lats)):
        for ix in range(len(lons)):
            print(f"{lats[iy]:>8.3f} {lons[ix]:>9.3f} {counts[iy, ix]:>5d} "
                  f"{raw_density[iy, ix]:>8.3f} {multiplier[iy, ix]:>7.4f}")

    flat = multiplier.ravel()
    mean_factor = float(flat.mean())
    median_factor = float(np.median(flat))
    std_factor = float(flat.std())
    n_above = int(np.sum(flat > 1.0))
    n_below = int(np.sum(flat < 1.0))
    n_neutral = flat.size - n_above - n_below

    argmin = np.unravel_index(int(flat.argmin()), multiplier.shape)
    argmax = np.unravel_index(int(flat.argmax()), multiplier.shape)
    print(f"\n{flat.size} cells sampled ({int(counts.sum())} POIs total).")
    print("\nCell-to-cell spread across the region:")
    print(f"  min    : {float(flat.min()):.4f}  "
          f"({lats[argmin[0]]:.3f}, {lons[argmin[1]]:.3f})")
    print(f"  max    : {float(flat.max()):.4f}  "
          f"({lats[argmax[0]]:.3f}, {lons[argmax[1]]:.3f})")
    print(f"  std    : {std_factor:.4f}")
    print(f"  median : {median_factor:.4f}  (robust to outlier cells, unlike the mean)")
    print(f"  cells above neutral (mult > 1.0) : {n_above}/{flat.size}")
    print(f"  cells below neutral (mult < 1.0) : {n_below}/{flat.size}")
    if n_neutral:
        print(f"  cells exactly neutral (== 1.0)   : {n_neutral}/{flat.size}")

    if std_factor > 0:
        z = (multiplier - mean_factor) / std_factor
        anomalous = [(int(iy), int(ix)) for iy, ix in zip(*np.where(np.abs(z) > 2.0))]
        if anomalous:
            print("\nAnomalous cells (|z-score| > 2.0 vs the region's mean/std — reported, "
                  "not excluded;\nthese are real density contrasts, not data errors):")
            for iy, ix in anomalous:
                direction = "unusually dense" if z[iy, ix] > 0 else "unusually sparse"
                print(f"  ({lats[iy]:.3f}, {lons[ix]:.3f}) : mult={multiplier[iy, ix]:.4f}  "
                      f"(z={z[iy, ix]:+.2f}, {direction})")
        else:
            print("\nNo statistically anomalous cells (all within 2 std of the region mean).")

    print(f"\n{'=' * 68}")
    print(f"AREA MEAN MULTIPLIER "
          f"(average of {flat.size} cell infra_multiplier values): {mean_factor:.4f}")
    print(f"  (median across the same {flat.size} cells: {median_factor:.4f})")
    print(f"{'=' * 68}")

    _beta_report(base_beta, mean_factor, "this area-mean multiplier")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Look up any city and run infrastructure_grid.py on it (POINT or AREA mode)."
    )
    ap.add_argument("--city", default=None, help="City name to search for (skips the interactive prompt)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--point", action="store_true",
                      help="Report only the city's own grid cell (analog of the meteo single-day view)")
    mode.add_argument("--area", action="store_true",
                      help="Report a grid of cells around the city with spread stats (analog of the meteo month view)")
    ap.add_argument("--cell-size", type=float, default=0.5,
                    help="Grid cell size in degrees (default 0.5)")
    ap.add_argument("--radius", type=int, default=2,
                    help="Number of cells on each side of the city (grid is 2*radius+1 per axis; default 2)")
    ap.add_argument("--alpha", type=float, default=_DEFAULT_ALPHA,
                    help=f"Z-score scaling factor for the multiplier (default {_DEFAULT_ALPHA})")
    ap.add_argument("--base-beta", type=float, default=0.30,
                    help="Example base transmission rate before infrastructure adjustment (default 0.30)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.radius < 1:
        print("--radius must be at least 1 (a single cell has no reference distribution).")
        sys.exit(1)

    session = requests.Session()

    city_query = args.city or input("Enter a city name: ").strip()
    if not city_query:
        print("No city entered.")
        sys.exit(1)

    area_mode = args.area
    if not args.area and not args.point:
        raw = input("Mode — [p]oint (single cell) or [a]rea (grid), blank = point: ").strip().lower()
        area_mode = raw.startswith("a")

    label, lat, lon = resolve_city(city_query, session)
    print(f"\nCity       : {label}  ({lat:.4f}, {lon:.4f})")

    if area_mode:
        report_area(lat, lon, args.cell_size, args.radius, args.alpha, args.base_beta, session)
    else:
        report_point(lat, lon, args.cell_size, args.radius, args.alpha, args.base_beta, session)

    session.close()


if __name__ == "__main__":
    main()
