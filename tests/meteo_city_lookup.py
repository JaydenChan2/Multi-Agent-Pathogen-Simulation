#!/usr/bin/env python3
"""
meteo_city_lookup.py — Interactive city lookup for meteorological_factor.py.

Type any city name (e.g. "Austin", "Paris", "Springfield") and this geocodes
it via Open-Meteo's free geocoding API, then runs the REAL pipeline functions
(fetch_meteo_point -> compute_absolute_humidity -> compute_meteo_factor ->
apply_meteo_to_beta) for that location. This is a manual/ad-hoc exploration
tool (hits the live network), not a pytest suite — see test_factor_accuracy.py
for the automated unit tests.

Usage
-----
    I                        # fully interactive prompts
    python3 tests/meteo_city_lookup.py --city Austin           # skip the city prompt
    python3 tests/meteo_city_lookup.py --city Paris --date 2026-07-15
    python3 tests/meteo_city_lookup.py --city Springfield --date 2026-01-15 --base-beta 0.25
    python3 tests/meteo_city_lookup.py --city Austin --month 2026-01       # full-month summary

Run with the project venv (has netCDF4/requests/scipy):
    /Users/jaydenchan/maps/.venv/bin/python3 tests/meteo_city_lookup.py
"""
from __future__ import annotations

import argparse
import calendar
import sys
from datetime import date as _date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
from meteorological_factor import (
    fetch_meteo_point,
    compute_absolute_humidity,
    compute_meteo_factor,
    apply_meteo_to_beta,
    f_humidity,
    f_uv,
    f_temp,
    _open_meteo_url,
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


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


def fetch_meteo_month(
    lat: float,
    lon: float,
    year: int,
    month: int,
    session: requests.Session,
    w_ah: float = 0.50,
    w_uv: float = 0.30,
    w_t: float = 0.20,
) -> list[dict]:
    """
    Fetch daily T/RH/UV for an entire calendar month in one API call and
    compute meteo_beta_factor for each day using the real pipeline math.

    Mirrors fetch_meteo_point's UV fallback (seasonal proxy when the API
    returns null) and endpoint selection (_open_meteo_url), but batches the
    whole month into a single request instead of one call per day. The
    archive/forecast endpoint choice is made once using the month's middle
    date, so a month straddling the ~5-day archive lag boundary may be
    missing a few of its most recent days.
    """
    first = _date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    last = _date(year, month, last_day)
    mid = _date(year, month, last_day // 2 + 1)
    base_url = _open_meteo_url(datetime(mid.year, mid.month, mid.day))

    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,uv_index_max",
        "timezone": "UTC",
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
    }
    resp = session.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    dates = daily.get("time", [])
    T_list = daily.get("temperature_2m_mean", [])
    RH_list = daily.get("relative_humidity_2m_mean", [])
    UV_list = daily.get("uv_index_max", [])

    records = []
    for i, d in enumerate(dates):
        T = T_list[i] if i < len(T_list) else None
        RH = RH_list[i] if i < len(RH_list) else None
        UV_raw = UV_list[i] if i < len(UV_list) else None
        if T is None or RH is None:
            continue  # day has no core data (e.g. beyond forecast horizon) — skip it
        if UV_raw is None:
            doy = datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday
            solar_angle = np.cos(np.radians(lat - 23.5 * np.cos(2 * np.pi * (doy - 172) / 365)))
            UV_raw = max(0.5, float(8.0 * solar_angle))
        AH = compute_absolute_humidity(T, RH)
        factor = compute_meteo_factor(AH, UV_raw, T, w_ah, w_uv, w_t)
        records.append({"date": d, "T": T, "RH": RH, "UV": UV_raw, "AH": AH, "factor": factor})
    return records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Look up any city and run meteorological_factor.py on it")
    ap.add_argument("--city", default=None, help="City name to search for (skips the interactive prompt)")
    date_group = ap.add_mutually_exclusive_group()
    date_group.add_argument("--date", default=None, help="Single day, YYYY-MM-DD (skips the interactive prompt)")
    date_group.add_argument("--month", default=None,
                             help="Whole calendar month, YYYY-MM — fetches every day in the month and "
                                  "reports a monthly-mean weighting instead of a single day")
    ap.add_argument("--base-beta", type=float, default=0.30,
                     help="Example base transmission rate before weather adjustment (default 0.30)")
    return ap.parse_args()


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


def report_single_day(lat: float, lon: float, date_str: str, base_beta: float, session: requests.Session) -> None:
    print(f"Date       : {date_str}")
    print("-" * 60)

    result = fetch_meteo_point(lat, lon, date_str, session)
    if result is None:
        print("Weather API call failed — no data returned for this location/date.")
        sys.exit(1)

    T, RH, UV = result["T"], result["RH"], result["UV"]
    print("Raw API observations:")
    print(f"  Temperature (T)        : {T:.2f} °C")
    print(f"  Relative humidity (RH) : {RH:.1f} %")
    print(f"  UV index (UV)          : {UV:.2f}")

    AH = compute_absolute_humidity(T, RH)
    print(f"\nDerived absolute humidity (AH): {AH:.3f} g/m^3")

    print("\nSub-factors (each centred at 1.0 at its reference condition):")
    print(f"  f_humidity(AH={AH:.2f}) = {f_humidity(AH):.4f}   (ref AH=7.0 g/m^3)")
    print(f"  f_uv(UV={UV:.2f})       = {f_uv(UV):.4f}   (ref UV=3.0)")
    print(f"  f_temp(T={T:.2f})       = {f_temp(T):.4f}   (ref T=10.0 C)")

    meteo_factor = compute_meteo_factor(AH, UV, T)
    print(f"\nCombined meteo_beta_factor (w_ah=0.50, w_uv=0.30, w_t=0.20): {meteo_factor:.4f}")

    new_beta = apply_meteo_to_beta(base_beta, meteo_factor)
    print("\nExample beta adjustment (METEO_BETA_WEIGHT=0.60):")
    print(f"  base_beta  = {base_beta:.4f}")
    print(f"  new_beta   = {new_beta:.4f}  "
          f"({'+' if new_beta >= base_beta else ''}{(new_beta/base_beta - 1)*100:.2f}% vs base)")


def report_month(lat: float, lon: float, month_str: str, base_beta: float, session: requests.Session) -> None:
    try:
        year, month = (int(p) for p in month_str.split("-", 1))
    except ValueError:
        print(f"--month must be in YYYY-MM format, got '{month_str}'.")
        sys.exit(1)

    print(f"Month      : {month_str}")
    print("-" * 60)

    records = fetch_meteo_month(lat, lon, year, month, session)
    if not records:
        print("Weather API call failed or returned no data for this location/month.")
        sys.exit(1)

    print(f"{'Date':<12} {'T(°C)':>7} {'RH(%)':>6} {'UV':>5} {'AH(g/m3)':>9} {'factor':>7}")
    for r in records:
        print(f"{r['date']:<12} {r['T']:>7.2f} {r['RH']:>6.1f} {r['UV']:>5.2f} {r['AH']:>9.3f} {r['factor']:>7.4f}")

    factors = np.array([r["factor"] for r in records])
    mean_factor = float(factors.mean())
    median_factor = float(np.median(factors))
    n_above = int(np.sum(factors > 1.0))
    n_below = int(np.sum(factors < 1.0))
    n_neutral = len(records) - n_above - n_below
    print(f"\n{len(records)} days fetched.")
    print("\nDay-to-day spread within the month:")
    print(f"  min    : {float(factors.min()):.4f}  ({records[int(factors.argmin())]['date']})")
    print(f"  max    : {float(factors.max()):.4f}  ({records[int(factors.argmax())]['date']})")
    print(f"  std    : {float(factors.std()):.4f}")
    print(f"  median : {median_factor:.4f}  (robust to outlier days, unlike the mean)")
    print(f"  days favouring transmission (factor > 1.0)   : {n_above}/{len(records)}")
    print(f"  days suppressing transmission (factor < 1.0) : {n_below}/{len(records)}")
    if n_neutral:
        print(f"  days exactly neutral (factor == 1.0)         : {n_neutral}/{len(records)}")

    print(f"\n{'=' * 60}")
    print(f"MONTHLY MULTIPLIER for {month_str} "
          f"(average of {len(records)} daily meteo_beta_factor values): {mean_factor:.4f}")
    print(f"  (median across the same {len(records)} days: {median_factor:.4f})")
    print(f"{'=' * 60}")

    new_beta = apply_meteo_to_beta(base_beta, mean_factor)
    print("\nExample beta adjustment using this monthly multiplier (METEO_BETA_WEIGHT=0.60):")
    print(f"  base_beta  = {base_beta:.4f}")
    print(f"  new_beta   = {new_beta:.4f}  "
          f"({'+' if new_beta >= base_beta else ''}{(new_beta/base_beta - 1)*100:.2f}% vs base)")


def main() -> None:
    args = parse_args()
    session = requests.Session()

    city_query = args.city or input("Enter a city name: ").strip()
    if not city_query:
        print("No city entered.")
        sys.exit(1)

    month_str = args.month
    date_str = args.date
    if month_str is None and date_str is None:
        raw = input("Enter a date (YYYY-MM-DD) or a month (YYYY-MM), blank = today: ").strip()
        if not raw:
            date_str = _date.today().isoformat()
        elif len(raw) == 7:
            month_str = raw
        else:
            date_str = raw

    label, lat, lon = resolve_city(city_query, session)
    print(f"\nCity       : {label}  ({lat:.4f}, {lon:.4f})")

    if month_str:
        report_month(lat, lon, month_str, args.base_beta, session)
    else:
        report_single_day(lat, lon, date_str, args.base_beta, session)

    session.close()


if __name__ == "__main__":
    main()
