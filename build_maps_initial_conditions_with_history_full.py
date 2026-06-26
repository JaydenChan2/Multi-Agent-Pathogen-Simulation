#!/usr/bin/env python3
from __future__ import annotations

import argparse  # Parse command-line arguments.
import csv  # Read and write CSV files.
import re  # Handle filename and variable-name pattern matching.
from datetime import datetime  # Parse and compare dates.
from pathlib import Path  # Work with filesystem paths.
from typing import List, Optional, Dict, Tuple  # Type hints for collections and optional values.

import numpy as np  # Numerical arrays and vectorized calculations.
from netCDF4 import Dataset  # Read and write NetCDF files.


STATE_ABBR_TO_FIPS = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10, "DC": 11,
    "FL": 12, "GA": 13, "HI": 15, "ID": 16, "IL": 17, "IN": 18, "IA": 19, "KS": 20,
    "KY": 21, "LA": 22, "ME": 23, "MD": 24, "MA": 25, "MI": 26, "MN": 27, "MS": 28,
    "MO": 29, "MT": 30, "NE": 31, "NV": 32, "NH": 33, "NJ": 34, "NM": 35, "NY": 36,
    "NC": 37, "ND": 38, "OH": 39, "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45,
    "SD": 46, "TN": 47, "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54,
    "WI": 55, "WY": 56, "PR": 72
}


def parse_simple_namelist(path: Path) -> dict:
    """Read the &init_nml block from a simple namelist file and return parsed key/value pairs."""
    entries = {}
    inside = False
    for raw_line in path.read_text().splitlines():
        # Strip Fortran-style comments and surrounding whitespace first.
        line = raw_line.split("!")[0].strip()
        # Skip blank lines after comment removal.
        if not line:
            continue
        # Start parsing only after we reach the init_nml section header.
        if line.lower().startswith("&init_nml"):
            inside = True
            continue
        # Stop at the end of the namelist block.
        if line.strip() == "/":
            break
        # Ignore everything outside the target block, and ignore non-assignments.
        if not inside or "=" not in line:
            continue

        # Split the assignment into key and value.
        key, val = line.split("=", 1)
        # Normalize keys to lowercase so lookups are consistent.
        key = key.strip().lower()
        # Remove trailing commas that are common in namelist syntax.
        val = val.strip().rstrip(",")

        # Convert common boolean spellings to Python booleans.
        if val.lower() in (".true.", "true"):
            entries[key] = True
        elif val.lower() in (".false.", "false"):
            entries[key] = False
        # Remove surrounding quotes for string values.
        elif (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
            entries[key] = val[1:-1]
        else:
            # Try numeric conversion first; fall back to the raw string if parsing fails.
            try:
                if "." in val or "e" in val.lower():
                    entries[key] = float(val)
                else:
                    entries[key] = int(val)
            except Exception:
                entries[key] = val
    return entries


def parse_date(s: str) -> datetime:
    """Parse a date string using the supported input formats."""
    # Try each supported date format until one succeeds.
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse date: {s}")


def read_csv_rows(path: Path):
    """Read a CSV file and return its header row plus all data rows."""
    # Use utf-8-sig so files with a BOM still parse correctly.
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    # A CSV with no rows cannot provide either a header or data.
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    # Strip whitespace from the header so later column matching is robust.
    header = [h.strip() for h in rows[0]]
    return header, rows[1:]


def find_date_col(header: List[str]) -> int:
    """Return the index of the Date column, or 0 if no exact match is found."""
    # Prefer an explicit Date column, but fall back to column 0 if needed.
    for i, h in enumerate(header):
        if h.strip().lower() == "date":
            return i
    return 0


def normalize_variant_name(name: str) -> str:
    """Normalize a variant label into the canonical uppercase, dot-separated form."""
    # Different CSVs use slightly different naming conventions, so normalize them here.
    return name.strip().upper().replace("_", ".")


def is_real_variant(name: str) -> bool:
    """Return True when the column name looks like an actual variant rather than metadata."""
    # Filter out summary and metadata columns that are not variant-specific measurements.
    n = name.strip().upper()
    bad = {
        "TOTAL_STATE_INFECTIONS", "TOTAL_STAT", "TOTAL_INFECTIONS", "TOTAL", "REGION",
        "US", "HHS", "STATE", "STATE_NAME", "_SOURCE", ""
    }
    return n not in bad


def load_us_variant_timeseries(path: Path):
    """Load the national variant time series CSV into dates, regions, totals, and per-variant arrays."""
    # The first three columns are the date, region, and total case count.
    header, rows = read_csv_rows(path)
    date_col, region_col, total_col = 0, 1, 2
    # Keep only columns that represent real variants.
    variant_cols = [i for i in range(3, len(header)) if is_real_variant(normalize_variant_name(header[i]))]
    variants = [normalize_variant_name(header[i]) for i in variant_cols]

    dates, regions, totals = [], [], []
    series = {v: [] for v in variants}

    for row in rows:
        if not row or len(row) < len(header):
            continue
        # Parse the row into aligned date, region, total, and variant series values.
        dates.append(parse_date(row[date_col].strip()))
        regions.append(row[region_col].strip())
        totals.append(float(row[total_col] or 0.0))
        for i, v in zip(variant_cols, variants):
            val = row[i].strip()
            series[v].append(float(val) if val else 0.0)

    return dates, regions, np.array(totals, dtype=float), {k: np.array(v, dtype=float) for k, v in series.items()}


def select_active_variants(dates, totals, series, init_date: datetime, threshold_pct: float):
    """Choose variants whose share exceeds the threshold on or before the initialization date."""
    # Find the latest national observation at or before the initialization date.
    idx_candidates = [i for i, d in enumerate(dates) if d <= init_date]
    if not idx_candidates:
        raise ValueError("No US variant rows on or before init_date")

    idx = idx_candidates[-1]
    total = totals[idx]
    active, shares = [], {}

    # Compute each variant's share of the national total at that date.
    for v, arr in series.items():
        share = 100.0 * arr[idx] / total if total > 0 else 0.0
        shares[v] = share
        if share >= threshold_pct:
            active.append(v)

    active.sort(key=lambda v: shares[v], reverse=True)
    return idx, active, shares

def compute_variant_ages(dates, totals, series, active, init_date, emergence_threshold_pct=1.0):
    """Estimate how many days before initialization each active variant first crossed the emergence threshold."""
    ages = {}

    # Search backward from the initialization date to find the first emergence point.
    for v in active:

        first_idx = None

        for i in range(len(dates)):

            if dates[i] > init_date:
                break

            total = totals[i]

            if total <= 0.0:
                continue

            share = 100.0 * series[v][i] / total

            if share >= emergence_threshold_pct:
                first_idx = i
                break

        # Use a large fallback age when the variant never crosses the threshold.
        if first_idx is None:
            ages[v] = 999.0
        else:
            ages[v] = (init_date - dates[first_idx]).days

    ages["OTHER"] = 999.0

    return ages

def estimate_peak_growth_rate(arr, window_days: int = 7, min_value: float = 1.0) -> float:
    """Estimate the maximum exponential growth rate over a sliding window."""
    # Scan overlapping windows and fit log-linear growth where the counts are large enough.
    best_r = 0.0
    n = len(arr)
    for end_idx in range(window_days - 1, n):
        start_idx = end_idx - window_days + 1
        y = arr[start_idx:end_idx + 1].astype(float)
        mask = y > min_value
        if np.count_nonzero(mask) < 2:
            continue
        t = np.arange(window_days, dtype=float)[mask]
        ly = np.log(y[mask])
        if len(t) < 2:
            continue
        # The slope of log(counts) over time is the exponential growth rate.
        r, _ = np.polyfit(t, ly, 1)
        if np.isfinite(r):
            best_r = max(best_r, float(r))
    return best_r


def write_active_variants_csv(path: Path, active, shares, betas, peak_r, ages):
    """Write the selected variant summary table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Keep the CSV columns aligned with the values later consumed by the build pipeline.
        w.writerow(["variant", "share_percent", "r_peak", "beta0_guess", "mean_age_days"])
        for v in active:
            w.writerow([v, f"{shares[v]:.6f}", f"{peak_r[v]:.8f}", f"{betas[v]:.8f}", f"{ages[v]:.1f}"])


def write_namelist_fragment(path: Path, active, betas):
    """Write a small namelist fragment with the active variants and beta guesses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the labels as quoted strings so the Fortran namelist parser sees them correctly.
    labels = ", ".join([f"'{v}'" for v in active])
    # Serialize the beta guesses in the same order as the active variants.
    beta_vals = ", ".join([f"{betas[v]:.8f}" for v in active])
    with path.open("w", encoding="utf-8") as f:
        f.write("! MAPS preprocessor-generated variant fragment\n")
        f.write(f"nv = {len(active)},\n")
        f.write(f"variant_labels = {labels},\n")
        f.write(f"beta0_by_variant = {beta_vals},\n")


def read_grid_and_var(path: Path):
    """Read a lat/lon grid NetCDF file and return the coordinates plus its single data variable."""
    with Dataset(path, "r") as ds:
        # Load the coordinate axes first so the caller can validate grid alignment.
        lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        # The file should contain exactly one non-coordinate field.
        candidates = [n for n in ds.variables if n not in ("lat", "lon")]
        if len(candidates) != 1:
            raise ValueError(f"{path} must contain exactly one data variable besides lat/lon; found {candidates}")
        var_name = candidates[0]
        # Convert masked values or NaNs to zeros so downstream math stays well-defined.
        raw = ds.variables[var_name][:]
        data = np.ma.filled(raw, 0.0) if np.ma.isMaskedArray(raw) else np.array(raw)
        data = np.array(data)
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isfinite(data), data, 0.0)
    return lat, lon, data, var_name


def infer_state_fips_from_filename(path: Path) -> Optional[int]:
    """Infer a state FIPS code from tokens in the filename stem."""
    # Break the filename into tokens and look for a known state abbreviation.
    stem = path.stem.upper()
    parts = re.split(r"[^A-Z0-9]+", stem)
    for p in parts:
        if p in STATE_ABBR_TO_FIPS:
            return STATE_ABBR_TO_FIPS[p]
    return None


def load_state_variant_rows(path: Path):
    """Load a state-level variant CSV into dates and per-variant series arrays."""
    header, rows = read_csv_rows(path)
    # Detect the date column and skip metadata columns that are not variant values.
    date_col = find_date_col(header)
    ignore_names = {
        "DATE", "REGION", "TOTAL_STAT", "TOTAL_STATE_INFECTIONS", "TOTAL_INFECTIONS",
        "HHS", "STATE", "STATE_NAME", "_SOURCE"
    }

    variant_cols, variant_names = [], []
    for i, h in enumerate(header):
        hnorm = h.strip().upper()
        if i == date_col or hnorm in ignore_names:
            continue
        vname = normalize_variant_name(h)
        if is_real_variant(vname):
            variant_cols.append(i)
            variant_names.append(vname)

    dates = []
    series = {v: [] for v in variant_names}

    for row in rows:
        if not row or len(row) < len(header):
            continue
        # Parse each row into a date plus one time series per detected variant.
        dates.append(parse_date(row[date_col].strip()))
        for i, v in zip(variant_cols, variant_names):
            val = row[i].strip()
            try:
                series[v].append(float(val) if val else 0.0)
            except ValueError:
                series[v].append(0.0)

    return dates, {k: np.array(v, dtype=float) for k, v in series.items()}


def find_state_files(state_dir: Path):
    """Return all state variant CSV files in the given directory."""
    # Only include files that match the state-level variant naming convention.
    return sorted([p for p in state_dir.glob("*_variant_infections.csv") if p.is_file()])


def compute_state_populations(pop_grid: np.ndarray, state_id_grid: np.ndarray):
    """Compute total population per state from the population and state-ID grids."""
    pops = {}
    # Aggregate nonnegative population over every unique state ID present in the grid.
    for sid in np.unique(state_id_grid):
        sid_int = int(sid)
        if sid_int <= 0:
            continue
        mask = (state_id_grid == sid_int)
        pops[sid_int] = float(np.sum(np.maximum(pop_grid[mask], 0.0)))
    return pops

def build_state_history(
    state_variant_dir: Path,
    active_variants: List[str],
    init_date: datetime,
    state_index: Dict[int, int],
    history_days: int,
) -> np.ndarray:
    """
    Build the state-by-variant infection history tensor used for immunity convolution.

    state_ItoC(state, variant, history)

    history index 0 = infections on init_date
    history index 1 = infections 1 day before init_date
    ...
    """
    nstate = len(state_index)
    nv = len(active_variants)
    state_ItoC = np.zeros((nstate, nv, history_days), dtype=np.float32)

    # Track only observed variants here; OTHER is synthesized from the remainder.
    observed_variants = [v for v in active_variants if v != "OTHER"]
    iv_other = active_variants.index("OTHER") if "OTHER" in active_variants else None

    # Read each state file, map it to a state index, and accumulate history by age.
    for sf in find_state_files(state_variant_dir):
        fips = infer_state_fips_from_filename(sf)
        if fips is None or fips not in state_index:
            continue

        sidx = state_index[fips]
        dates, series = load_state_variant_rows(sf)

        for row_idx, d in enumerate(dates):
            # Convert the row date into an age measured backward from init_date.
            age = (init_date - d).days
            if age < 0 or age >= history_days:
                continue

            observed_sum = 0.0

            for iv, v in enumerate(active_variants):
                if v == "OTHER":
                    continue
                arr = series.get(v, None)
                if arr is None or row_idx >= len(arr):
                    continue
                val = max(0.0, float(arr[row_idx]))
                state_ItoC[sidx, iv, age] += val
                observed_sum += val

            # Put the residual mass into OTHER so the history stays closed.
            if iv_other is not None:
                total_row = 0.0
                for arr in series.values():
                    if row_idx < len(arr):
                        total_row += max(0.0, float(arr[row_idx]))
                state_ItoC[sidx, iv_other, age] += max(0.0, total_row - observed_sum)

    return state_ItoC


def compute_state_sum_ndi_by_variant(state_dir: Path, init_date: datetime, active_variants):
    """Sum the national diagnosed infections across states for each active variant."""
    ndi_by_variant = {v: 0.0 for v in active_variants}
    state_files = find_state_files(state_dir)
    observed_variants = [v for v in active_variants if v != "OTHER"]

    # Aggregate the latest pre-init state observations into a national total per variant.
    for sf in state_files:
        fips = infer_state_fips_from_filename(sf)
        if fips is None:
            continue

        dates, series = load_state_variant_rows(sf)
        idx_candidates = [i for i, d in enumerate(dates) if d <= init_date]
        if not idx_candidates:
            continue
        idx = idx_candidates[-1]

        # Sum the observed variants directly from the CSV row.
        state_sum = 0.0

        for v in observed_variants:
            arr = series.get(v, None)
            if arr is not None:
                val = float(arr[idx])
                ndi_by_variant[v] += val
                state_sum += val

        # Put any remaining total into OTHER so the national totals stay consistent.
        total_row = sum(float(arr[idx]) for arr in series.values())
        other_val = max(0.0, total_row - state_sum)

        if "OTHER" in ndi_by_variant:
            ndi_by_variant["OTHER"] += other_val

    return ndi_by_variant


def build_compartment_grids(
    active_variants,
    shares,
    init_date,
    windows,
    fractions,
    s0_fraction,
    target_ndi_total,
    epsilon_init,
    pop_grid,
    state_id_grid,
    state_dir: Path,
):
    
    """Build the initial S, I, A, R, Ch, and Cn grids from state and variant inputs."""
    # Start with empty compartment grids for every grid cell and active variant.
    ny, nx = pop_grid.shape
    nv = len(active_variants)

    I_grid = np.zeros((ny, nx, nv), dtype=np.float32)
    A_grid = np.zeros((ny, nx, nv), dtype=np.float32)
    R_grid = np.zeros((ny, nx, nv), dtype=np.float32)
    Ch_grid = np.zeros((ny, nx, nv), dtype=np.float32)
    Cn_grid = np.zeros((ny, nx, nv), dtype=np.float32)

    state_pops = compute_state_populations(pop_grid, state_id_grid)
    state_variant_comp = {}

    state_files = find_state_files(state_dir)
    print(f"Found {len(state_files)} state files in {state_dir}")

    # Split the active set into observed variants and the synthetic OTHER bucket.
    observed_variants = [v for v in active_variants if v != "OTHER"]
    iv_other = active_variants.index("OTHER")

    # ------------------------------------------------------------------
    # 1. Read state CSVs and build state-level I/A/R and memory seeds
    # ------------------------------------------------------------------
    for sf in state_files:
        fips = infer_state_fips_from_filename(sf)
        print(f"Reading state file: {sf.name} -> inferred FIPS = {fips}")

        if fips is None:
            print("  SKIP: could not infer state FIPS from filename")
            continue
        if fips not in state_pops or state_pops[fips] <= 0.0:
            print(f"  SKIP: FIPS {fips} not present in state_id grid or has zero population")
            continue

        dates, series = load_state_variant_rows(sf)
        print(f"  Variants in file: {list(series.keys())[:10]}")

        # Use the last row on or before init_date as the state snapshot.
        idx_candidates = [i for i, d in enumerate(dates) if d <= init_date]
        if not idx_candidates:
            print("  SKIP: no rows on or before init_date")
            continue
        idx = idx_candidates[-1]

        for v in observed_variants:
            arr = series.get(v, None)
            if arr is None:
                state_variant_comp[(fips, v)] = {"I": 0.0, "A": 0.0, "R": 0.0, "Cseed": 0.0}
                continue

            # Sum each residence-time window to approximate current compartment sizes.
            def window_sum(a0, a1):
                lo = max(0, a0)
                hi = min(idx + 1, a1)
                return float(np.sum(arr[lo:hi])) if hi > lo else 0.0

            I_tot = window_sum(idx - windows["I"] + 1, idx + 1)
            A_tot = fractions["A"] * window_sum(
                idx - windows["A"] - windows["A_lag"] + 1,
                idx - windows["A_lag"] + 1,
            )
            R_tot = fractions["R"] * window_sum(
                idx - windows["R"] - windows["R_lag"] + 1,
                idx - windows["R_lag"] + 1,
            )
            C_seed = window_sum(
                idx - windows["C"] - windows["C_lag"] + 1,
                idx - windows["C_lag"] + 1,
            )

            state_variant_comp[(fips, v)] = {"I": I_tot, "A": A_tot, "R": R_tot, "Cseed": C_seed}

    print("Finished reading state totals.")

    # ------------------------------------------------------------------
    # 2. Put observed-variant I/A/R onto the grid using population weights
    # ------------------------------------------------------------------
    for sid, state_pop in state_pops.items():
        if state_pop <= 0.0:
            continue

        mask = (state_id_grid == sid)
        weights = np.zeros_like(pop_grid, dtype=np.float64)
        weights[mask] = np.maximum(pop_grid[mask], 0.0) / state_pop

        for iv, v in enumerate(observed_variants):
            comp = state_variant_comp.get((sid, v), None)
            if comp is None:
                continue

            if comp["I"] > 0.0:
                I_grid[:, :, iv] += (weights * comp["I"]).astype(np.float32)
            if comp["A"] > 0.0:
                A_grid[:, :, iv] += (weights * comp["A"]).astype(np.float32)
            if comp["R"] > 0.0:
                R_grid[:, :, iv] += (weights * comp["R"]).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Build OTHER I/A/R from missing current-share closure
    # ------------------------------------------------------------------
    total_active_share = sum(shares[v] for v in active_variants if v != "OTHER") / 100.0
    other_share = max(0.0, 1.0 - total_active_share)

    if total_active_share > 0.0:
        observed_I_sum = np.sum(I_grid[:, :, :iv_other], axis=2)
        observed_A_sum = np.sum(A_grid[:, :, :iv_other], axis=2)
        observed_R_sum = np.sum(R_grid[:, :, :iv_other], axis=2)

        I_grid[:, :, iv_other] = (other_share * observed_I_sum / total_active_share).astype(np.float32)
        A_grid[:, :, iv_other] = (other_share * observed_A_sum / total_active_share).astype(np.float32)
        R_grid[:, :, iv_other] = (other_share * observed_R_sum / total_active_share).astype(np.float32)
    else:
        I_grid[:, :, iv_other] = 0.0
        A_grid[:, :, iv_other] = 0.0
        R_grid[:, :, iv_other] = 0.0

    # ------------------------------------------------------------------
    # 4. I/A/R are already history-derived from residence-time windows
    # ------------------------------------------------------------------
    iar_scale = 1.0

    print()
    print("History-derived compartment totals")
    print("----------------------------------")
    print(f"I total from history = {np.sum(I_grid):12.1f}")
    print(f"A total from history = {np.sum(A_grid):12.1f}")
    print(f"R total from history = {np.sum(R_grid):12.1f}")
    print()


    # ------------------------------------------------------------------
    # 5. Enforce population mask on I/A/R immediately
    # ------------------------------------------------------------------
    pop_mask = pop_grid >= 1.0
    for iv in range(nv):
        I_grid[:, :, iv] *= pop_mask
        A_grid[:, :, iv] *= pop_mask
        R_grid[:, :, iv] *= pop_mask

    # ------------------------------------------------------------------
    # 6. Set S target and compute total C available cell-by-cell
    # ------------------------------------------------------------------
    S_target = s0_fraction * pop_grid
    IAR_total = np.sum(I_grid + A_grid + R_grid, axis=2)

    iar_cap = np.maximum(pop_grid - S_target, 0.0)
    over_iar = IAR_total > iar_cap
    if np.any(over_iar):
        factor = np.ones_like(pop_grid, dtype=np.float64)
        factor[over_iar] = iar_cap[over_iar] / np.maximum(IAR_total[over_iar], 1.0e-12)
        for iv in range(nv):
            I_grid[:, :, iv] = (I_grid[:, :, iv] * factor).astype(np.float32)
            A_grid[:, :, iv] = (A_grid[:, :, iv] * factor).astype(np.float32)
            R_grid[:, :, iv] = (R_grid[:, :, iv] * factor).astype(np.float32)

    IAR_total = np.sum(I_grid + A_grid + R_grid, axis=2)
    C_target_total = np.maximum(pop_grid - S_target - IAR_total, 0.0)

    # ------------------------------------------------------------------
    # 7. Distribute C_target_total across variants using state-level C seeds
    # ------------------------------------------------------------------
    fh = fractions["Ch"] / max(fractions["Ch"] + fractions["Cn"], 1.0e-12)
    fn = fractions["Cn"] / max(fractions["Ch"] + fractions["Cn"], 1.0e-12)

    for sid, state_pop in state_pops.items():
        if state_pop <= 0.0:
            continue

        mask = (state_id_grid == sid)
        if not np.any(mask):
            continue

        weights = np.zeros_like(pop_grid, dtype=np.float64)
        weights[mask] = np.maximum(pop_grid[mask], 0.0) / state_pop

        state_C_total = float(np.sum(C_target_total[mask]))
        if state_C_total <= 0.0:
            continue

        cseed = np.zeros(nv, dtype=np.float64)
        for iv, v in enumerate(observed_variants):
            comp = state_variant_comp.get((sid, v), None)
            if comp is not None:
                cseed[iv] = max(0.0, comp["Cseed"])

        observed_seed_sum = float(np.sum(cseed[:iv_other]))
        if total_active_share > 0.0:
            cseed[iv_other] = other_share * observed_seed_sum / total_active_share
        else:
            cseed[iv_other] = 1.0

        seed_sum = float(np.sum(cseed))

        if seed_sum <= 0.0:
            for iv, v in enumerate(active_variants):
                cseed[iv] = max(0.0, shares.get(v, 0.0))
            seed_sum = float(np.sum(cseed))

        if seed_sum <= 0.0:
            cseed[:] = 0.0
            cseed[iv_other] = 1.0
            seed_sum = 1.0

        cweights = cseed / seed_sum

        for iv in range(nv):
            C_state_variant = state_C_total * cweights[iv]
            if C_state_variant <= 0.0:
                continue

            Ch_grid[:, :, iv] += (weights * (fh * C_state_variant)).astype(np.float32)
            Cn_grid[:, :, iv] += (weights * (fn * C_state_variant)).astype(np.float32)

    # ------------------------------------------------------------------
    # 8. Enforce population mask on memory
    # ------------------------------------------------------------------
    for iv in range(nv):
        Ch_grid[:, :, iv] *= pop_mask
        Cn_grid[:, :, iv] *= pop_mask

    # ------------------------------------------------------------------
    # 9. Fill any missing C into OTHER exactly once
    # ------------------------------------------------------------------
    C_actual = np.sum(Ch_grid + Cn_grid, axis=2)
    missing = np.maximum(C_target_total - C_actual, 0.0)
    if np.any(missing > 0.0):
        Ch_grid[:, :, iv_other] += (fh * missing).astype(np.float32)
        Cn_grid[:, :, iv_other] += (fn * missing).astype(np.float32)

    # ------------------------------------------------------------------
    # 10. Final exact closure
    # ------------------------------------------------------------------
    total_all = np.sum(I_grid + A_grid + R_grid + Ch_grid + Cn_grid, axis=2)
    S_grid = (pop_grid - total_all).astype(np.float32)
    S_grid = np.where(np.abs(S_grid) < 1.0e-6, 0.0, S_grid).astype(np.float32)

    neg_mask = S_grid < 0.0
    if np.any(neg_mask):
        overflow = -S_grid[neg_mask]
        other_mem = Ch_grid[:, :, iv_other] + Cn_grid[:, :, iv_other]
        removable = np.minimum(overflow, other_mem[neg_mask])

        denom = np.maximum(other_mem[neg_mask], 1.0e-12)
        ch_frac = Ch_grid[:, :, iv_other][neg_mask] / denom
        cn_frac = Cn_grid[:, :, iv_other][neg_mask] / denom

        Ch_grid[:, :, iv_other][neg_mask] -= (ch_frac * removable).astype(np.float32)
        Cn_grid[:, :, iv_other][neg_mask] -= (cn_frac * removable).astype(np.float32)

        total_all = np.sum(I_grid + A_grid + R_grid + Ch_grid + Cn_grid, axis=2)
        S_grid = np.maximum(pop_grid - total_all, 0.0).astype(np.float32)

    print("Max/min grids by variant:")
    for iv, v in enumerate(active_variants):
        print(f"  {v}: Imax={np.max(I_grid[:, :, iv])} Imin={np.min(I_grid[:, :, iv])}")

    print(f"S target fraction    = {s0_fraction}")
    print(f"Mean N0XY            = {np.mean(pop_grid)}")
    print(f"Mean S               = {np.mean(S_grid)}")
    print(f"Target NDIs total    = {target_ndi_total}")
    print(f"Applied I/A/R scale  = {iar_scale}")
    print(f"Scaled I total       = {np.sum(I_grid)}")

    total_compartments = np.sum(I_grid + A_grid + R_grid + Ch_grid + Cn_grid, axis=2)
    total_check = total_compartments + S_grid

    print(f"Mean total_check     = {np.mean(total_check)}")
    print(f"Max |total-N0XY|     = {np.max(np.abs(total_check - pop_grid))}")

    return S_grid, I_grid, A_grid, R_grid, Ch_grid, Cn_grid


def safe_varname(prefix: str, v: str) -> str:
    """Convert a variant label into a NetCDF-safe variable name."""
    return prefix + "_" + re.sub(r"[^0-9A-Za-z_]", "_", v)


def write_init_nc(
    path,
    lat,
    lon,
    variants,
    S_grid,
    I_grid,
    A_grid,
    R_grid,
    Ch_grid,
    Cn_grid,
    target_ndi_total,
    variant_targets,
    state_ids,
    state_ItoC,
    history_days,
):
    """Write the full initialization NetCDF file containing grids, metadata, and history."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("variant", len(variants))
        ds.createDimension("state", len(state_ids))
        ds.createDimension("history", history_days)
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)

        state_var = ds.createVariable("state_fips", "i4", ("state",))
        state_var[:] = np.array(state_ids, dtype=np.int32)
        state_var.long_name = "FIPS code for state dimension"

        hist_var = ds.createVariable("history_age_days", "i4", ("history",))
        hist_var[:] = np.arange(history_days, dtype=np.int32)
        hist_var.long_name = "days before initialization date; 0 is init date"

        hist = ds.createVariable("state_ItoC", "f4", ("state", "variant", "history"), zlib=True)
        hist[:, :, :] = state_ItoC
        hist.long_name = "state and variant resolved infection history used for immunity convolution"
        hist.units = "persons per day"

        ndi_tot = ds.createVariable("NDI_total_target", "f8")
        ndi_tot.assignValue(float(target_ndi_total))

        ndi_v = ds.createVariable("NDI_variant_target", "f8", ("variant",))
        ndi_v[:] = variant_targets

        for iv, v in enumerate(variants):
            vname = safe_varname("NDI_target", v)
            var = ds.createVariable(vname, "f8")
            var.assignValue(float(variant_targets[iv]))

        latv = ds.createVariable("lat", "f8", ("lat",))
        lonv = ds.createVariable("lon", "f8", ("lon",))
        latv[:] = lat
        lonv[:] = lon
        latv.units = "degrees_north"
        lonv.units = "degrees_east"

        s_var = ds.createVariable("S", "f4", ("lat", "lon"), zlib=True)
        s_var[:, :] = S_grid
        s_var.long_name = "initial susceptible population"

        for iv, v in enumerate(variants):
            fields = {
                safe_varname("I", v): I_grid[:, :, iv],
                safe_varname("A", v): A_grid[:, :, iv],
                safe_varname("R", v): R_grid[:, :, iv],
                safe_varname("C_h", v): Ch_grid[:, :, iv],
                safe_varname("C_n", v): Cn_grid[:, :, iv],
            }

            for name, field in fields.items():
                var = ds.createVariable(name, "f4", ("lat", "lon"), zlib=True)
                var[:, :] = field
                var.variant_label = v

        ds.title = "MAPS init with state-level infection history for immunity convolution"


def main():
    """Run the MAPS initial-condition builder from the supplied namelist configuration."""
    ap = argparse.ArgumentParser(description="Build MAPS real-grid initial conditions")
    ap.add_argument("namelist", help="Path to init_template.nml")
    args = ap.parse_args()

    cfg = parse_simple_namelist(Path(args.namelist))

    init_date = parse_date(str(cfg["init_date"]))
    threshold_pct = float(cfg.get("active_variant_threshold_pct", 2.5))
    gamma_ref = float(cfg.get("gamma_ref", 0.07))
    s0_fraction = float(cfg.get("s0_fraction", 0.15))
    growth_window_days = int(cfg.get("growth_window_days", 7))
    epsilon_init = float(cfg.get("epsilon_init", 0.0333333333))
    history_days = int(cfg.get("history_days", 300))

    epsilon = float(cfg.get("epsilon", 0.2857))
    gamma   = float(cfg.get("gamma",   0.089))
    gammaR  = float(cfg.get("gammar",  0.1132))
    print()
    print("DEBUG")
    print("-----")
    print("epsilon =", epsilon)
    print("gamma   =", gamma)
    print("gammaR  =", gammaR)
    print()
    
    I_window = max(1, int(round(1.0 / epsilon)))
    A_window = max(1, int(round(1.0 / gamma)))
    R_window = max(1, int(round(1.0 / gammaR)))

    print()
    print("Residence time windows")
    print("----------------------")
    print(f"I window = {I_window} days")
    print(f"A window = {A_window} days")
    print(f"R window = {R_window} days")
    print()
    
    windows = {
    "I": I_window,
    "A": A_window,
    "A_lag": int(cfg.get("a_lag_days", 0)),
    "R": R_window,
    "R_lag": int(cfg.get("r_lag_days", 7)),
    "C": int(cfg.get("c_window_days", 60)),
    "C_lag": int(cfg.get("c_lag_days", 21)),
    }
    
    fractions = {
        "A": float(cfg.get("a_fraction", 0.10)),
        "R": float(cfg.get("r_fraction", 0.05)),
        "Ch": float(cfg.get("ch_recovery_fraction", 0.85)),
        "Cn": float(cfg.get("cn_recovery_fraction", 0.15)),
    }

    us_variant_file = Path(cfg["us_variant_file"])
    state_variant_dir = Path(cfg["state_variant_dir"])
    pop_mapsgrid_file = Path(cfg["population_mapsgrid_file"])
    state_id_mapsgrid_file = Path(cfg["state_id_mapsgrid_file"])
    output_init_file = Path(cfg["output_init_file"])
    output_variant_file = Path(cfg["output_variant_file"])
    output_namelist_fragment = Path(cfg["output_namelist_fragment"])

    dates, regions, totals, us_series = load_us_variant_timeseries(us_variant_file)
    idx, active, shares = select_active_variants(dates, totals, us_series, init_date, threshold_pct)
    variant_ages = compute_variant_ages(dates, totals, us_series, active, init_date, emergence_threshold_pct=1.0)
        
    betas, peak_r = {}, {}
    for v in active:
        rmax = estimate_peak_growth_rate(us_series[v], window_days=growth_window_days, min_value=1.0)
        peak_r[v] = rmax
        betas[v] = max(0.0, rmax + gamma_ref)

    if "OTHER" not in active:
        beta_other = min(betas.values()) if len(betas) > 0 else gamma_ref
        active.append("OTHER")
        total_active_share = sum(shares[v] for v in active if v != "OTHER")
        shares["OTHER"] = max(0.0, 100.0 - total_active_share)
        peak_r["OTHER"] = 0.0
        betas["OTHER"] = beta_other
        
    write_active_variants_csv(output_variant_file, active, shares, betas, peak_r, variant_ages)
    write_namelist_fragment(output_namelist_fragment, active, betas)

    lat_p, lon_p, pop_grid, _ = read_grid_and_var(pop_mapsgrid_file)
    lat_s, lon_s, state_id_grid, _ = read_grid_and_var(state_id_mapsgrid_file)

    pop_grid = np.where(np.isfinite(pop_grid), pop_grid, 0.0)
    pop_grid = np.maximum(pop_grid, 0.0)
    state_id_grid = np.where(np.isfinite(state_id_grid), state_id_grid, 0).astype(np.int32)

    if pop_grid.shape != state_id_grid.shape:
        raise ValueError("population_mapsgrid and state_id_mapsgrid shapes do not match")
    if not (np.allclose(lat_p, lat_s) and np.allclose(lon_p, lon_s)):
        raise ValueError("population_mapsgrid and state_id_mapsgrid lat/lon do not match")

    ndi_by_variant = compute_state_sum_ndi_by_variant(state_variant_dir, init_date, active)
    variant_targets = np.array([ndi_by_variant[v] for v in active], dtype=np.float64)
    target_ndi_total = float(np.sum(variant_targets))

    print("Target total NDI:", target_ndi_total)
    print("Variant NDIs:")
    for v, val in zip(active, variant_targets):
        print(f"  {v}: {val}")

    # Build the compartment fields by projecting state-level counts onto the spatial grid.
    S_grid, I_grid, A_grid, R_grid, Ch_grid, Cn_grid = build_compartment_grids(
        active,
        shares,
        init_date,
        windows,
        fractions,
        s0_fraction,
        target_ndi_total,
        epsilon_init,
        pop_grid.astype(np.float64),
        state_id_grid.astype(np.int32),
        state_variant_dir,
    )

    print()
    print("History-derived compartment totals")
    print("----------------------------------")
    print(f"I total = {np.sum(I_grid):12.1f}")
    print(f"A total = {np.sum(A_grid):12.1f}")
    print(f"R total = {np.sum(R_grid):12.1f}")
    print()

    # Build the state history tensor used later by the NetCDF output.
    state_pops = compute_state_populations(pop_grid.astype(np.float64), state_id_grid.astype(np.int32))
    state_ids = sorted(state_pops.keys())
    state_index = {fips: i for i, fips in enumerate(state_ids)}

    state_ItoC = build_state_history(
        state_variant_dir=state_variant_dir,
        active_variants=active,
        init_date=init_date,
        state_index=state_index,
        history_days=history_days,
    )

    # Persist all derived outputs for the downstream MAPS workflow.
    print(f"Built state_ItoC with shape {state_ItoC.shape}")
    print(f"Total state_ItoC mass = {float(np.sum(state_ItoC))}")

    write_init_nc(
        output_init_file,
        lat_p,
        lon_p,
        active,
        S_grid,
        I_grid,
        A_grid,
        R_grid,
        Ch_grid,
        Cn_grid,
        target_ndi_total,
        variant_targets,
        state_ids,
        state_ItoC,
        history_days,
    )

    print("Selected active variants:")
    for v in active:
        print(f"  {v:20s} share={shares[v]:7.3f}% beta0_guess={betas[v]:.8f}")

    print(f"Wrote: {output_variant_file}")
    print(f"Wrote: {output_namelist_fragment}")
    print(f"Wrote: {output_init_file}")
    print(f"Init date: {init_date.strftime('%Y-%m-%d')}")
    print(f"History days: {history_days}")


if __name__ == "__main__":
    main()
