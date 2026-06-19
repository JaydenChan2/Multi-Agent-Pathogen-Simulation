# Script Documentation

## build_maps_initial_conditions_with_history_full.py

Builds the MAPS (Multi-Agent Pathogen Simulation) NetCDF initial conditions file from real-world variant and population data.

**Usage:**
```
python build_maps_initial_conditions_with_history_full.py <namelist_path>
```

**What it does:**

1. Reads configuration from a Fortran-style namelist file (`&init_nml` block), which specifies paths, dates, and epidemiological parameters.

2. Loads a US-level variant time series CSV and selects *active variants* — those with a national share above `active_variant_threshold_pct` (default 2.5%) on or before `init_date`. An `OTHER` catch-all variant is always appended to absorb residual share.

3. Estimates a per-variant initial transmission rate (`beta0`) from the peak log-linear growth rate observed in the US time series, using a sliding window.

4. Reads state-level infection CSV files (one per state, named `<STATE>_variant_infections.csv`) and distributes infections across the model grid weighted by population. Compartments built are:
   - **S** (Susceptible): set by `s0_fraction` of population
   - **I** (Infectious): infections within the residence-time window `1/epsilon`
   - **A** (Asymptomatic/pre-symptomatic): infections within `1/gamma` window, scaled by `a_fraction`
   - **R** (Recovered/removed): infections within `1/gammaR` window, scaled by `r_fraction`
   - **Ch / Cn** (immune memory, high and low cross-immunity): distributed from `C_target = population − S − I − A − R`, split by state-level historical case seeds

5. Builds a `state_ItoC` array — the per-state, per-variant daily infection history going back `history_days` days — for use in downstream immunity convolution.

6. Writes three output files (paths set in namelist):
   - **`output_init_file`** (NetCDF): all compartment grids plus state history
   - **`output_variant_file`** (CSV): active variants with shares, peak growth rates, beta guesses, and mean ages
   - **`output_namelist_fragment`** (text): Fortran namelist snippet with `nv`, `variant_labels`, and `beta0_by_variant`

**Key namelist parameters:**

| Parameter | Description |
|---|---|
| `init_date` | Simulation start date (`YYYY-MM-DD`) |
| `active_variant_threshold_pct` | Minimum national share (%) to include a variant |
| `s0_fraction` | Fraction of population initially susceptible |
| `epsilon` / `gamma` / `gammaR` | Exposed→I / I→removed / R→recovered rates (1/day) |
| `history_days` | Days of infection history to store for immunity convolution |
| `us_variant_file` | Path to US-level variant time series CSV |
| `state_variant_dir` | Directory containing per-state variant CSVs |
| `population_mapsgrid_file` | NetCDF file with population grid |
| `state_id_mapsgrid_file` | NetCDF file with FIPS state ID grid |

---

## edit_namelist.py

Edits a MAPS Fortran namelist in-place, setting variant parameters (beta0, immune escape IE0, cross-immunity matrix) and run-specific metadata for a single ensemble member.

**Usage:**
```
python edit_namelist.py <nml_path> <start_date> <member_name> <beta_scale> <ie_scale> [override_file] [parameter_mode]
```

| Argument | Description |
|---|---|
| `nml_path` | Path to the namelist file to edit |
| `start_date` | Forecast start date in `MM-DD-YYYY` format |
| `member_name` | Ensemble member identifier (e.g. `HZ1_v1`) |
| `beta_scale` | Global beta multiplier fallback |
| `ie_scale` | Global immune escape multiplier fallback |
| `override_file` | (optional) CSV with member-specific parameter overrides; defaults to `maps_variant_member_overrides.csv` |
| `parameter_mode` | `GLOBAL`, `DATABASE`, or `OVERRIDE` (default: `OVERRIDE`) |

**Parameter resolution priority (highest → lowest):**

1. **OVERRIDE** — member+variant-specific multipliers from `MAPS_FORECASTS/variant_database/<override_file>`
2. **DATABASE** — variant-specific multipliers from `MAPS_FORECASTS/variant_database/maps_variant_parameter_database.csv` (only used when `nmatch >= 30`)
3. **GLOBAL** — the `beta_scale` / `ie_scale` command-line arguments

**Per-variant beta0 / IE0 formulas:**
```
beta0_base = 0.23 + 0.20 * beta_guess      (from active variants CSV)
ie0_base   = 1.00 − 0.25 * beta_guess

beta0 = beta0_base * beta_mult
IE0   = ie0_base   * ie_mult
```

**Age adjustment:** If a variant emerged fewer than 30 days before the start date, beta is capped at 1.5× and IE is capped at `0.40 + 0.60 * (age/30)` to limit over-extrapolation for newly emerging variants.

**Cross-immunity matrix:** Built from a phylogenetic family tree (`variant_family_tree.csv`). Immunity values are assigned by taxonomic relationship:
- Same variant: 0.90
- Same family: 0.75
- Same branch: 0.55
- Same clade: 0.35
- Unrelated: 0.20

**Namelist fields updated in-place:**
- `netcdf_filename`, `init_i_file`, `output_dir`, `expname`
- `month`, `day`, `year`
- `nv`, `variant_labels`, `beta0_by_variant`, `IE0_by_variant`
- `cross_immunity_reinfection_flat`

---

## hz12_run.sh

Convenience script that launches forecasts for two ensemble members, **HZ1** and **HZ2**, with identical parameters.

**Usage:**
```
bash hz12_run.sh
```

**What it does:**

Calls `MAPS_FORECASTS/scripts/run_forecast.sh` twice with the following fixed arguments:

| Argument | Value |
|---|---|
| Start date | `05-17-2026` |
| Members | `HZ1`, `HZ2` |
| beta_scale | `1.00` |
| ie_scale | `0.50` |
| Ensemble size | `8` |
| Member range | `0-7` |
| Override file | `hz_variant_overrides.csv` |
| Parameter mode | `OVERRIDE` |

Both runs use the same date and parameter settings, differing only in their member identifier. The member ID is used by `edit_namelist.py` (called internally by `run_forecast.sh`) to look up member-specific overrides.
