# meteorological_factor.py — How It Works

## What This Script Does

`meteorological_factor.py` produces a spatially-resolved weather-based multiplier grid that tells the MAPS model how strongly (or weakly) weather conditions favour disease transmission on a given forecast date. The output is a NetCDF file (`maps_meteo_<YYYY-MM-DD>.nc`) containing a single 2-D array called `meteo_beta_factor`.

A value of **1.0** means neutral — weather neither helps nor hinders spread.  
A value **> 1.0** means conditions favour transmission (cold, dry, low UV).  
A value **< 1.0** means conditions suppress transmission (warm, humid, high UV).

---

## Where It Sits in the Pipeline

```
meteorological_factor.py          →  maps_meteo_<date>.nc
         ↓
build_maps_initial_conditions_with_history_full.py   (IC pre-processor)
         ↓
edit_namelist.py                                     (per-ensemble-member editor)
         ↓
Fortran MAPS model
```

Run this script **once per forecast cycle**, before the IC pre-processor. The IC pre-processor then loads the output file to incorporate the weather modifier into the initial conditions.

---

## Step-by-Step Walkthrough

### Step 1 — Get the Reference Grid

The script needs to know the MAPS model's lat/lon grid so its output lines up pixel-for-pixel with every other grid file. It gets these coordinates from one of two sources:

- `--namelist path/to/init_template.nml` — reads the namelist file used to run the model, finds the `population_mapsgrid_file` key, and loads the lat/lon coordinates from that population NetCDF.
- `--pop-grid path/to/population.nc` — reads the population NetCDF directly.
- `--dry-run` with neither flag — uses a built-in synthetic 4×5 grid (for format testing only; no real geography).

The result is two 1-D arrays: `maps_lat` and `maps_lon`.

---

### Step 2 — Sample Weather at Coarse Resolution

Making one API call per MAPS grid cell would be slow and unnecessary — weather changes slowly in space. Instead the script builds a coarser sampling grid (default 1° spacing) that covers the continental US bounding box (24°–50°N, 126°–65°W).

It queries the **Open-Meteo API** at each coarse point to retrieve three daily variables:

| Variable | Meaning |
|---|---|
| `temperature_2m_mean` | Daily mean temperature at 2 m above ground (°C) |
| `relativehumidity_2m_mean` | Daily mean relative humidity (%) |
| `uv_index_max` | Daily maximum UV index |

**Which endpoint gets called depends on the date:**
- Dates more than 5 days in the past → `archive-api.open-meteo.com` (ERA5-based reanalysis)
- More recent dates → `api.open-meteo.com` (forecast)

If a point returns no data, the cell is filled with a neutral factor of 1.0.

---

### Step 3 — Convert to Absolute Humidity

Relative humidity (%) alone is not a good transmission predictor because it depends on temperature. The script converts to **absolute humidity** (g/m³) using the Magnus formula:

```
saturation_vapour_pressure = 6.112 × exp(17.67 × T / (T + 243.5))   [hPa]
actual_vapour_pressure     = (RH / 100) × saturation_vapour_pressure
absolute_humidity          = 216.7 × actual_vapour_pressure / (273.15 + T)
```

Absolute humidity directly controls how long virus particles survive in aerosols — this is the scientific basis established by Shaman & Kohn (2009).

---

### Step 4 — Compute Three Sub-Factors

Each of the three variables is converted to a dimensionless sub-factor centred at 1.0. All three are clamped to [0.50, 1.50].

**Humidity sub-factor (`f_humidity`)**
```
f_humidity = 1.0 − 0.040 × (AH − 7.0)
```
Reference: AH = 7.0 g/m³ (US annual mean). Lower AH → longer aerosol survival → factor above 1.0.

**UV sub-factor (`f_uv`)**
```
f_uv = 1.0 − 0.025 × (UV − 3.0)
```
Reference: UV index 3 (moderate, spring/autumn). Higher UV kills virus particles faster → factor below 1.0.

**Temperature sub-factor (`f_temp`)**
```
f_temp = 1.0 − 0.010 × (T − 10.0)
```
Reference: 10°C. Warmer temperatures reduce indoor crowding and accelerate viral surface decay → factor below 1.0.

---

### Step 5 — Combine into a Single Factor

The three sub-factors are combined as a weighted sum:

```
meteo_beta_factor = (w_AH × f_humidity + w_UV × f_uv + w_T × f_temp)
                    ─────────────────────────────────────────────────
                              w_AH + w_UV + w_T
```

Default weights: humidity 0.50, UV 0.30, temperature 0.20. These can be overridden via CLI flags (`--humidity-weight`, `--uv-weight`, `--temp-weight`). The weights are normalised internally so they do not need to sum to 1.0.

The result is clamped to [0.50, 1.50] per cell.

---

### Step 6 — Interpolate to the Full MAPS Grid

The coarse sample grid is **bilinearly interpolated** to the full MAPS lat/lon grid using `scipy.interpolate.RegularGridInterpolator`. This upsamples smoothly without introducing artificial sharp edges.

Points that fall outside the sample domain (e.g., ocean cells just outside the bounding box) are extrapolated from the nearest edge, then re-clamped to [0.50, 1.50].

---

### Step 7 — Write the Output NetCDF

The final grid is written to a NetCDF file that matches the exact format the IC pre-processor's `read_grid_and_var()` function expects:

| Property | Value |
|---|---|
| Dimensions | `lat`, `lon` |
| Coordinate variables | `lat` (float64), `lon` (float64) |
| Data variable | `meteo_beta_factor` (float32) |
| Compression | zlib |
| Data variable count | exactly 1 (required by `read_grid_and_var`) |

**Why pre-fill NaN with 1.0?** The IC pre-processor replaces any NaN values with 0.0 when loading a grid. A 0.0 multiplier would zero out beta entirely for that cell — catastrophic. Pre-filling with 1.0 (neutral) means missing data has no effect.

**Why no `valid_range` attribute?** The `netCDF4` Python library treats `valid_range` as a masking instruction. When the IC pre-processor calls `np.ma.filled(data, 0.0)`, masked values would become 0.0, corrupting the multiplier. The range is instead documented in the variable's `comment` attribute.

---

## Running the Script

**Dry run (format testing, no network):**
```bash
python3 meteorological_factor.py \
    --init-date 2026-05-17 \
    --output maps_meteo_2026-05-17.nc \
    --dry-run
```

**Real run from a namelist file:**
```bash
python3 meteorological_factor.py \
    --init-date 2026-05-17 \
    --output MAPS_FORECASTS/initial_conditions/maps_meteo_2026-05-17.nc \
    --namelist MAPS_FORECASTS/init_template.nml
```

**Real run from a population grid directly:**
```bash
python3 meteorological_factor.py \
    --init-date 2026-05-17 \
    --output maps_meteo_2026-05-17.nc \
    --pop-grid MAPS_FORECASTS/static_grids/population.nc \
    --resolution 1.0 \
    --humidity-weight 0.50 \
    --uv-weight 0.30 \
    --temp-weight 0.20
```

---

## CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--init-date` | Yes | — | Forecast date (`YYYY-MM-DD`) |
| `--output` | Yes | — | Path for the output `.nc` file |
| `--namelist` | No* | — | Fortran `.nml` file to read grid from |
| `--pop-grid` | No* | — | Population NetCDF to read grid from |
| `--resolution` | No | `1.0` | API sampling resolution in degrees |
| `--humidity-weight` | No | `0.50` | Weight for humidity sub-factor |
| `--uv-weight` | No | `0.30` | Weight for UV sub-factor |
| `--temp-weight` | No | `0.20` | Weight for temperature sub-factor |
| `--request-delay` | No | `0.06` | Seconds between Open-Meteo API calls |
| `--dry-run` | No | off | Synthetic data, no network calls |

*`--namelist` or `--pop-grid` is required unless `--dry-run` is used alone.

---

## Dependencies

| Package | When Needed |
|---|---|
| `numpy` | Always |
| `netCDF4` | Always |
| `requests` | Real run only (not dry-run) |
| `scipy` | Real run only (not dry-run) |

Install: `pip install requests scipy`
