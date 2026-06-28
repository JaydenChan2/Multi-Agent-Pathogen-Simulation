# Implementation Plan: Meteorological & Infrastructure Quality Factors

## Overview

This document outlines the plan to incorporate two new environmental factors into the MAPS (Multi-Agent Pathogen Simulation) model:

1. **Meteorological Factor** — weather conditions (temperature, humidity, UV) that affect viral survival and transmission rates
2. **Infrastructure Quality Factor** — expanded infrastructure coverage (schools, libraries, public transit, retail, etc.) that affects per-cell contact rates

Both factors translate into spatially-resolved grid multipliers that modulate the base transmission rate (`beta`) before it reaches the Fortran simulation core.

---

## 1. Meteorological Factor

### Scientific Basis

Respiratory pathogen transmission is strongly affected by ambient conditions:

| Variable | Effect on Transmission |
|---|---|
| Low absolute humidity | Longer aerosol survival → higher beta |
| Temperature 5–15°C | Optimal viral stability in aerosols |
| High UV index | Faster viral inactivation → lower beta |
| Precipitation / cloudiness | Drives people indoors → higher contact rate |
| Wind speed | Dilutes outdoor aerosol concentration |

This is well-established in epidemiological literature (Shaman & Kohn 2009; Kissler et al. 2020). The model should compute a `meteo_beta_factor(lat, lon, t)` — a dimensionless grid multiplier applied to `beta0` for each simulation day.

### Recommended Formula

```
meteo_factor = w_AH * f_humidity(AH)
             + w_UV * f_uv(UV_index)
             + w_T  * f_temp(T_celsius)

beta_effective = beta0 * meteo_factor
```

All three sub-functions normalize to 1.0 at neutral conditions so the multiplier is interpretable. Weights (`w_*`) are new namelist parameters.

### Recommended APIs

| API | What It Provides | Cost | Notes |
|---|---|---|---|
| **Open-Meteo** (primary) | Temperature, relative humidity, precipitation, UV index — hourly, 1 km resolution, global | Free, no key | Best choice: no registration, generous rate limits, historical reanalysis back to 1940 |
| **NASA POWER API** | Daily climate averages at 0.5° resolution, including surface solar radiation | Free, no key | Good for UV proxy (surface downwelling SW) |
| **NOAA Climate Data Online** | US station observations, gridded PRISM-style products | Free, requires API key | Useful for precipitation and temperature validation |
| **ERA5 via CDS API** | Gold-standard reanalysis, global 0.25° resolution, all variables | Free but requires ECMWF account | Best for historical runs; slower to access |

**Recommendation:** Use Open-Meteo for real-time and near-forecast use cases; add ERA5 as a fallback for historical simulations.

Open-Meteo endpoint example:
```
https://api.open-meteo.com/v1/forecast?
  latitude={lat}&longitude={lon}
  &daily=temperature_2m_mean,precipitation_sum,uv_index_max
  &hourly=relativehumidity_2m
  &start_date={YYYY-MM-DD}&end_date={YYYY-MM-DD}
```

---

## 2. Infrastructure Quality Factor

### Scientific Basis

The current model accounts for disease spread via transportation hubs (airports, transit) and hospitals. However, high-contact community settings drive a significant share of transmission:

| Infrastructure Type | Transmission Role |
|---|---|
| Schools (K–12, universities) | High-density, sustained child/young-adult contact |
| Libraries | Indoor, often enclosed, cross-demographic mixing |
| Public transit (buses, subways) | High-volume, enclosed, short-duration contacts |
| Retail and commercial | Moderate-volume, indoor, recurring visits |
| Community / recreation centers | Sustained indoor gathering |
| Religious facilities | Weekly high-density indoor events |
| Restaurants / food service | Indoor dining, reduced ventilation |

The goal is to produce a per-grid-cell `infrastructure_contact_multiplier` — a scalar > 1.0 in infrastructure-dense areas that inflates the effective contact rate:

```
beta_effective = beta0 * infrastructure_factor(lat, lon)
```

### Infrastructure Score Formula

```
infra_score(cell) = sum over types k:  weight_k * density_k(cell)
```

Where `density_k` is the count or capacity of infrastructure type `k` per unit population in the cell. The score is then normalized to a multiplier centered at 1.0:

```
infra_multiplier = 1.0 + alpha * (infra_score - mean_infra_score) / std_infra_score
```

`alpha` is a new namelist parameter controlling how strongly infrastructure density modulates transmission.

### Recommended APIs / Data Sources

| Source | What It Provides | Cost | Notes |
|---|---|---|---|
| **OpenStreetMap via Overpass API** (primary) | POI counts by type (schools, libraries, transit stops, etc.) at any geographic bounding box | Free | Most complete open dataset; Python library `overpy` simplifies queries |
| **US Department of Education NCES** | K–12 school locations, enrollment counts (authoritative) | Free CSV download | Covers all US public and private schools |
| **National Center for Education Statistics (IPEDS)** | University/college locations and enrollment | Free CSV download | Pairs with NCES for full education coverage |
| **US Transit Agency GTFS Feeds** | Transit stop locations and ridership (standardized format) | Free (aggregated at transit.land) | Accurate stop-level transit density |
| **Google Places API** | POI density, operating hours, popularity | Paid ($17/1000 requests) | Most comprehensive but costly at national scale |
| **SafeGraph / Veraset (academic)** | Anonymized mobility and visit patterns by venue | Free for academic research | Provides actual visit counts, not just facility existence |

**Recommendation:** Use OSM Overpass for the initial build (free, comprehensive), supplement with NCES for schools (authoritative enrollment data), and GTFS feeds for transit. Add SafeGraph if academic access is available, as it captures actual utilization rather than raw facility count.

Python library for OSM:
```python
pip install overpy
```

Example query for schools in a bounding box:
```python
import overpy
api = overpy.API()
result = api.query("""
  [out:json];
  node["amenity"="school"](40.0,-80.0,41.0,-79.0);
  out body;
""")
```

---

## 3. File Structure Decision

### Create New Files (Recommended)

The two new factors involve distinct data pipelines (API calls, rasterization, normalization) that are independent of the core initial-condition builder. Mixing them into `build_maps_initial_conditions_with_history_full.py` would make that file significantly harder to maintain and debug.

**New files to create:**

| File | Purpose |
|---|---|
| `fetch_meteorological_data.py` | Pulls weather data for a date range and bounding box, rasterizes to the MAPS grid, writes a NetCDF file (`maps_meteo_{date}.nc`) |
| `build_infrastructure_grid.py` | Queries OSM / NCES / GTFS, computes per-cell infrastructure scores, writes a NetCDF file (`maps_infrastructure.nc`) |

**Existing files to modify:**

| File | What Changes |
|---|---|
| `build_maps_initial_conditions_with_history_full.py` | Add loading of the two new NetCDF grids; compute combined `beta_modifier_grid = meteo_factor_grid * infra_multiplier_grid`; pass into `write_init_nc` as new variables |
| `edit_namelist.py` | Add writes for new namelist parameters: `meteo_grid_file`, `infrastructure_grid_file`, `meteo_beta_weight`, `infra_contact_weight` |
| `SCRIPTS.md` | Document the two new scripts and their namelist parameters |

`build_infrastructure_grid.py` only needs to run once (or infrequently) since infrastructure changes slowly. `fetch_meteorological_data.py` runs per forecast cycle, like the IC builder.

---

## 4. Step-by-Step Implementation Plan

### Phase 1 — Meteorological Pipeline

**Step 1.1 — Create `fetch_meteorological_data.py`**

- Accept CLI args: `--start-date`, `--end-date`, `--namelist` (to get the grid file for lat/lon bounds)
- Read the existing `population_mapsgrid_file` to extract the lat/lon grid
- For each grid cell (or a coarser resolution with bilinear interpolation), call Open-Meteo to fetch: `temperature_2m_mean`, `relativehumidity_2m`, `uv_index_max`, `precipitation_sum`
- Convert relative humidity + temperature to absolute humidity:
  ```
  AH = (RH/100) * 6.112 * exp(17.67 * T / (T + 243.5)) * 2.1674 / (273.15 + T)
  ```
- Compute `meteo_factor` per cell per day using the formula in Section 1
- Write output NetCDF: dimensions `(lat, lon, day)`, variables `meteo_beta_factor`, `temperature`, `abs_humidity`, `uv_index`

**Step 1.2 — Add namelist parameters**

Add to `init_template.nml`:
```fortran
meteo_grid_file = 'MAPS_FORECASTS/initial_conditions/maps_meteo_YYYY-MM-DD.nc',
meteo_beta_weight = 0.30,   ! how strongly weather modulates beta (0 = off)
meteo_humidity_weight = 0.50,
meteo_uv_weight = 0.30,
meteo_temp_weight = 0.20,
```

**Step 1.3 — Integrate into `build_maps_initial_conditions_with_history_full.py`**

- Add `parse_simple_namelist` reads for the new keys
- Add `load_meteo_grid(path, init_date)` function — reads the NetCDF and extracts the single-day slice for `init_date`
- Multiply `beta0` estimates by the spatial mean of `meteo_beta_factor` when writing the namelist fragment
- Include `meteo_beta_factor` as a new variable in `write_init_nc` so the Fortran model can use it day-by-day

---

### Phase 2 — Infrastructure Pipeline

**Step 2.1 — Create `build_infrastructure_grid.py`**

- Accept CLI args: `--namelist` (for grid bounds), `--output`
- Define infrastructure type weights (tunable via CLI or config):
  ```python
  INFRA_WEIGHTS = {
      "school":        1.5,   # high contact rate
      "university":    1.3,
      "library":       0.8,
      "transit_stop":  1.2,
      "supermarket":   0.9,
      "community_centre": 1.0,
      "place_of_worship": 0.7,
      "restaurant":    0.6,
  }
  ```
- Query OSM Overpass for each type across the US bounding box (tile the queries to avoid timeouts)
- Supplement schools with NCES data (CSV download — more accurate enrollment counts)
- Rasterize POI counts onto the MAPS lat/lon grid; normalize by population to get per-capita density
- Compute `infra_score` and `infra_multiplier` as described in Section 2
- Write `maps_infrastructure.nc`: dimensions `(lat, lon)`, variables `infra_multiplier`, `infra_score`, per-type density layers

**Step 2.2 — Add namelist parameters**

Add to `init_template.nml`:
```fortran
infrastructure_grid_file = 'MAPS_FORECASTS/static_grids/maps_infrastructure.nc',
infra_contact_weight = 0.20,   ! how strongly infrastructure modulates beta (0 = off)
```

**Step 2.3 — Integrate into `build_maps_initial_conditions_with_history_full.py`**

- Add `load_infrastructure_grid(path)` function
- Combine factors into a single `env_modifier_grid`:
  ```python
  env_modifier_grid = (
      meteo_weight * meteo_factor_grid
      + infra_weight * infra_multiplier_grid
  ) / (meteo_weight + infra_weight)
  ```
- Include `env_modifier_grid` in the init NetCDF output under variable name `env_beta_modifier`
- Log summary statistics (mean, std, min, max) for both grids so runs are auditable

---

### Phase 3 — Fortran Model Integration (Downstream)

This phase involves the MAPS Fortran model itself, which is not in this repository but consumes the NetCDF output.

- Add `env_beta_modifier(lat, lon)` as an optional read in the Fortran IC loader
- Apply it each timestep: `beta_t = beta0 * env_modifier(lat, lon)` (or per-day for the meteo time series)
- New namelist parameters `meteo_beta_weight` and `infra_contact_weight` should be forwarded from `edit_namelist.py`

---

## 5. Summary of New Dependencies

```
pip install overpy          # OSM Overpass queries
pip install requests        # Open-Meteo API calls
pip install scipy           # bilinear interpolation for grid resampling
# netCDF4, numpy already present
```

For NCES school data: manual CSV download from https://nces.ed.gov/ccd/ (no API key needed).  
For GTFS transit: bulk feed download from https://transit.land/ or individual agency GTFS feeds.

---

## 6. Testing Strategy

| Test | What to Verify |
|---|---|
| Meteorological | `meteo_factor` stays within [0.5, 2.0]; summer vs. winter grids visually differ |
| Infrastructure | High-density urban cells (NYC, LA) score higher than rural cells |
| Combined modifier | `env_beta_modifier` mean ≈ 1.0 (no artificial inflation/deflation of national R) |
| Integration | `write_init_nc` output contains new variables and passes existing compartment closure checks |
| Namelist | `edit_namelist.py` writes new keys without breaking existing Fortran model reads |

## Notes:
We should incorporate how these factors would be ranked in the priority/tiering system, as of now he has a few factors with effect that do not have equal worth on the calculation for the beta; it's a weighted average based on ranking in a priority tier list that how heavily the beta is changed. 
Also DEFINITELY need his help because ForTran is doing the hard math with all the numbers made by the python backend. So he would have to do some fortran nonsense to make sure that our numbers aren't just going into nothing.
