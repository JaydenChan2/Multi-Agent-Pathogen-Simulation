# infrastructure_grid.py — How It Works

## What This Script Does

`infrastructure_grid.py` produces a spatially-resolved contact-rate multiplier grid that tells the MAPS model how much community infrastructure — schools, transit, libraries, retail, etc. — is packed into each grid cell. The output is a static NetCDF file (`maps_infrastructure.nc`) containing a single 2-D array called `infra_multiplier`.

A value of **1.0** means average infrastructure density.  
A value **> 1.0** means denser-than-average infrastructure (urban cores, school districts).  
A value **< 1.0** means sparser-than-average infrastructure (rural cells).

---

## Where It Sits in the Pipeline

```
infrastructure_grid.py            →  maps_infrastructure.nc  (run ONCE; static)
         ↓
build_maps_initial_conditions_with_history_full.py   (IC pre-processor)
         ↓
edit_namelist.py                                     (per-ensemble-member editor)
         ↓
Fortran MAPS model
```

Unlike the meteorological script, this is a **static pre-computation** step. Infrastructure changes slowly, so you only need to re-run it when you want to refresh the underlying OpenStreetMap or school data — not for every forecast cycle.

---

## Step-by-Step Walkthrough

### Step 1 — Get the Reference Grid (and Population)

The script needs the MAPS lat/lon grid for two reasons: to align the output pixel-for-pixel with every other grid file, and to normalise infrastructure counts by population.

- `--namelist path/to/init_template.nml` — reads the namelist, finds `population_mapsgrid_file`, loads lat/lon and population data from that NetCDF.
- `--pop-grid path/to/population.nc` — reads the population NetCDF directly.
- `--dry-run` with neither flag — uses a built-in synthetic 4×5 grid for testing.

The population array is used later in Step 5 to convert raw POI counts into a per-capita density, which prevents large cities from dominating simply because of their raw numbers.

---

### Step 2 — Query OpenStreetMap for Infrastructure POIs

The continental US is too large for a single Overpass API query (it would time out). Instead the script tiles the bounding box (24°–50°N, 126°–65°W) into 5°×5° chunks and sends one query per tile.

Each query returns every node matching one of these infrastructure types:

| OSM Tag | Weight | Epidemiological Rationale |
|---|---|---|
| `amenity=school` | 1.50 | Dense, sustained child/young-adult contacts |
| `amenity=university` | 1.30 | High mixing across age groups |
| `amenity=bus_station` | 1.20 | High-volume, enclosed, short-duration contacts |
| `amenity=subway_entrance` | 1.20 | Same dynamics as bus station |
| `amenity=community_centre` | 1.00 | Sustained indoor gatherings |
| `amenity=supermarket` | 0.90 | Recurring high-volume indoor visits |
| `amenity=gym` | 0.80 | Sustained exertion, elevated aerosol risk |
| `amenity=library` | 0.80 | Indoor, cross-demographic, lower density |
| `amenity=place_of_worship` | 0.70 | Weekly high-density, typically brief |
| `amenity=restaurant` | 0.60 | Indoor dining, moderate dwell time |
| `amenity=hospital` | 0.50 | Partially covered by the existing baseline |
| `amenity=fast_food` | 0.40 | Shorter dwell than full-service dining |
| `shop=supermarket` | 0.90 | Duplicate OSM tagging convention for supermarkets |
| `leisure=fitness_centre` | 0.80 | Duplicate OSM tagging convention for gyms |
| `public_transport=stop_position` | 1.00 | Transit stops not captured by bus_station |

Weights are sourced from published contact-pattern matrices (Mossong et al. 2008; Hoang et al. 2019). A weight of 1.0 represents baseline community contact rate.

Each tile query retries up to 3 times on failure with a 10-second pause. If a tile still fails, it is skipped and a warning is printed — the overall build is not aborted.

There is a configurable delay between tile queries (default 2 seconds) to be polite to the public Overpass instance.

---

### Step 3 — (Optional) Supplement Schools with NCES Data

OSM `amenity=school` nodes carry no enrollment information — a 50-student rural school and a 3,000-student urban high school look the same. The `--nces-csv` flag accepts a CSV from the [NCES Common Core of Data](https://nces.ed.gov/ccd/) which provides per-school enrollment counts.

When supplied, each NCES school is added to the POI list with a weight of:
```
weight = 1.50 × (enrollment / 1000)
```
This means a 1,000-student school contributes the same weight as one OSM school node. A 2,000-student school contributes twice as much.

Required CSV columns: `LATCOD` (latitude), `LONCOD` (longitude), `MEMBER` (enrollment). Schools with missing coordinates or zero enrollment are skipped.

---

### Step 4 — Rasterise POIs onto the MAPS Grid

Each collected POI (lat, lon, weight) is assigned to its nearest MAPS grid cell by rounding to the nearest grid index. The weight is accumulated into that cell's total.

The result is a 2-D array `raw_density` of shape `(n_lat, n_lon)` where each cell holds the sum of weights of all POIs that mapped to it.

POIs that fall outside the grid extent are silently discarded.

---

### Step 5 — Normalise to a Centred Multiplier

Raw POI weight sums are not directly comparable across a city-to-rural gradient. The script converts to a dimensionless multiplier centred at 1.0 in four sub-steps:

**1. Per-capita density**
```
per_capita_density = raw_density / population
```
Cells with zero population are excluded from statistics and assigned a multiplier of 1.0.

**2. Z-score standardisation**
```
z = (per_capita_density − mean) / std
```
The mean and std are computed only over populated cells.

**3. Scale to a multiplier**
```
infra_multiplier = 1.0 + alpha × z
```
`alpha` (default 0.15) controls how strongly infrastructure density modulates the contact rate. With alpha=0.15, a cell that is 1 standard deviation above average gets a multiplier of 1.15.

**4. Clamp**
The final value is clamped to [0.70, 1.50] per cell. This prevents extreme outliers (e.g., Manhattan) from producing unrealistic transmission rates.

If all valid cells have identical density (zero variance), the script returns a flat 1.0 grid rather than dividing by zero.

---

### Step 6 — Write the Output NetCDF

The final grid is written to a NetCDF file that matches the exact format the IC pre-processor's `read_grid_and_var()` function expects:

| Property | Value |
|---|---|
| Dimensions | `lat`, `lon` |
| Coordinate variables | `lat` (float64), `lon` (float64) |
| Data variable | `infra_multiplier` (float32) |
| Compression | zlib |
| Data variable count | exactly 1 (required by `read_grid_and_var`) |

**Why pre-fill NaN with 1.0?** The IC pre-processor replaces NaN with 0.0 when loading a grid. A 0.0 multiplier would zero out contact rates for that cell. Pre-filling NaN with 1.0 (neutral) means missing data has no effect.

**Why no `valid_range` attribute?** The `netCDF4` Python library treats `valid_range` as a masking instruction. Masked values become 0.0 when the IC pre-processor calls `np.ma.filled(data, 0.0)`, which would corrupt multiplier cells. The actual range is documented in the variable's `comment` attribute instead.

---

## Running the Script

**Dry run (format testing, no network):**
```bash
python3 infrastructure_grid.py \
    --output maps_infrastructure.nc \
    --dry-run
```

**Real run from a namelist file:**
```bash
python3 infrastructure_grid.py \
    --output MAPS_FORECASTS/static_grids/maps_infrastructure.nc \
    --namelist MAPS_FORECASTS/init_template.nml
```

**Real run with NCES school data:**
```bash
python3 infrastructure_grid.py \
    --output maps_infrastructure.nc \
    --pop-grid MAPS_FORECASTS/static_grids/population.nc \
    --nces-csv /path/to/nces_ccd_schools.csv \
    --alpha 0.15
```

**Tuning the sensitivity:**
```bash
python3 infrastructure_grid.py \
    --output maps_infrastructure.nc \
    --pop-grid population.nc \
    --alpha 0.25        # stronger urban/rural contrast
    --tile-size 3.0     # smaller tiles (more requests, less timeout risk)
    --tile-delay 3.0    # slower query rate (be more polite to Overpass)
```

---

## CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--output` | Yes | — | Path for the output `.nc` file |
| `--namelist` | No* | — | Fortran `.nml` file to read grid and population from |
| `--pop-grid` | No* | — | Population NetCDF to read grid and population from |
| `--nces-csv` | No | — | NCES CCD school CSV for enrollment-weighted schools |
| `--alpha` | No | `0.15` | Z-score scaling factor for the multiplier |
| `--tile-size` | No | `5.0` | Overpass tile size in degrees |
| `--tile-delay` | No | `2.0` | Seconds between Overpass tile requests |
| `--dry-run` | No | off | Synthetic data, no network calls |

*`--namelist` or `--pop-grid` is required unless `--dry-run` is used alone.

---

## Choosing the `--alpha` Value

`alpha` is the primary tuning knob. It maps z-scores to multipliers:

| alpha | 1 SD above average → multiplier |
|---|---|
| 0.10 | 1.10 (subtle, conservative) |
| 0.15 | 1.15 (default, moderate) |
| 0.20 | 1.20 (stronger urban signal) |
| 0.30 | 1.30 (aggressive; verify against observed outbreak data) |

Start with the default (0.15) and adjust based on how much the model's R-effective changes between high-density and low-density cells in a historical validation run.

---

## Dependencies

| Package | When Needed |
|---|---|
| `numpy` | Always |
| `netCDF4` | Always |
| `requests` | Real run only (not dry-run) |

Install: `pip install requests`

`scipy` is **not** needed for this script (no interpolation — the rasterisation step uses direct nearest-cell assignment).
