# meteorological_factor.py — Pipeline Diagram

## External Dependencies

| Dependency | Type | Purpose | Cost / Key |
|---|---|---|---|
| **Open-Meteo Archive API** | REST API | Historical ERA5 reanalysis — temperature, RH, UV (dates > 5 days ago) | Free, no key |
| **Open-Meteo Forecast API** | REST API | Near-real-time forecast — same variables (dates within last 5 days) | Free, no key |
| **scipy** `RegularGridInterpolator` | Python library | Bilinear interpolation from coarse sample grid to full MAPS grid | `pip install scipy` |
| **netCDF4** | Python library | Read population reference grid; write output `.nc` file | `pip install netCDF4` |
| **numpy** | Python library | Array math, Magnus formula, clamping | `pip install numpy` |
| **requests** | Python library | HTTP calls to Open-Meteo | `pip install requests` |
| **MAPS population NetCDF** | Local file | Provides the lat/lon reference grid so the output is pixel-aligned with all other MAPS grids | Produced by the IC pre-processor |

---

## Full Pipeline Flowchart

```mermaid
flowchart TD

    START([▶ python meteorological_factor.py]) --> ARGS

    subgraph ARGS["1 · Parse CLI Arguments"]
        A1[/"--init-date  YYYY-MM-DD"\]
        A2[/"--output  path/to/maps_meteo_DATE.nc"\]
        A3[/"--resolution  degrees  default 1.0"\]
        A4[/"--humidity-weight  --uv-weight  --temp-weight  WEIGHT"\]
        A5[/"--dry-run  flag"\]
    end

    ARGS --> GRID_SRC

    subgraph GRID_SRC["2 · Resolve Reference Lat/Lon Grid"]
        GS1{Source?}
        GS1 -->|"--dry-run only\nno file needed"| GS2["Built-in synthetic\n4×5 grid\n32–44°N, 120–100°W"]
        GS1 -->|"--namelist\ninit_template.nml"| GS3["Parse &init_nml block\nread population_mapsgrid_file key\nopen that NetCDF"]
        GS1 -->|"--pop-grid\npath to .nc"| GS4["Open population\nNetCDF directly"]
        GS3 --> GS5["Extract lat[ ] and lon[ ]\narrays  float64"]
        GS4 --> GS5
        GS2 --> GS5
    end

    GS5 --> MODE

    subgraph MODE["3 · Mode Decision"]
        M1{"--dry-run?"}
    end

    MODE -->|yes| DRYRUN
    MODE -->|no| DEPCHECK

    subgraph DRYRUN["DRY-RUN PATH — no network calls"]
        DR1["make_dry_run_factor_grid\nlat-gradient + Gaussian noise\nseed=42 for reproducibility"]
        DR2["Values clipped to 0.75–1.25\nMimics a mild winter pattern\nhigher latitude = slightly higher factor"]
        DR1 --> DR2
    end

    subgraph DEPCHECK["4 · Dependency Check"]
        DC1{"requests\ninstalled?"}
        DC1 -->|no| DC_ERR["EXIT — pip install requests"]
        DC1 -->|yes| DC2{"scipy\ninstalled?"}
        DC2 -->|no| DC_ERR2["EXIT — pip install scipy"]
        DC2 -->|yes| SAMPLE
    end

    subgraph SAMPLE["5 · Build Coarse Sampling Grid"]
        S1["Clip bounding box to\nUS extent: 24–50°N, 126–65°W"]
        S2["np.arange at --resolution degrees\ne.g. 3° → 9 lat × 21 lon = 189 points"]
        S1 --> S2
    end

    SAMPLE --> FETCH

    subgraph FETCH["6 · Fetch Weather Data — one call per sample point"]
        F1{"Date more than\n5 days ago?"}
        F1 -->|yes| F2["Open-Meteo ARCHIVE endpoint\narchive-api.open-meteo.com/v1/archive\nERA5 reanalysis data"]
        F1 -->|no| F3["Open-Meteo FORECAST endpoint\napi.open-meteo.com/v1/forecast\nNWP model output"]
        F2 --> F4["Request daily variables\ntemperature_2m_mean  °C\nrelative_humidity_2m_mean  %\nuv_index_max"]
        F3 --> F4
        F4 --> F5{"UV returned\nby API?"}
        F5 -->|yes| F6["Use API UV value"]
        F5 -->|no| F7["Solar angle fallback\ncos latitude − 23.5·cos day-of-year\nscaled to 0–8 UV index range"]
        F6 --> F8["Store T, RH, UV\nfor this point"]
        F7 --> F8
        F8 --> F9["Sleep --request-delay seconds\nbetween calls  default 0.06s\nrespects Open-Meteo rate limits"]
    end

    FETCH --> PHYSICS

    subgraph PHYSICS["7 · Meteorological Physics — per sample point"]
        P1["Magnus formula → Absolute Humidity\nAH = 216.7 × RH/100 × 6.112 × exp·17.67T÷T+243.5 ÷ 273.15+T\nResult in g m⁻³"]
        P2["f_humidity AH\n1.0 − 0.040 × AH − 7.0\nClamped 0.50–1.50"]
        P3["f_uv UV\n1.0 − 0.025 × UV − 3.0\nClamped 0.50–1.50"]
        P4["f_temp T\n1.0 − 0.010 × T − 10.0\nClamped 0.50–1.50"]
        P5["compute_meteo_factor\nweighted blend  WEIGHT\nw_ah·f_hum + w_uv·f_uv + w_t·f_temp\n÷ w_ah+w_uv+w_t\nDefault weights: 0.50 / 0.30 / 0.20"]
        P1 --> P2
        P1 --> P3
        P1 --> P4
        P2 --> P5
        P3 --> P5
        P4 --> P5
    end

    PHYSICS --> INTERP

    subgraph INTERP["8 · Bilinear Interpolation — scipy"]
        I1["scipy RegularGridInterpolator\nmethod=linear\nbounds_error=False"]
        I2["Map coarse sample grid\nonto full MAPS lat×lon grid\n~thousands of cells"]
        I3["Clip interpolated values\nto 0.50–1.50"]
        I1 --> I2 --> I3
    end

    DRYRUN --> NANFILL
    INTERP --> NANFILL

    subgraph NANFILL["9 · NaN Safety Pre-fill"]
        N1["Replace any remaining\nNaN or inf with 1.0\nnot 0.0"]
        N2["The IC pre-processor reader\nfills NaN with 0.0\nA 0.0 multiplier zeros out beta\n1.0 is neutral — no effect on beta"]
        N1 --> N2
    end

    NANFILL --> WRITE

    subgraph WRITE["10 · Write NetCDF — maps_meteo_DATE.nc"]
        W1["Dimensions: lat, lon"]
        W2["lat  float64  degrees_north\nlon  float64  degrees_east"]
        W3["meteo_beta_factor  float32  lat×lon\nzlib compressed\nExactly ONE data variable\nso read_grid_and_var passes its guard"]
        W4["Global attributes\ntitle, source, conventions CF-1.8"]
        W1 --> W2 --> W3 --> W4
    end

    WRITE --> APPLY

    subgraph APPLY["11 · Apply to Beta — edit_namelist.py reads this file"]
        AP1["load_spatial_mean reads maps_meteo_DATE.nc\ncollapses grid to one scalar mean"]
        AP2["apply_meteo_to_beta\nnew_beta = base × 1.0 + 0.60 × mean − 1.0\nWEIGHT = 0.60  METEO_BETA_WEIGHT"]
        AP3["Passed to Fortran model\nvia the namelist .nml file"]
        AP1 --> AP2 --> AP3
    end

    WRITE --> END([✓ Done])
```

---

## Notes on Non-Intuitive Steps

### Why absolute humidity instead of relative humidity?
Aerosol survival — how long virus-laden particles stay airborne — depends on the **actual water vapour content** of the air, not how saturated the air is relative to its capacity. Two days can have the same relative humidity but very different absolute humidity if temperatures differ. Cold winter air at 70% RH actually holds far less water vapour than warm summer air at 70% RH. Since aerosol survival drives airborne transmission, absolute humidity is the physically correct variable. The Magnus formula converts temperature + relative humidity into this more meaningful quantity.

### Why sample at coarse resolution and interpolate, rather than querying every MAPS cell?
The MAPS grid can have thousands of cells. Making one API call per cell would take hours and hit rate limits. Instead, we sample at 1–3° spacing (~100–200 points), compute the physics at each sample, then use bilinear interpolation (scipy) to fill in the full fine-resolution grid. This is the same approach used in numerical weather prediction downscaling.

### Why two different Open-Meteo endpoints?
Open-Meteo's ERA5 reanalysis (archive) data has a processing lag of approximately 5 days — recent days are not yet ingested. For dates within that lag window the script automatically switches to the near-real-time forecast endpoint instead. The 5-day threshold is `_ARCHIVE_LAG_DAYS = 5`.

### Why does UV fall back to a solar angle calculation?
The ERA5 reanalysis does not always include a UV index variable, particularly for older dates. The fallback computes a physically reasonable estimate from the cosine of the solar declination angle, which captures the dominant drivers of UV (latitude and season). It is less accurate than a measured or modelled UV value but is far better than defaulting to the neutral reference value for all missing points.

### Why pre-fill NaN with 1.0 and not 0.0?
The `read_grid_and_var()` function in the IC pre-processor replaces NaN/masked values with **0.0** after reading. If a grid cell were 0.0, multiplying it into beta would zero out transmission entirely for that cell — an obviously wrong result. Pre-filling with **1.0** (the neutral multiplier) before writing means the reader's fill has no effect on cells where the API returned no data.

### What do the WEIGHT values mean?
There are two layers of weighting:
- **Sub-factor weights** (`w_ah=0.50, w_uv=0.30, w_t=0.20`): humidity drives 50% of the composite factor, UV drives 30%, temperature 20%. These reflect scientific literature (Shaman & Kohn 2009 emphasises humidity most strongly).
- **Whole-factor attenuation** (`METEO_BETA_WEIGHT=0.60`): even after computing the composite factor, only 60% of its deviation from neutral is applied to beta. This prevents weather from dominating over variant-specific biology. At 0.60, a 10% weather signal raises beta by 6%.

---

## Output File Schema

```
maps_meteo_YYYY-MM-DD.nc
├── Dimensions
│   ├── lat  (n_lat)
│   └── lon  (n_lon)
└── Variables
    ├── lat             float64  (lat,)   degrees_north
    ├── lon             float64  (lon,)   degrees_east
    └── meteo_beta_factor  float32  (lat, lon)  zlib=True
        ├── long_name  : "dimensionless transmission-rate multiplier ..."
        ├── units      : "1"
        └── comment    : "init_date=...; range=[0.5, 1.5]"
```

Values > 1.0 → conditions favour transmission (cold, dry, low UV)
Values < 1.0 → conditions suppress transmission (warm, humid, high UV)
Value = 1.0 → neutral (reference conditions, no effect on beta)
