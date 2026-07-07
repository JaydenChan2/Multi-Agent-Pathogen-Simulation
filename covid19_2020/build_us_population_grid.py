#!/usr/bin/env python3
"""
build_us_population_grid.py

Creates a synthetic but demographically realistic US population grid at
0.5-degree resolution for the COVID-19 2020 simulation. Population density
is modelled as a mixture of Gaussian "urban attractors" anchored to the 20
largest US metro areas (2020 Census estimates), plus a low-density rural
baseline. Total sums to approximately 328 million (US 2020 Census).

Output: covid19_2020/us_population_2020.nc
"""
from __future__ import annotations
import numpy as np
from netCDF4 import Dataset
from pathlib import Path

# 0.5-degree grid covering the continental US
LAT = np.arange(24.25, 50.0, 0.5)   # 52 rows
LON = np.arange(-124.75, -65.0, 0.5) # 120 cols

# Major metro areas: (lat, lon, population_millions, spread_deg)
# 2020 Census metro populations, approximate centroids
METROS = [
    # name                lat     lon    pop_M  spread
    ("New York",         40.71, -74.00,  20.1,  1.2),
    ("Los Angeles",      34.05, -118.25, 13.2,  1.5),
    ("Chicago",          41.88, -87.63,   9.6,  1.0),
    ("Dallas",           32.78, -96.80,   7.7,  1.2),
    ("Houston",          29.76, -95.37,   7.1,  1.2),
    ("Washington DC",    38.90, -77.04,   6.4,  1.0),
    ("Miami",            25.77, -80.19,   6.2,  1.0),
    ("Philadelphia",     39.95, -75.16,   6.1,  0.8),
    ("Atlanta",          33.75, -84.39,   6.0,  1.0),
    ("Phoenix",          33.45, -112.07,  4.9,  1.0),
    ("Boston",           42.36, -71.06,   4.9,  0.8),
    ("Riverside CA",     33.98, -117.37,  4.6,  1.0),
    ("Seattle",          47.61, -122.33,  4.0,  0.9),
    ("Minneapolis",      44.98, -93.27,   3.6,  0.8),
    ("San Diego",        32.72, -117.16,  3.3,  0.7),
    ("Tampa",            27.95, -82.46,   3.2,  0.7),
    ("Denver",           39.74, -104.98,  2.9,  0.8),
    ("St Louis",         38.63, -90.20,   2.8,  0.7),
    ("Baltimore",        39.29, -76.61,   2.8,  0.6),
    ("Portland OR",      45.52, -122.68,  2.5,  0.7),
    ("San Francisco",    37.77, -122.42,  4.7,  0.8),
    ("Detroit",          42.33, -83.05,   4.3,  0.8),
    ("Las Vegas",        36.17, -115.14,  2.2,  0.6),
    ("Memphis",          35.15, -90.05,   1.3,  0.5),
    ("New Orleans",      29.95, -90.07,   1.3,  0.5),
    ("Nashville",        36.17, -86.78,   2.0,  0.7),
    ("Kansas City",      39.10, -94.58,   2.2,  0.7),
    ("Columbus OH",      39.96, -82.99,   2.1,  0.6),
    ("Indianapolis",     39.77, -86.16,   2.1,  0.6),
    ("Charlotte",        35.23, -80.84,   2.6,  0.7),
]

grid_lat, grid_lon = np.meshgrid(LAT, LON, indexing="ij")
pop = np.zeros_like(grid_lat)

# Add each metro as a 2D Gaussian
for _, mlat, mlon, pop_m, spread in METROS:
    d2 = ((grid_lat - mlat) / spread) ** 2 + ((grid_lon - mlon) / spread) ** 2
    pop += pop_m * 1e6 * np.exp(-0.5 * d2)

# Rural baseline: ~60 people/km² across non-coastal interior
rural = np.full_like(pop, 15_000.0)
# Mask ocean (very rough): west coast and east coast thin out
pop = pop + rural

# Clip ocean and Canada / Mexico
pop[LAT < 25.5, :] *= 0.1   # near Mexico border
for i, lat_val in enumerate(LAT):
    if lat_val > 48.5:
        pop[i, :] *= 0.05   # near Canada border

# Scale to exactly 328 million total
pop = np.maximum(pop, 0.0)
pop *= 328_000_000.0 / pop.sum()

out = Path(__file__).parent / "us_population_2020.nc"
with Dataset(out, "w", format="NETCDF4") as ds:
    ds.createDimension("lat", len(LAT))
    ds.createDimension("lon", len(LON))

    lv = ds.createVariable("lat", "f8", ("lat",))
    lv[:] = LAT; lv.units = "degrees_north"

    lov = ds.createVariable("lon", "f8", ("lon",))
    lov[:] = LON; lov.units = "degrees_east"

    pv = ds.createVariable("population", "f4", ("lat", "lon"), zlib=True)
    pv[:, :] = pop.astype(np.float32)
    pv.long_name = "2020 US population per grid cell (synthetic)"
    pv.units = "persons"

    ds.title = "Synthetic US population grid for COVID-19 2020 simulation"
    ds.source = "Gaussian metro attractors anchored to 2020 US Census metro populations"
    ds.total_population = f"{pop.sum():.0f}"

print(f"Wrote: {out}")
print(f"Grid  : {len(LAT)} lat × {len(LON)} lon  (0.5° resolution)")
print(f"Total : {pop.sum()/1e6:.1f} million persons")
print(f"Max cell: {pop.max()/1e6:.3f} M  Min: {pop.min():.0f}")
