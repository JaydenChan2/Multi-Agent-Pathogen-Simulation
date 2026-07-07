#!/usr/bin/env python3
"""
test_factor_accuracy.py — Unit tests for formula correctness in the new factor scripts.

Covers:
  - Magnus formula / absolute humidity conversion (known analytical values)
  - Each meteorological sub-factor at its reference point and at known off-reference inputs
  - Clamping behaviour under extreme inputs
  - Seasonal plausibility: winter conditions should produce factor > 1, summer < 1
  - Infrastructure rasterisation: POI → grid-cell assignment
  - Z-score normalisation: mean ≈ 1.0, alpha controls spread, zero-variance safety
  - NetCDF output schema: both write functions satisfy the read_grid_and_var contract

Run with:
    /Users/jaydenchan/maps/.venv/bin/python3 -m pytest tests/test_factor_accuracy.py -v
or:
    /Users/jaydenchan/maps/.venv/bin/python3 tests/test_factor_accuracy.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import meteorological_factor as mf
import infrastructure_grid as ig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_grid_and_var(path: Path):
    """Exact copy of read_grid_and_var from the IC pre-processor."""
    with Dataset(path, "r") as ds:
        lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        candidates = [n for n in ds.variables if n not in ("lat", "lon")]
        if len(candidates) != 1:
            raise ValueError(
                f"{path} must have exactly one data variable besides lat/lon; "
                f"found {candidates}"
            )
        var_name = candidates[0]
        raw = ds.variables[var_name][:]
        data = np.ma.filled(raw, 0.0) if np.ma.isMaskedArray(raw) else np.array(raw)
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isfinite(data), data, 0.0)
    return lat, lon, data, var_name


# ---------------------------------------------------------------------------
# Magnus formula / absolute humidity
# ---------------------------------------------------------------------------

class TestAbsoluteHumidity(unittest.TestCase):

    def test_warm_moderate_humidity(self):
        """T=20°C, RH=50% → AH ≈ 8.64 g/m³ (standard reference value)."""
        ah = mf.compute_absolute_humidity(20.0, 50.0)
        self.assertAlmostEqual(ah, 8.64, places=1,
                               msg="AH at 20°C / 50% RH should be ≈8.64 g/m³")

    def test_freezing_saturated(self):
        """T=0°C, RH=100% → AH ≈ 4.85 g/m³."""
        ah = mf.compute_absolute_humidity(0.0, 100.0)
        self.assertAlmostEqual(ah, 4.85, places=1,
                               msg="AH at 0°C / 100% RH should be ≈4.85 g/m³")

    def test_completely_dry(self):
        """RH=0% → AH = 0 regardless of temperature."""
        ah = mf.compute_absolute_humidity(30.0, 0.0)
        self.assertAlmostEqual(ah, 0.0, places=6,
                               msg="AH at RH=0% must be 0")

    def test_increases_with_temperature(self):
        """At fixed RH, AH rises monotonically with T."""
        temps = [-10, 0, 10, 20, 30]
        ahs = [mf.compute_absolute_humidity(t, 80.0) for t in temps]
        for i in range(len(ahs) - 1):
            self.assertLess(ahs[i], ahs[i + 1],
                            msg=f"AH should increase with T: T={temps[i]} → {temps[i+1]}")

    def test_increases_with_relative_humidity(self):
        """At fixed T, AH rises monotonically with RH."""
        rhs = [10, 30, 50, 70, 90]
        ahs = [mf.compute_absolute_humidity(20.0, rh) for rh in rhs]
        for i in range(len(ahs) - 1):
            self.assertLess(ahs[i], ahs[i + 1],
                            msg=f"AH should increase with RH: {rhs[i]}% → {rhs[i+1]}%")


# ---------------------------------------------------------------------------
# Humidity sub-factor
# ---------------------------------------------------------------------------

class TestHumiditySubfactor(unittest.TestCase):

    def test_reference_returns_one(self):
        """f_humidity at the reference AH (7.0 g/m³) must be exactly 1.0."""
        self.assertAlmostEqual(mf.f_humidity(7.0), 1.0, places=9)

    def test_high_ah_suppresses(self):
        """Humid air → shorter aerosol survival → factor below 1."""
        self.assertLess(mf.f_humidity(12.0), 1.0,
                        msg="High AH should give factor < 1.0")

    def test_low_ah_elevates(self):
        """Dry air → longer aerosol survival → factor above 1."""
        self.assertGreater(mf.f_humidity(2.0), 1.0,
                           msg="Low AH should give factor > 1.0")

    def test_known_value(self):
        """f_humidity(12.0) = 1.0 − 0.040 × (12.0 − 7.0) = 0.80."""
        self.assertAlmostEqual(mf.f_humidity(12.0), 0.80, places=9)

    def test_known_value_dry(self):
        """f_humidity(2.0) = 1.0 − 0.040 × (2.0 − 7.0) = 1.20."""
        self.assertAlmostEqual(mf.f_humidity(2.0), 1.20, places=9)


# ---------------------------------------------------------------------------
# UV sub-factor
# ---------------------------------------------------------------------------

class TestUVSubfactor(unittest.TestCase):

    def test_reference_returns_one(self):
        """f_uv at the reference UV index (3.0) must be exactly 1.0."""
        self.assertAlmostEqual(mf.f_uv(3.0), 1.0, places=9)

    def test_high_uv_suppresses(self):
        """Strong UV photodegrades virus → factor below 1."""
        self.assertLess(mf.f_uv(10.0), 1.0,
                        msg="High UV should give factor < 1.0")

    def test_low_uv_elevates(self):
        """Weak UV → slower inactivation → factor above 1."""
        self.assertGreater(mf.f_uv(0.5), 1.0,
                           msg="Low UV should give factor > 1.0")

    def test_known_value(self):
        """f_uv(10.0) = 1.0 − 0.025 × (10.0 − 3.0) = 0.825."""
        self.assertAlmostEqual(mf.f_uv(10.0), 0.825, places=9)


# ---------------------------------------------------------------------------
# Temperature sub-factor
# ---------------------------------------------------------------------------

class TestTempSubfactor(unittest.TestCase):

    def test_reference_returns_one(self):
        """f_temp at the reference temperature (10°C) must be exactly 1.0."""
        self.assertAlmostEqual(mf.f_temp(10.0), 1.0, places=9)

    def test_warm_suppresses(self):
        """Warm air → less crowding indoors, faster surface decay → factor below 1."""
        self.assertLess(mf.f_temp(30.0), 1.0,
                        msg="High T should give factor < 1.0")

    def test_cold_elevates(self):
        """Cold air → more crowding indoors, longer survival → factor above 1."""
        self.assertGreater(mf.f_temp(-5.0), 1.0,
                           msg="Low T should give factor > 1.0")

    def test_known_value(self):
        """f_temp(30.0) = 1.0 − 0.010 × (30.0 − 10.0) = 0.80."""
        self.assertAlmostEqual(mf.f_temp(30.0), 0.80, places=9)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

class TestClamping(unittest.TestCase):

    def test_humidity_upper_clamp(self):
        """Extremely dry conditions must not exceed FACTOR_MAX."""
        val = mf.f_humidity(-100.0)  # 1.0 - 0.040 * (-107) = 5.28 → clamp
        self.assertLessEqual(val, mf._FACTOR_MAX)
        self.assertAlmostEqual(val, mf._FACTOR_MAX, places=9)

    def test_humidity_lower_clamp(self):
        """Extremely humid conditions must not fall below FACTOR_MIN."""
        val = mf.f_humidity(1000.0)
        self.assertGreaterEqual(val, mf._FACTOR_MIN)
        self.assertAlmostEqual(val, mf._FACTOR_MIN, places=9)

    def test_uv_lower_clamp(self):
        """Blazing UV must not fall below FACTOR_MIN."""
        val = mf.f_uv(100.0)  # 1 - 0.025*(100-3) = 1 - 2.425 = -1.425 → clamp
        self.assertAlmostEqual(val, mf._FACTOR_MIN, places=9)

    def test_temp_upper_clamp(self):
        """Extreme cold must not exceed FACTOR_MAX."""
        val = mf.f_temp(-1000.0)
        self.assertAlmostEqual(val, mf._FACTOR_MAX, places=9)

    def test_composite_always_bounded(self):
        """compute_meteo_factor output must always lie in [FACTOR_MIN, FACTOR_MAX]."""
        cases = [
            (-50.0, 0.1, -50.0),   # cold, dry, no UV
            (50.0, 99.0, 50.0),    # hot, saturated, extreme UV
            (10.0, 50.0, 3.0),     # reference conditions
        ]
        for T, RH, UV in cases:
            AH = mf.compute_absolute_humidity(T, RH)
            factor = mf.compute_meteo_factor(AH, UV, T)
            self.assertGreaterEqual(factor, mf._FACTOR_MIN,
                                    msg=f"Factor below min at T={T}, RH={RH}, UV={UV}")
            self.assertLessEqual(factor, mf._FACTOR_MAX,
                                 msg=f"Factor above max at T={T}, RH={RH}, UV={UV}")


# ---------------------------------------------------------------------------
# Composite factor: reference point and seasonal plausibility
# ---------------------------------------------------------------------------

class TestCompositeMeteoFactor(unittest.TestCase):

    def test_reference_returns_one(self):
        """At AH=7.0, UV=3.0, T=10.0, composite factor must be exactly 1.0."""
        factor = mf.compute_meteo_factor(7.0, 3.0, 10.0)
        self.assertAlmostEqual(factor, 1.0, places=9,
                               msg="All sub-factors at reference → composite = 1.0")

    def test_winter_above_one(self):
        """Typical US winter (cold, dry, low UV) must produce factor > 1.0."""
        # T=-5°C, RH=50%, UV=1.0 — classic January Midwest
        AH = mf.compute_absolute_humidity(-5.0, 50.0)  # ≈1.7 g/m³
        factor = mf.compute_meteo_factor(AH, 1.0, -5.0)
        self.assertGreater(factor, 1.0,
                           msg="Cold dry winter should give factor > 1.0 (more transmission)")

    def test_summer_below_one(self):
        """Typical US summer (warm, humid, high UV) must produce factor < 1.0."""
        # T=28°C, RH=75%, UV=8.0 — typical July Southeast
        AH = mf.compute_absolute_humidity(28.0, 75.0)  # ≈20 g/m³
        factor = mf.compute_meteo_factor(AH, 8.0, 28.0)
        self.assertLess(factor, 1.0,
                        msg="Hot humid summer should give factor < 1.0 (less transmission)")

    def test_winter_greater_than_summer(self):
        """Winter factor must exceed summer factor."""
        AH_win = mf.compute_absolute_humidity(-5.0, 50.0)
        factor_winter = mf.compute_meteo_factor(AH_win, 1.0, -5.0)

        AH_sum = mf.compute_absolute_humidity(28.0, 75.0)
        factor_summer = mf.compute_meteo_factor(AH_sum, 8.0, 28.0)

        self.assertGreater(factor_winter, factor_summer,
                           msg="Winter factor must exceed summer factor")

    def test_weights_normalised(self):
        """Weights that don't sum to 1.0 should still produce 1.0 at reference."""
        factor = mf.compute_meteo_factor(7.0, 3.0, 10.0, w_ah=3.0, w_uv=2.0, w_t=5.0)
        self.assertAlmostEqual(factor, 1.0, places=9,
                               msg="Unnormalised weights at reference should still give 1.0")

    def test_zero_weights_returns_one(self):
        """If all weights are 0, factor defaults to 1.0 (neutral)."""
        factor = mf.compute_meteo_factor(7.0, 3.0, 10.0, w_ah=0.0, w_uv=0.0, w_t=0.0)
        self.assertEqual(factor, 1.0)


# ---------------------------------------------------------------------------
# Synthetic grid generators
# ---------------------------------------------------------------------------

class TestDryRunGrids(unittest.TestCase):

    def setUp(self):
        self.lat = np.linspace(30.0, 48.0, 10)
        self.lon = np.linspace(-120.0, -70.0, 15)

    def test_meteo_grid_shape(self):
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        self.assertEqual(grid.shape, (len(self.lat), len(self.lon)))

    def test_meteo_grid_dtype(self):
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        self.assertEqual(grid.dtype, np.float32)

    def test_meteo_grid_range(self):
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        self.assertGreaterEqual(float(grid.min()), 0.70,
                                msg="Dry-run meteo grid should not go below 0.70")
        self.assertLessEqual(float(grid.max()), 1.30,
                             msg="Dry-run meteo grid should not exceed 1.30")

    def test_meteo_grid_no_nan(self):
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        self.assertEqual(int(np.sum(~np.isfinite(grid))), 0)

    def test_meteo_grid_reproducible(self):
        g1 = mf.make_dry_run_factor_grid(self.lat, self.lon, seed=42)
        g2 = mf.make_dry_run_factor_grid(self.lat, self.lon, seed=42)
        np.testing.assert_array_equal(g1, g2,
                                      err_msg="Same seed should produce identical grid")

    def test_infra_grid_shape(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        self.assertEqual(grid.shape, (len(self.lat), len(self.lon)))

    def test_infra_grid_dtype(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        self.assertEqual(grid.dtype, np.float32)

    def test_infra_grid_range(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        self.assertGreaterEqual(float(grid.min()), ig._MULTIPLIER_MIN)
        self.assertLessEqual(float(grid.max()), ig._MULTIPLIER_MAX)

    def test_infra_grid_no_nan(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        self.assertEqual(int(np.sum(~np.isfinite(grid))), 0)


# ---------------------------------------------------------------------------
# Infrastructure rasterisation
# ---------------------------------------------------------------------------

class TestRasterization(unittest.TestCase):

    def test_single_poi_at_center(self):
        """A POI exactly at a grid-cell centre should land in that cell."""
        lat = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        lon = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        density = ig.rasterize_pois([(1.0, 1.0, 5.0)], lat, lon)
        self.assertEqual(density[1, 1], 5.0)
        self.assertEqual(density.sum(), 5.0, "Total weight should be conserved")

    def test_weight_accumulates_in_same_cell(self):
        """Multiple POIs rounding to the same cell must accumulate."""
        lat = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        lon = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        pois = [(1.1, 0.9, 2.0), (0.8, 1.2, 3.0)]  # both round to [1, 1]
        density = ig.rasterize_pois(pois, lat, lon)
        self.assertEqual(density[1, 1], 5.0)

    def test_out_of_bounds_discarded(self):
        """POIs outside the grid extent must be silently dropped."""
        lat = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        lon = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        density = ig.rasterize_pois([(99.0, 99.0, 999.0)], lat, lon)
        self.assertEqual(density.sum(), 0.0)

    def test_empty_poi_list(self):
        """Empty POI list must return an all-zero density grid."""
        lat = np.array([0.0, 1.0], dtype=np.float64)
        lon = np.array([0.0, 1.0], dtype=np.float64)
        density = ig.rasterize_pois([], lat, lon)
        self.assertEqual(density.sum(), 0.0)

    def test_correct_corner_assignment(self):
        """POIs near the grid boundary should round to the correct edge cell."""
        lat = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        lon = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        # Closest to top-right corner
        density = ig.rasterize_pois([(1.9, 1.8, 7.0)], lat, lon)
        self.assertEqual(density[2, 2], 7.0)


# ---------------------------------------------------------------------------
# Z-score normalisation (compute_infra_multiplier)
# ---------------------------------------------------------------------------

class TestZScoreNormalization(unittest.TestCase):

    def test_mean_is_one(self):
        """Mean multiplier across populated cells must be close to 1.0 (z-scores sum to 0).

        Minor deviation is expected when clamping clips a handful of extreme cells,
        so the tolerance is ±0.02 rather than exact.
        """
        rng = np.random.default_rng(0)
        raw = rng.exponential(scale=5.0, size=(12, 12))
        pop = np.ones((12, 12), dtype=np.float64) * 1000.0
        mult = ig.compute_infra_multiplier(raw, pop, alpha=0.15)
        self.assertAlmostEqual(float(mult.mean()), 1.0, delta=0.02,
                               msg="Mean multiplier must be within ±0.02 of 1.0")

    def test_zero_variance_returns_flat_ones(self):
        """Uniform density → no z-score variation → flat 1.0 grid."""
        raw = np.full((6, 6), 5.0, dtype=np.float64)
        pop = np.full((6, 6), 500.0, dtype=np.float64)
        mult = ig.compute_infra_multiplier(raw, pop, alpha=0.15)
        np.testing.assert_array_almost_equal(
            mult, 1.0, decimal=5,
            err_msg="Uniform density must produce flat 1.0 multiplier"
        )

    def test_alpha_controls_spread(self):
        """Larger alpha must produce larger standard deviation in multipliers."""
        rng = np.random.default_rng(1)
        raw = rng.exponential(scale=5.0, size=(10, 10))
        pop = np.ones((10, 10), dtype=np.float64) * 1000.0
        mult_low  = ig.compute_infra_multiplier(raw, pop, alpha=0.05)
        mult_high = ig.compute_infra_multiplier(raw, pop, alpha=0.30)
        self.assertLess(float(mult_low.std()), float(mult_high.std()),
                        msg="Higher alpha must produce larger spread")

    def test_zero_population_cells_get_one(self):
        """Cells with zero population must be assigned multiplier 1.0."""
        raw = np.array([[5.0, 2.0], [3.0, 1.0]], dtype=np.float64)
        pop = np.array([[100.0, 0.0], [200.0, 0.0]], dtype=np.float64)
        mult = ig.compute_infra_multiplier(raw, pop, alpha=0.15)
        self.assertAlmostEqual(mult[0, 1], 1.0, places=5,
                               msg="Zero-pop cell must get 1.0")
        self.assertAlmostEqual(mult[1, 1], 1.0, places=5)

    def test_range_clamped(self):
        """No multiplier should escape [MULTIPLIER_MIN, MULTIPLIER_MAX]."""
        rng = np.random.default_rng(2)
        raw = rng.exponential(scale=50.0, size=(20, 20))
        raw[0, 0] = 100_000.0  # extreme outlier
        pop = np.ones((20, 20), dtype=np.float64) * 1000.0
        mult = ig.compute_infra_multiplier(raw, pop, alpha=5.0)  # very large alpha
        self.assertGreaterEqual(float(mult.min()), ig._MULTIPLIER_MIN)
        self.assertLessEqual(float(mult.max()), ig._MULTIPLIER_MAX)

    def test_dense_cell_gets_higher_multiplier(self):
        """The most POI-dense cell must get a higher multiplier than the least dense."""
        raw = np.zeros((4, 4), dtype=np.float64)
        raw[0, 0] = 100.0  # very dense
        raw[3, 3] = 1.0    # sparse
        pop = np.ones((4, 4), dtype=np.float64) * 1000.0
        mult = ig.compute_infra_multiplier(raw, pop, alpha=0.15)
        self.assertGreater(mult[0, 0], mult[3, 3],
                           msg="Denser cell must have higher multiplier")


# ---------------------------------------------------------------------------
# NetCDF output format: both write functions satisfy read_grid_and_var
# ---------------------------------------------------------------------------

class TestNetCDFFormat(unittest.TestCase):

    def setUp(self):
        self.lat = np.linspace(30.0, 47.0, 8)
        self.lon = np.linspace(-120.0, -70.0, 12)
        self._tmpfiles = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _tmpfile(self) -> Path:
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        f.close()
        p = Path(f.name)
        self._tmpfiles.append(p)
        return p

    # --- meteo ---

    def test_meteo_nc_passes_read_grid_and_var(self):
        """write_meteo_nc output must be loadable by read_grid_and_var without error."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        path = self._tmpfile()
        mf.write_meteo_nc(path, self.lat, self.lon, grid, "2026-05-17")
        lat, lon, data, var_name = _read_grid_and_var(path)
        self.assertEqual(var_name, "meteo_beta_factor")

    def test_meteo_nc_data_dtype(self):
        """meteo_beta_factor variable must be float32."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        path = self._tmpfile()
        mf.write_meteo_nc(path, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(path)
        self.assertEqual(data.dtype, np.float32)

    def test_meteo_nc_shape(self):
        """Output data must match (n_lat, n_lon)."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        path = self._tmpfile()
        mf.write_meteo_nc(path, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(path)
        self.assertEqual(data.shape, (len(self.lat), len(self.lon)))

    def test_meteo_nc_no_nan(self):
        """Output must contain no NaN (pre-filled with 1.0)."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        path = self._tmpfile()
        mf.write_meteo_nc(path, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(path)
        self.assertEqual(int(np.sum(~np.isfinite(data))), 0)

    def test_meteo_nc_nan_prefill(self):
        """NaN values in the input grid must be replaced with 1.0, not 0.0."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon).astype(np.float32)
        grid[0, 0] = np.nan
        path = self._tmpfile()
        mf.write_meteo_nc(path, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(path)
        # read_grid_and_var fills remaining NaN with 0.0 — so if the writer
        # correctly pre-filled NaN with 1.0, the cell should read back as 1.0.
        self.assertAlmostEqual(float(data[0, 0]), 1.0, places=5,
                               msg="NaN cells must be written as 1.0 (neutral), not 0.0")

    def test_meteo_nc_coord_dtype(self):
        """lat and lon coordinate variables must be float64."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        path = self._tmpfile()
        mf.write_meteo_nc(path, self.lat, self.lon, grid, "2026-05-17")
        with Dataset(path, "r") as ds:
            self.assertEqual(ds.variables["lat"][:].dtype, np.float64)
            self.assertEqual(ds.variables["lon"][:].dtype, np.float64)

    # --- infra ---

    def test_infra_nc_passes_read_grid_and_var(self):
        """write_infra_nc output must be loadable by read_grid_and_var without error."""
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        path = self._tmpfile()
        ig.write_infra_nc(path, self.lat, self.lon, grid, alpha=0.15)
        lat, lon, data, var_name = _read_grid_and_var(path)
        self.assertEqual(var_name, "infra_multiplier")

    def test_infra_nc_data_dtype(self):
        """infra_multiplier variable must be float32."""
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        path = self._tmpfile()
        ig.write_infra_nc(path, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(path)
        self.assertEqual(data.dtype, np.float32)

    def test_infra_nc_shape(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        path = self._tmpfile()
        ig.write_infra_nc(path, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(path)
        self.assertEqual(data.shape, (len(self.lat), len(self.lon)))

    def test_infra_nc_no_nan(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        path = self._tmpfile()
        ig.write_infra_nc(path, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(path)
        self.assertEqual(int(np.sum(~np.isfinite(data))), 0)

    def test_infra_nc_nan_prefill(self):
        """NaN in infra grid must be written as 1.0, not 0.0."""
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        grid[2, 3] = np.nan
        path = self._tmpfile()
        ig.write_infra_nc(path, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(path)
        self.assertAlmostEqual(float(data[2, 3]), 1.0, places=5,
                               msg="NaN infra cells must be written as 1.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
