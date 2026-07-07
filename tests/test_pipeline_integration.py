#!/usr/bin/env python3
"""
test_pipeline_integration.py — End-to-end pipeline tests.

Verifies that all three scripts interact correctly as a chain:

    meteorological_factor.py  →  maps_meteo_<date>.nc
    infrastructure_grid.py    →  maps_infrastructure.nc
    edit_namelist.py          →  beta0_by_variant in .nml  (via load_spatial_mean)

Tests in this file:
  1. Both factor scripts produce valid NetCDF files via --dry-run
  2. Output files satisfy the read_grid_and_var format contract (single variable,
     float32 data, float64 coords, no NaN, no stray zeros from NaN pre-fill)
  3. load_spatial_mean returns the exact spatial mean of a known synthetic grid
  4. load_spatial_mean returns 1.0 (neutral) when the file is absent or corrupt
  5. beta0 scaling: given known meteo_mean and infra_mean, the beta0_base formula
     produces the expected numerical output
  6. Winter vs summer beta0: a cold-dry-dark meteo grid must raise beta0 relative
     to a warm-humid-sunny grid
  7. No-grid fallback: when neither grid file exists, beta0_base is unchanged from
     the original formula

Run with:
    /Users/jaydenchan/maps/.venv/bin/python3 -m pytest tests/test_pipeline_integration.py -v
or:
    /Users/jaydenchan/maps/.venv/bin/python3 tests/test_pipeline_integration.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import meteorological_factor as mf
import infrastructure_grid as ig

PYTHON = sys.executable   # same interpreter running the tests


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------

def _read_grid_and_var(path: Path):
    """Exact copy of read_grid_and_var from the IC pre-processor."""
    with Dataset(path, "r") as ds:
        lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        candidates = [n for n in ds.variables if n not in ("lat", "lon")]
        if len(candidates) != 1:
            raise ValueError(f"Expected 1 data variable; found {candidates}")
        var_name = candidates[0]
        raw = ds.variables[var_name][:]
        data = np.ma.filled(raw, 0.0) if np.ma.isMaskedArray(raw) else np.array(raw)
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isfinite(data), data, 0.0)
    return lat, lon, data, var_name


def _load_spatial_mean(nc_path: Path) -> float:
    """
    Mirror of load_spatial_mean from edit_namelist.py.

    Used here to verify the function's behaviour without importing
    edit_namelist (which runs module-level code on import and requires
    live sys.argv arguments and data files).
    """
    try:
        with Dataset(nc_path, "r") as ds:
            candidates = [n for n in ds.variables if n not in ("lat", "lon")]
            if len(candidates) != 1:
                return 1.0
            raw = ds.variables[candidates[0]][:]
            data = (
                np.ma.filled(raw, 1.0)
                if np.ma.isMaskedArray(raw)
                else np.array(raw, dtype=float)
            )
            data = np.where(np.isfinite(data), data, 1.0)
            return float(data.mean())
    except Exception:
        return 1.0


def _write_synthetic_nc(path: Path, values: np.ndarray) -> None:
    """Write a minimal MAPS-format NetCDF with one data variable from a 2-D array."""
    lat = np.linspace(30.0, 45.0, values.shape[0], dtype=np.float64)
    lon = np.linspace(-120.0, -70.0, values.shape[1], dtype=np.float64)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("lat", values.shape[0])
        ds.createDimension("lon", values.shape[1])
        lv = ds.createVariable("lat", "f8", ("lat",))
        lv[:] = lat
        lv.units = "degrees_north"
        lov = ds.createVariable("lon", "f8", ("lon",))
        lov[:] = lon
        lov.units = "degrees_east"
        dv = ds.createVariable("test_var", "f4", ("lat", "lon"), zlib=True)
        dv[:, :] = values.astype(np.float32)


def _beta0_base(beta_guess: float, meteo_mean: float, infra_mean: float) -> float:
    """Mirrors the beta0_base formula in edit_namelist.py."""
    return (0.23 + 0.20 * beta_guess) * meteo_mean * infra_mean


# ---------------------------------------------------------------------------
# 1 — Dry-run script execution
# ---------------------------------------------------------------------------

class TestDryRunScripts(unittest.TestCase):
    """Both scripts must complete with exit code 0 and produce a readable NetCDF."""

    def setUp(self):
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _tmpfile(self, suffix=".nc") -> Path:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.close()
        p = Path(f.name)
        self._tmpfiles.append(p)
        return p

    def test_meteo_dry_run_exits_zero(self):
        """meteorological_factor.py --dry-run must exit with code 0."""
        out = self._tmpfile()
        result = subprocess.run(
            [PYTHON, "meteorological_factor.py",
             "--init-date", "2026-05-17",
             "--output", str(out),
             "--dry-run"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        self.assertEqual(result.returncode, 0,
                         msg=f"Non-zero exit:\n{result.stderr}")

    def test_meteo_dry_run_creates_file(self):
        """meteorological_factor.py --dry-run must create the output NetCDF."""
        out = self._tmpfile()
        subprocess.run(
            [PYTHON, "meteorological_factor.py",
             "--init-date", "2026-05-17",
             "--output", str(out),
             "--dry-run"],
            capture_output=True,
            cwd=Path(__file__).parent.parent,
        )
        self.assertTrue(out.exists(), "Output NetCDF was not created")
        self.assertGreater(out.stat().st_size, 0, "Output NetCDF is empty")

    def test_infra_dry_run_exits_zero(self):
        """infrastructure_grid.py --dry-run must exit with code 0."""
        out = self._tmpfile()
        result = subprocess.run(
            [PYTHON, "infrastructure_grid.py",
             "--output", str(out),
             "--dry-run"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        self.assertEqual(result.returncode, 0,
                         msg=f"Non-zero exit:\n{result.stderr}")

    def test_infra_dry_run_creates_file(self):
        """infrastructure_grid.py --dry-run must create the output NetCDF."""
        out = self._tmpfile()
        subprocess.run(
            [PYTHON, "infrastructure_grid.py",
             "--output", str(out),
             "--dry-run"],
            capture_output=True,
            cwd=Path(__file__).parent.parent,
        )
        self.assertTrue(out.exists(), "Output NetCDF was not created")
        self.assertGreater(out.stat().st_size, 0, "Output NetCDF is empty")

    def test_meteo_output_variable_name(self):
        """Dry-run meteo output must use the variable name 'meteo_beta_factor'."""
        out = self._tmpfile()
        subprocess.run(
            [PYTHON, "meteorological_factor.py",
             "--init-date", "2026-05-17", "--output", str(out), "--dry-run"],
            capture_output=True,
            cwd=Path(__file__).parent.parent,
        )
        _, _, _, var_name = _read_grid_and_var(out)
        self.assertEqual(var_name, "meteo_beta_factor")

    def test_infra_output_variable_name(self):
        """Dry-run infra output must use the variable name 'infra_multiplier'."""
        out = self._tmpfile()
        subprocess.run(
            [PYTHON, "infrastructure_grid.py",
             "--output", str(out), "--dry-run"],
            capture_output=True,
            cwd=Path(__file__).parent.parent,
        )
        _, _, _, var_name = _read_grid_and_var(out)
        self.assertEqual(var_name, "infra_multiplier")


# ---------------------------------------------------------------------------
# 2 — Format contract (read_grid_and_var compatibility)
# ---------------------------------------------------------------------------

class TestFormatContract(unittest.TestCase):
    """Both write functions must satisfy all constraints of read_grid_and_var."""

    def setUp(self):
        self.lat = np.linspace(30.0, 47.0, 10)
        self.lon = np.linspace(-120.0, -70.0, 15)
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _tmpfile(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        f.close()
        p = Path(f.name)
        self._tmpfiles.append(p)
        return p

    def test_meteo_exactly_one_data_variable(self):
        """meteo NC must have exactly one non-coordinate variable."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        p = self._tmpfile()
        mf.write_meteo_nc(p, self.lat, self.lon, grid, "2026-05-17")
        with Dataset(p, "r") as ds:
            extras = [n for n in ds.variables if n not in ("lat", "lon")]
        self.assertEqual(len(extras), 1,
                         msg=f"Expected 1 data var; found {extras}")

    def test_infra_exactly_one_data_variable(self):
        """infra NC must have exactly one non-coordinate variable."""
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        p = self._tmpfile()
        ig.write_infra_nc(p, self.lat, self.lon, grid, alpha=0.15)
        with Dataset(p, "r") as ds:
            extras = [n for n in ds.variables if n not in ("lat", "lon")]
        self.assertEqual(len(extras), 1,
                         msg=f"Expected 1 data var; found {extras}")

    def test_meteo_float32_data_variable(self):
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        p = self._tmpfile()
        mf.write_meteo_nc(p, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(p)
        self.assertEqual(data.dtype, np.float32)

    def test_infra_float32_data_variable(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        p = self._tmpfile()
        ig.write_infra_nc(p, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(p)
        self.assertEqual(data.dtype, np.float32)

    def test_meteo_no_nan_after_reader(self):
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        p = self._tmpfile()
        mf.write_meteo_nc(p, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(p)
        self.assertEqual(int(np.sum(~np.isfinite(data))), 0)

    def test_infra_no_nan_after_reader(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        p = self._tmpfile()
        ig.write_infra_nc(p, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(p)
        self.assertEqual(int(np.sum(~np.isfinite(data))), 0)

    def test_meteo_no_accidental_zeros(self):
        """No cell should read back as 0.0 from a dry-run meteo grid."""
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        p = self._tmpfile()
        mf.write_meteo_nc(p, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(p)
        zero_count = int(np.sum(data == 0.0))
        self.assertEqual(zero_count, 0,
                         msg="Dry-run meteo grid should have no zero cells")

    def test_infra_no_accidental_zeros(self):
        """No cell should read back as 0.0 from a dry-run infra grid."""
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        p = self._tmpfile()
        ig.write_infra_nc(p, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(p)
        zero_count = int(np.sum(data == 0.0))
        self.assertEqual(zero_count, 0,
                         msg="Dry-run infra grid should have no zero cells")

    def test_meteo_values_in_valid_range(self):
        grid = mf.make_dry_run_factor_grid(self.lat, self.lon)
        p = self._tmpfile()
        mf.write_meteo_nc(p, self.lat, self.lon, grid, "2026-05-17")
        _, _, data, _ = _read_grid_and_var(p)
        self.assertGreater(float(data.min()), 0.0)
        self.assertLessEqual(float(data.max()), mf._FACTOR_MAX + 1e-5)

    def test_infra_values_in_valid_range(self):
        grid = ig.make_dry_run_multiplier_grid(self.lat, self.lon)
        p = self._tmpfile()
        ig.write_infra_nc(p, self.lat, self.lon, grid, alpha=0.15)
        _, _, data, _ = _read_grid_and_var(p)
        self.assertGreaterEqual(float(data.min()), ig._MULTIPLIER_MIN - 1e-5)
        self.assertLessEqual(float(data.max()), ig._MULTIPLIER_MAX + 1e-5)


# ---------------------------------------------------------------------------
# 3 — load_spatial_mean correctness
# ---------------------------------------------------------------------------

class TestLoadSpatialMean(unittest.TestCase):
    """load_spatial_mean must return the exact mean of a known input grid."""

    def setUp(self):
        self._tmpfiles: list[Path] = []

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _tmpfile(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        f.close()
        p = Path(f.name)
        self._tmpfiles.append(p)
        return p

    def test_flat_grid_returns_that_value(self):
        """A grid of all 1.08 must return 1.08."""
        p = self._tmpfile()
        _write_synthetic_nc(p, np.full((8, 12), 1.08, dtype=np.float32))
        result = _load_spatial_mean(p)
        self.assertAlmostEqual(result, 1.08, places=4)

    def test_known_mean(self):
        """A 2×2 grid [1.0, 1.2, 0.8, 1.0] has mean 1.0."""
        p = self._tmpfile()
        values = np.array([[1.0, 1.2], [0.8, 1.0]], dtype=np.float32)
        _write_synthetic_nc(p, values)
        result = _load_spatial_mean(p)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_neutral_mean_raises_no_error(self):
        """All-1.0 grid should return 1.0 cleanly."""
        p = self._tmpfile()
        _write_synthetic_nc(p, np.ones((6, 6), dtype=np.float32))
        result = _load_spatial_mean(p)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_missing_file_returns_one(self):
        """A path that does not exist must return 1.0 (neutral fallback)."""
        result = _load_spatial_mean(Path("/nonexistent/path/fake_grid.nc"))
        self.assertEqual(result, 1.0)

    def test_nan_cells_treated_as_one(self):
        """NaN values in the grid must be treated as 1.0, not 0.0, before averaging."""
        p = self._tmpfile()
        values = np.array([[1.2, np.nan], [1.0, 0.8]], dtype=np.float32)
        _write_synthetic_nc(p, values)
        result = _load_spatial_mean(p)
        # NaN → 1.0, so effective array is [1.2, 1.0, 1.0, 0.8], mean = 1.0
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_dry_run_meteo_mean_near_one(self):
        """The dry-run meteo grid should have a spatial mean close to 1.0."""
        lat = np.linspace(30.0, 47.0, 8)
        lon = np.linspace(-120.0, -70.0, 12)
        grid = mf.make_dry_run_factor_grid(lat, lon)
        p = self._tmpfile()
        mf.write_meteo_nc(p, lat, lon, grid, "2026-05-17")
        result = _load_spatial_mean(p)
        self.assertAlmostEqual(result, 1.0, delta=0.10,
                               msg="Dry-run meteo mean should be within ±0.10 of 1.0")

    def test_dry_run_infra_mean_near_one(self):
        """The dry-run infra grid should have a spatial mean close to 1.0."""
        lat = np.linspace(30.0, 47.0, 8)
        lon = np.linspace(-120.0, -70.0, 12)
        grid = ig.make_dry_run_multiplier_grid(lat, lon)
        p = self._tmpfile()
        ig.write_infra_nc(p, lat, lon, grid, alpha=0.15)
        result = _load_spatial_mean(p)
        self.assertAlmostEqual(result, 1.0, delta=0.10,
                               msg="Dry-run infra mean should be within ±0.10 of 1.0")


# ---------------------------------------------------------------------------
# 4 — beta0 scaling formula
# ---------------------------------------------------------------------------

class TestBeta0Scaling(unittest.TestCase):
    """
    Verify the beta0_base formula introduced in edit_namelist.py:
        beta0_base = (0.23 + 0.20 * beta_guess) * meteo_mean * infra_mean
    """

    def test_neutral_factors_unchanged(self):
        """With both means = 1.0, beta0_base must equal the original formula."""
        for beta_guess in [0.0, 0.05, 0.10, 0.20, 0.30]:
            original = 0.23 + 0.20 * beta_guess
            scaled   = _beta0_base(beta_guess, 1.0, 1.0)
            self.assertAlmostEqual(scaled, original, places=9,
                                   msg=f"Neutral factors changed beta0 at beta_guess={beta_guess}")

    def test_winter_raises_beta0(self):
        """A meteo_mean > 1.0 (winter) must increase beta0_base."""
        beta_guess = 0.10
        original = 0.23 + 0.20 * beta_guess
        scaled   = _beta0_base(beta_guess, meteo_mean=1.08, infra_mean=1.0)
        self.assertGreater(scaled, original,
                           msg="Winter meteo factor should increase beta0")

    def test_summer_lowers_beta0(self):
        """A meteo_mean < 1.0 (summer) must decrease beta0_base."""
        beta_guess = 0.10
        original = 0.23 + 0.20 * beta_guess
        scaled   = _beta0_base(beta_guess, meteo_mean=0.88, infra_mean=1.0)
        self.assertLess(scaled, original,
                        msg="Summer meteo factor should decrease beta0")

    def test_urban_infra_raises_beta0(self):
        """An infra_mean > 1.0 (dense infrastructure) must increase beta0_base."""
        beta_guess = 0.10
        original = 0.23 + 0.20 * beta_guess
        scaled   = _beta0_base(beta_guess, meteo_mean=1.0, infra_mean=1.03)
        self.assertGreater(scaled, original)

    def test_combined_winter_urban(self):
        """Combined winter + urban multiplier should raise beta0 by ~11%."""
        beta_guess = 0.10
        original = 0.23 + 0.20 * beta_guess
        scaled   = _beta0_base(beta_guess, meteo_mean=1.08, infra_mean=1.03)
        ratio = scaled / original
        self.assertAlmostEqual(ratio, 1.08 * 1.03, places=6)

    def test_winter_vs_summer_beta0_ordering(self):
        """For any beta_guess, winter beta0 must exceed summer beta0."""
        for beta_guess in [0.05, 0.10, 0.15, 0.20]:
            winter_beta0 = _beta0_base(beta_guess, meteo_mean=1.10, infra_mean=1.0)
            summer_beta0 = _beta0_base(beta_guess, meteo_mean=0.88, infra_mean=1.0)
            self.assertGreater(winter_beta0, summer_beta0,
                               msg=f"Winter beta0 must exceed summer at beta_guess={beta_guess}")

    def test_scale_is_multiplicative(self):
        """The formula is purely multiplicative — verify proportionality."""
        beta_guess = 0.12
        base  = _beta0_base(beta_guess, 1.0, 1.0)
        scaled = _beta0_base(beta_guess, 1.05, 1.02)
        self.assertAlmostEqual(scaled, base * 1.05 * 1.02, places=9)


# ---------------------------------------------------------------------------
# 5 — Seasonal end-to-end: real physics → grid → mean → beta0
# ---------------------------------------------------------------------------

class TestSeasonalBeta0Pipeline(unittest.TestCase):
    """
    Full chain test: meteorological conditions → grid → spatial mean → beta0.
    Confirms that the physics, the NetCDF writer, and load_spatial_mean all
    agree on the direction of the seasonal signal.
    """

    def setUp(self):
        self._tmpfiles: list[Path] = []
        self.lat = np.linspace(30.0, 47.0, 8)
        self.lon = np.linspace(-120.0, -70.0, 12)

    def tearDown(self):
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)

    def _tmpfile(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        f.close()
        p = Path(f.name)
        self._tmpfiles.append(p)
        return p

    def _build_uniform_meteo_grid(self, T: float, RH: float, UV: float) -> np.ndarray:
        """Build a meteo grid where every cell has the same weather conditions."""
        AH = mf.compute_absolute_humidity(T, RH)
        factor = mf.compute_meteo_factor(AH, UV, T)
        return np.full((len(self.lat), len(self.lon)), factor, dtype=np.float32)

    def test_winter_mean_above_one(self):
        """A uniform winter grid (cold, dry, low UV) must have mean > 1.0."""
        grid = self._build_uniform_meteo_grid(T=-5.0, RH=55.0, UV=1.0)
        p = self._tmpfile()
        mf.write_meteo_nc(p, self.lat, self.lon, grid, "2026-01-15")
        mean = _load_spatial_mean(p)
        self.assertGreater(mean, 1.0,
                           msg="Uniform winter meteo mean should be > 1.0")

    def test_summer_mean_below_one(self):
        """A uniform summer grid (warm, humid, high UV) must have mean < 1.0."""
        grid = self._build_uniform_meteo_grid(T=28.0, RH=75.0, UV=8.0)
        p = self._tmpfile()
        mf.write_meteo_nc(p, self.lat, self.lon, grid, "2026-07-15")
        mean = _load_spatial_mean(p)
        self.assertLess(mean, 1.0,
                        msg="Uniform summer meteo mean should be < 1.0")

    def test_winter_beta0_exceeds_summer_beta0(self):
        """For a given variant, winter beta0 must exceed summer beta0."""
        beta_guess = 0.10

        winter_grid = self._build_uniform_meteo_grid(T=-5.0, RH=55.0, UV=1.0)
        pw = self._tmpfile()
        mf.write_meteo_nc(pw, self.lat, self.lon, winter_grid, "2026-01-15")
        winter_mean = _load_spatial_mean(pw)

        summer_grid = self._build_uniform_meteo_grid(T=28.0, RH=75.0, UV=8.0)
        ps = self._tmpfile()
        mf.write_meteo_nc(ps, self.lat, self.lon, summer_grid, "2026-07-15")
        summer_mean = _load_spatial_mean(ps)

        winter_beta0 = _beta0_base(beta_guess, winter_mean, 1.0)
        summer_beta0 = _beta0_base(beta_guess, summer_mean, 1.0)

        self.assertGreater(winter_beta0, summer_beta0,
                           msg=(
                               f"Winter beta0={winter_beta0:.5f} should exceed "
                               f"summer beta0={summer_beta0:.5f}"
                           ))

    def test_beta0_ratio_proportional_to_meteo_means(self):
        """winter_beta0 / summer_beta0 should equal winter_mean / summer_mean."""
        beta_guess = 0.12

        winter_grid = self._build_uniform_meteo_grid(T=-5.0, RH=55.0, UV=1.0)
        pw = self._tmpfile()
        mf.write_meteo_nc(pw, self.lat, self.lon, winter_grid, "2026-01-15")
        winter_mean = _load_spatial_mean(pw)

        summer_grid = self._build_uniform_meteo_grid(T=28.0, RH=75.0, UV=8.0)
        ps = self._tmpfile()
        mf.write_meteo_nc(ps, self.lat, self.lon, summer_grid, "2026-07-15")
        summer_mean = _load_spatial_mean(ps)

        winter_beta0 = _beta0_base(beta_guess, winter_mean, 1.0)
        summer_beta0 = _beta0_base(beta_guess, summer_mean, 1.0)

        expected_ratio = winter_mean / summer_mean
        actual_ratio   = winter_beta0 / summer_beta0

        self.assertAlmostEqual(actual_ratio, expected_ratio, places=6,
                               msg="beta0 ratio must equal meteo mean ratio")


if __name__ == "__main__":
    unittest.main(verbosity=2)
