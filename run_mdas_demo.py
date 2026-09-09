#!/usr/bin/env python3
"""
run_mdas_demo.py — Sample runner: census ingestion -> heuristic inference ->
MAPS agent (grid) initialization.

Demonstrates the full mdas pipeline offline, with synthetic data (no network
access required), mirroring the dry-run convention used elsewhere in this
repo (see meteorological_factor.py's make_dry_run_factor_grid).

The underlying values (case counts, seroprevalence, testing coverage,
population) come from a small built-in library of example regions, since
this demo has no real data source to query — but which regions run, and on
what observation date, are up to the caller.

Usage:
    python run_mdas_demo.py
    python run_mdas_demo.py --regions US-CA,US-NY --date 2021-09-15
    python run_mdas_demo.py --list-regions
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict

from mdas.adapter import build_population_init_fractions_by_region, to_compartment_overrides
from mdas.inference import default_engine
import mdas.inference.rules  # noqa: F401 -- registers the example rules on import
from mdas.ingestion.census import ingest_census_snapshot
from mdas.ingestion.timeseries import ingest_timeseries
from mdas.schemas import MetricType

# Synthetic per-region figures this demo draws from — not real epidemiological
# data. Add an entry here to make a new region selectable via --regions.
_REGION_LIBRARY: Dict[str, Dict[str, float]] = {
    "US-CA": {"cases": 120_000, "seroprevalence": 18.0, "tested_pct": 60.0, "ci_lower": 15.0, "ci_upper": 21.0, "population": 39_500_000},
    "US-TX": {"cases": 95_000, "seroprevalence": 9.0, "tested_pct": 25.0, "ci_lower": 6.0, "ci_upper": 12.0, "population": 29_000_000},
    "US-NY": {"cases": 80_000, "seroprevalence": 20.0, "tested_pct": 55.0, "ci_lower": 17.0, "ci_upper": 23.0, "population": 19_500_000},
    "US-FL": {"cases": 60_000, "seroprevalence": 12.0, "tested_pct": 30.0, "ci_lower": 9.0, "ci_upper": 15.0, "population": 21_500_000},
    "US-WA": {"cases": 45_000, "seroprevalence": 14.0, "tested_pct": 40.0, "ci_lower": 11.0, "ci_upper": 17.0, "population": 7_700_000},
    "US-IL": {"cases": 70_000, "seroprevalence": 16.0, "tested_pct": 35.0, "ci_lower": 13.0, "ci_upper": 19.0, "population": 12_700_000},
}

_DEFAULT_REGIONS = ["US-CA", "US-TX"]
_DEFAULT_DATE = date(2021, 6, 1)


def _banner(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _round_floats(value, digits: int = 4):
    """Recursively round floats for display; does not touch the real data."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round_floats(v, digits) for k, v in value.items()}
    return value


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    parser.add_argument(
        "--regions",
        type=str,
        default=",".join(_DEFAULT_REGIONS),
        help=f"Comma-separated region codes to run (default: {','.join(_DEFAULT_REGIONS)}). "
        f"Available: {', '.join(sorted(_REGION_LIBRARY))}.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=_DEFAULT_DATE.isoformat(),
        help=f"Observation date (YYYY-MM-DD) applied to every synthetic record (default: {_DEFAULT_DATE.isoformat()}).",
    )
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="Print the available region codes and their synthetic figures, then exit.",
    )
    args = parser.parse_args(argv)

    if args.list_regions:
        return args

    unknown = [r for r in args.regions.split(",") if r.strip() and r.strip() not in _REGION_LIBRARY]
    if unknown:
        parser.error(
            f"unknown region(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(_REGION_LIBRARY))}."
        )
    try:
        date.fromisoformat(args.date)
    except ValueError:
        parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")

    return args


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.list_regions:
        print("Available regions:")
        for region, figures in sorted(_REGION_LIBRARY.items()):
            print(f"  {region}: {figures}")
        return

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    observation_date = args.date

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        _banner("1. Ingestion")

        # 1a. Continuous feed: daily reported cumulative cases per region.
        cases_lines = ["region,date,value"]
        cases_lines += [f"{r},{observation_date},{_REGION_LIBRARY[r]['cases']}" for r in regions]
        cases_csv = tmp / "cumulative_cases.csv"
        cases_csv.write_text("\n".join(cases_lines) + "\n")
        case_records = ingest_timeseries(cases_csv, metric_type=MetricType.CUMULATIVE_CASES)

        # 1b. Census/survey snapshot: seroprevalence + testing coverage survey.
        survey_lines = ["region,date,seroprevalence,tested_pct,ci_lower,ci_upper"]
        survey_lines += [
            f"{r},{observation_date},{_REGION_LIBRARY[r]['seroprevalence']},"
            f"{_REGION_LIBRARY[r]['tested_pct']},{_REGION_LIBRARY[r]['ci_lower']},"
            f"{_REGION_LIBRARY[r]['ci_upper']}"
            for r in regions
        ]
        survey_csv = tmp / "survey_snapshot.csv"
        survey_csv.write_text("\n".join(survey_lines) + "\n")
        sero_records = ingest_census_snapshot(
            survey_csv,
            metric_type=MetricType.SEROPREVALENCE_PCT,
            value_field="seroprevalence",
            ci_lower_field="ci_lower",
            ci_upper_field="ci_upper",
        )
        tested_records = ingest_census_snapshot(
            survey_csv, metric_type=MetricType.TESTED_POPULATION_PCT, value_field="tested_pct"
        )

        measured_and_census = case_records + sero_records + tested_records
        print(f"  regions          : {', '.join(regions)}")
        print(f"  observation date : {observation_date}")
        print(f"  {len(case_records):>2} MEASURED_DIRECT   (cumulative cases)")
        print(f"  {len(sero_records):>2} CENSUS_DERIVED    (seroprevalence)")
        print(f"  {len(tested_records):>2} CENSUS_DERIVED    (testing coverage)")

        _banner("2. Heuristic inference")

        # Fill in vaccine uptake + latent infection estimates that weren't
        # directly observed.
        combined = default_engine.run(measured_and_census, include_inputs=True)
        imputed_count = len(combined) - len(measured_and_census)
        print(f"  rules run        : {', '.join(default_engine.list_rules())}")
        print(f"  records added    : {imputed_count} (HEURISTIC_IMPUTED)")

        _banner("3. Simulation bridge (MAPS init fractions)")

        # Fold everything into per-region init fractions and render the
        # compartment-override keys MAPS's init script reads. Case counts are
        # raw (not pre-normalized), so population figures are required to
        # convert them to fractions.
        population_by_region = {r: _REGION_LIBRARY[r]["population"] for r in regions}
        fractions_by_region = build_population_init_fractions_by_region(
            combined, population_by_region=population_by_region
        )
        for region, fractions in sorted(fractions_by_region.items()):
            overrides = to_compartment_overrides(fractions)
            print(f"\n  {region}")
            print("    fractions:")
            print(_indent(json.dumps(_round_floats(fractions.model_dump()), indent=2), "      "))
            print("    compartment cfg:")
            print(_indent(json.dumps(_round_floats(overrides), indent=2), "      "))
        print()


if __name__ == "__main__":
    main()
