"""
mdas/ingestion/base.py — Shared raw-record loading for both ingestion pipelines.

Mirrors the optional-import + disk-cache pattern used elsewhere in this repo
(see healthcare_capacity_grid.py's _load_from_api/_load_from_file): heavy
deps are imported lazily so importing mdas never requires pandas/requests
unless a code path that actually needs them is called, and API responses are
cached to disk keyed by URL hash so repeat runs don't re-hit the network.

Both mdas/ingestion/timeseries.py and mdas/ingestion/census.py route through
load_tabular_records() here — it is the one place that knows how to turn a
CSV file, a Parquet file, a local JSON file, or a raw JSON API response into
a flat list of dict rows. Everything after that (mapping columns onto
EpiMetricRecord fields, choosing a data_provenance) is pipeline-specific.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


_DEFAULT_CACHE_DIR = Path(".mdas_cache")

# Common keys under which a JSON API wraps its row list. Tried in order;
# if none match and the payload is already a list, it's used as-is.
_JSON_LIST_KEYS = ("data", "results", "rows", "records", "value")


def _hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[mdas.ingestion] {msg}", file=sys.stderr)


def _normalize_json_payload(payload: Any) -> List[Dict[str, Any]]:
    """Reduce a parsed JSON API response to a flat list of row dicts."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _JSON_LIST_KEYS:
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        # No list found: treat the whole object as a single row.
        return [payload]
    raise ValueError(f"Unsupported JSON payload type: {type(payload)!r}")


def _load_from_api(
    url: str,
    *,
    cache_dir: Optional[Path],
    force_refresh: bool,
    timeout_s: float,
    verbose: bool,
) -> List[Dict[str, Any]]:
    if not _REQUESTS_AVAILABLE:
        raise ImportError(
            "Fetching from an API URL requires the 'requests' package. "
            "Install it with: pip install requests"
        )
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_hash_key(url)}.json"

    if cache_path.exists() and not force_refresh:
        _log(verbose, f"cache hit for {url} -> {cache_path}")
        with cache_path.open("r") as f:
            return _normalize_json_payload(json.load(f))

    try:
        _log(verbose, f"fetching {url}")
        resp = _requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        with cache_path.open("w") as f:
            json.dump(payload, f)
        return _normalize_json_payload(payload)
    except Exception as exc:
        if cache_path.exists():
            _log(verbose, f"fetch failed ({exc}); falling back to stale cache")
            with cache_path.open("r") as f:
                return _normalize_json_payload(json.load(f))
        raise


def _load_from_file(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r") as f:
            return _normalize_json_payload(json.load(f))
    if suffix in (".csv", ".parquet"):
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                f"Reading {suffix} files requires the 'pandas' package. "
                "Install it with: pip install pandas"
            )
        df = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path} (expected .csv, .parquet, or .json)")


def load_tabular_records(
    source: Union[str, Path],
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    timeout_s: float = 30.0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load rows from a CSV/Parquet/JSON file or a raw JSON API endpoint.

    Parameters
    ----------
    source : str or Path
        A local file path (.csv/.parquet/.json) or an http(s):// URL.
    cache_dir : Path, optional
        Where to cache API responses (default: ./.mdas_cache). Ignored for
        local files.
    force_refresh : bool
        Bypass the cache and re-fetch even if a cached response exists.
    timeout_s : float
        HTTP timeout for API requests.
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    List[dict] — one dict per row, column/field names as given by the source.
    """
    source_str = str(source)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        return _load_from_api(
            source_str, cache_dir=cache_dir, force_refresh=force_refresh,
            timeout_s=timeout_s, verbose=verbose,
        )
    return _load_from_file(Path(source))
