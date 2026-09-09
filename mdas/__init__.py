"""
mdas — MAPS Data and Analysis System, Component 2: multi-source epidemiological ingestion.

Sub-packages / modules
-----------------------
schemas      Pydantic records shared by every ingestion path (mdas/schemas.py).
ingestion/   Distinct pipelines for continuous feeds (APIs/registries) and
             census/survey snapshots. Both normalize onto EpiMetricRecord.
inference/   Configurable assumption/heuristic rules that turn sparse
             measured + census data into HEURISTIC_IMPUTED records.
adapter      Converts the combined record set into grid-compatible population
             init fractions consumable by build_maps_initial_conditions_with_history_full.py.
"""
from __future__ import annotations
