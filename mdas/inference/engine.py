"""
mdas/inference/engine.py — Configurable assumption/imputation engine.

When direct measurements are sparse, the research team encodes an inferential
assumption as a "heuristic rule": a plain Python function that looks at the
MEASURED_DIRECT/CENSUS_DERIVED records available for one region and returns
zero or more new EpiMetricRecords tagged HEURISTIC_IMPUTED. The engine's job
is only to run the right rules over the right regions and to enforce the
HEURISTIC_IMPUTED tag on whatever they produce — the actual epidemiological
assumption lives entirely in the rule function, in mdas/inference/rules.py or
wherever the research team defines it.

How to register a new heuristic
--------------------------------
1. Write a function with this signature:

       def my_rule(records: List[EpiMetricRecord], *, region: str, pathogen: str) -> List[EpiMetricRecord]:
           ...

   `records` is every record (of any provenance) available for that one
   region; return the new records your rule infers (data_provenance is set
   for you, so you don't need to set it yourself, though setting it to
   HEURISTIC_IMPUTED explicitly is also fine).

2. Register it, either via decorator:

       from mdas.inference import register_rule

       @register_rule("vaccination_from_testing")
       def my_rule(records, *, region, pathogen):
           ...

   or directly against an engine instance:

       engine = InferenceEngine()
       engine.register("vaccination_from_testing", my_rule)

3. Run it (registering with `register_rule` adds it to the process-wide
   `default_engine`, so `default_engine.run(records)` will already include
   it; use a fresh `InferenceEngine()` instead if you want isolation, e.g.
   in tests):

       imputed = default_engine.run(records)  # region + pathogen handled internally

Rules are intentionally simple callables rather than a class hierarchy —
there is no ceremony beyond "takes records for one region, returns new
records" needed to add a new assumption.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from mdas.schemas import DataProvenance, EpiMetricRecord

HeuristicRule = Callable[..., List[EpiMetricRecord]]


class InferenceEngine:
    """A named registry of heuristic rules, run per-region over a record set."""

    def __init__(self) -> None:
        self._rules: Dict[str, HeuristicRule] = {}

    def register(self, name: str, rule: HeuristicRule) -> None:
        if name in self._rules:
            raise ValueError(f"a rule named {name!r} is already registered")
        self._rules[name] = rule

    def list_rules(self) -> List[str]:
        return sorted(self._rules)

    def run(
        self,
        records: List[EpiMetricRecord],
        *,
        rule_names: Optional[List[str]] = None,
        include_inputs: bool = False,
    ) -> List[EpiMetricRecord]:
        """
        Run the selected rules (default: all registered rules) once per
        distinct (region, pathogen) pair present in `records`.

        Returns the newly inferred HEURISTIC_IMPUTED records, plus the
        original input records too if include_inputs=True (handy for
        handing a single combined list straight to mdas/adapter.py).
        """
        names = rule_names if rule_names is not None else self.list_rules()
        groups: Dict[tuple, List[EpiMetricRecord]] = {}
        for rec in records:
            groups.setdefault((rec.region, rec.pathogen), []).append(rec)

        inferred: List[EpiMetricRecord] = []
        for (region, pathogen), group_records in groups.items():
            for name in names:
                rule = self._rules[name]
                new_records = rule(group_records, region=region, pathogen=pathogen)
                for rec in new_records:
                    if rec.data_provenance != DataProvenance.HEURISTIC_IMPUTED:
                        rec = rec.model_copy(update={"data_provenance": DataProvenance.HEURISTIC_IMPUTED})
                    if rec.source is None:
                        rec = rec.model_copy(update={"source": f"heuristic:{name}"})
                    inferred.append(rec)

        return (records + inferred) if include_inputs else inferred


# Process-wide default engine that register_rule() populates, so that simply
# importing mdas.inference.rules is enough to make its rules available via
# `default_engine.run(...)` without every caller wiring up their own registry.
default_engine = InferenceEngine()


def register_rule(name: str) -> Callable[[HeuristicRule], HeuristicRule]:
    """Decorator: register a rule function against the module-wide default_engine."""

    def decorator(fn: HeuristicRule) -> HeuristicRule:
        default_engine.register(name, fn)
        return fn

    return decorator
