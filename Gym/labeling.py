"""
Module: VSLAM-LAB - Gym - labeling.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 0.2
- Created: 2026-09-27
- Updated: 2026-09-27
- License: GPLv3 License

One function per label of Gym/label_definitions.yaml. Each takes the facts of a sequence and
the label's rule (its entry in the yaml) and returns {label name: value}, with None when a fact
it needs is missing. A function may return several labels: `ate` yields one
`ate:<baseline>:<mode>` per `ate:median:<baseline>:<mode>` fact. `LABEL_FUNCTIONS` maps the yaml
entry name to its function; a label without a function here, or a function without an entry
there, is an error at load time.
"""

from __future__ import annotations

from typing import Any, Callable

LabelFunction = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

ATE_FACT_PREFIX = "ate:median:"


def _band(value: float | None, low_max: float, medium_max: float, values: list[str]) -> str | None:
    """values[0] when value <= low_max, values[1] when <= medium_max, else values[2]."""
    if value is None:
        return None
    if value <= low_max:
        return values[0]
    if value <= medium_max:
        return values[1]
    return values[2]


def length(facts: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """length: short | medium | long by num_frames against short_max / medium_max."""
    return {"length": _band(facts.get("num_frames"), rule["short_max"], rule["medium_max"], rule["values"])}


def ate(facts: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """ate:<baseline>:<mode>: low | medium | high by ate:median:<baseline>:<mode> against low_max / medium_max."""
    return {"ate:" + name[len(ATE_FACT_PREFIX):]: _band(value, rule["low_max"], rule["medium_max"], rule["values"])
            for name, value in facts.items() if name.startswith(ATE_FACT_PREFIX)}


LABEL_FUNCTIONS: dict[str, LabelFunction] = {
    "length": length,
    "ate": ate,
}


def check_definitions(rules: dict[str, Any]) -> None:
    """Every yaml label has a function and every function has a yaml label."""
    missing = sorted(set(rules) - set(LABEL_FUNCTIONS))
    extra = sorted(set(LABEL_FUNCTIONS) - set(rules))
    if missing or extra:
        raise KeyError(f"label_definitions.yaml and labeling.py disagree: "
                       f"no function for {missing}, no definition for {extra}")
