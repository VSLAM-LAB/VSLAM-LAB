"""
Module: VSLAM-LAB - Gym - facts.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 0.3
- Created: 2026-09-27
- Updated: 2026-09-27
- License: GPLv3 License

Fact functions for Gym/labels/facts_<n>.csv. Each takes (dataset, sequence, ctx) and returns
{fact name: value}, with None for a value it could not read. A function may return several
facts: result facts are named per experiment, `ate:median:<baseline>:<mode>`, one per
experiment block of the exp yaml given to label_sequences.py. `ctx` is a FactContext with those
experiments. `FACT_FUNCTIONS` lists the functions in column order. Facts are what the label
functions of Gym/labeling.py read.
"""

from __future__ import annotations

import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from path_constants import VSLAM_LAB_EVALUATION_FOLDER, VSLAMLAB_EVALUATION  # noqa: E402
from utilities import make_printers, read_csv, sequence_rgb_csv  # noqa: E402

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


@dataclass
class ExperimentRef:
    name: str          # experiment folder under VSLAM-LAB-Evaluation (the exp yaml block name)
    baseline: str      # Module of the block
    mode: str          # Parameters.mode of the block


@dataclass
class FactContext:
    experiments: list[ExperimentRef] = field(default_factory=list)


FactFunction = Callable[[str, str, FactContext], dict[str, Any]]


def num_frames(dataset: str, sequence: str, ctx: FactContext) -> dict[str, Any]:
    """num_frames: rows of the sequence's rgb.csv, header excluded."""
    rgb_csv = sequence_rgb_csv(dataset, sequence)
    if not rgb_csv.is_file():
        print_warning(f"{dataset}/{sequence}: {rgb_csv} not found")
        return {"num_frames": None}
    with rgb_csv.open() as f:
        return {"num_frames": sum(1 for line in f if line.strip()) - 1}


def ate_median(dataset: str, sequence: str, ctx: FactContext) -> dict[str, Any]:
    """ate:median:<baseline>:<mode> per experiment: median ATE RMSE (m) over its evaluated runs,
    read from <VSLAM-LAB-Evaluation>/<experiment>/<DATASET>/<sequence>/vslamlab_evaluation/ate.csv."""
    facts: dict[str, Any] = {}
    for exp in ctx.experiments:
        name = f"ate:median:{exp.baseline}:{exp.mode}"
        ate_csv = VSLAMLAB_EVALUATION / exp.name / dataset.upper() / sequence / VSLAM_LAB_EVALUATION_FOLDER / "ate.csv"
        ate = read_csv(ate_csv)
        if ate.empty or "rmse" not in ate.columns:
            print_warning(f"{dataset}/{sequence}: no evaluated runs in {ate_csv}")
            facts[name] = None
            continue
        rmses = [float(v) for v in ate["rmse"].dropna()]
        facts[name] = round(statistics.median(rmses), 6) if rmses else None
    return facts


FACT_FUNCTIONS: list[FactFunction] = [num_frames, ate_median]
