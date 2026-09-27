"""
Module: VSLAM-LAB - Gym - label_sequences.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 0.6
- Created: 2026-09-27
- Updated: 2026-09-27
- License: GPLv3 License

Two steps, two files:
  collect_facts     computes every fact of Gym/facts.py for each targeted sequence and writes
                    Gym/labels/facts_<n>.csv (one row per sequence, one column per fact)
  label_sequences   applies every label of Gym/label_definitions.yaml, each computed by its
                    function in Gym/labeling.py from those facts, and writes
                    Gym/labels/labels_<n>.csv (one row per sequence, one column per label)

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets):
  pixi run label-sequences <dataset> [<sequence> ...]
  pixi run label-sequences --datasets d1 d2 ... | --sequences d s1 s2 ... | --exp exp.yaml | --configs config.yaml

The experiment blocks of the --exp yaml are also the source of result facts: each block gives
one `ate:median:<Module>:<mode>` column read from its evaluation folder. Without --exp only
sequence facts are collected.

Nothing is overwritten: each call writes a new pair facts_<n>.csv / labels_<n>.csv, with <n>
one past the highest index already present in Gym/labels/.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Gym.facts import FACT_FUNCTIONS, ExperimentRef, FactContext  # noqa: E402
from Gym.labeling import LABEL_FUNCTIONS, check_definitions  # noqa: E402
from utilities import (  # noqa: E402
    add_sequence_target_args, load_yaml_file, make_printers, resolve_sequence_targets_or_exit,
    write_csv_rows,
)
from vslamlab_utilities import load_experiments  # noqa: E402

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

GYM_DIR = REPO_ROOT / "Gym"
LABELS_DIR = GYM_DIR / "labels"
DEFINITIONS_YAML = GYM_DIR / "label_definitions.yaml"
DEFAULT_MODE = "mono"


@dataclass
class SequenceFacts:
    dataset: str
    sequence: str
    facts: dict[str, Any]       # fact name -> value, None when it could not be read

    def row(self, names: list[str]) -> list[str]:
        return [self.dataset, self.sequence] + _cells(self.facts, names)


@dataclass
class SequenceLabels:
    dataset: str
    sequence: str
    labels: dict[str, Any]      # label name -> value, None when a needed fact is missing

    def row(self, names: list[str]) -> list[str]:
        return [self.dataset, self.sequence] + _cells(self.labels, names)


def _cells(values: dict[str, Any], names: list[str]) -> list[str]:
    return ["" if values.get(n) is None else str(values[n]) for n in names]


def _columns(dicts: list[dict[str, Any]]) -> list[str]:
    """Union of the keys in first-seen order."""
    return list(dict.fromkeys(k for d in dicts for k in d))


def next_index(folder: Path, stems: tuple[str, ...] = ("facts", "labels")) -> int:
    """One past the highest <stem>_<n>.csv index in folder, over every stem, so the files of one
    call share an index and no earlier file is ever overwritten."""
    pattern = re.compile(rf"^({'|'.join(stems)})_(\d+)\.csv$")
    found = [int(m.group(2)) for p in folder.iterdir() if (m := pattern.match(p.name))]
    return max(found, default=-1) + 1


def experiment_refs(exp_yaml: Path | None) -> list[ExperimentRef]:
    """One ExperimentRef per block of the exp yaml: folder name, Module, Parameters.mode."""
    if exp_yaml is None:
        return []
    return [ExperimentRef(exp.name, exp.module, str(exp.parameters.get("mode", DEFAULT_MODE)))
            for exp in load_experiments(exp_yaml).values()]


def collect_facts(pairs: list[tuple[str, str]], ctx: FactContext) -> list[SequenceFacts]:
    """One SequenceFacts per (dataset, sequence): every fact of Gym/facts.py, read from disk."""
    facts: list[SequenceFacts] = []
    for dataset, sequence in sorted(set(pairs)):
        values: dict[str, Any] = {}
        for fn in FACT_FUNCTIONS:
            values.update(fn(dataset, sequence, ctx))
        facts.append(SequenceFacts(dataset, sequence, values))
        print_info(f"{dataset}/{sequence}: " + ", ".join(f"{k} {v}" for k, v in values.items()))
    return facts


def label_sequences(facts: list[SequenceFacts], rules: dict) -> list[SequenceLabels]:
    """One SequenceLabels per SequenceFacts: every label of label_definitions.yaml, each computed
    by its function in Gym/labeling.py from the sequence's facts and the label's rule."""
    labels: list[SequenceLabels] = []
    for f in facts:
        values: dict[str, Any] = {}
        for name, rule in rules.items():
            values.update(LABEL_FUNCTIONS[name](f.facts, rule))
        labels.append(SequenceLabels(f.dataset, f.sequence, values))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sequence facts and label them: Gym/labels/facts_<n>.csv + labels_<n>.csv.")
    add_sequence_target_args(parser)
    args = parser.parse_args()
    pairs = resolve_sequence_targets_or_exit(args, parser)
    rules = load_yaml_file(DEFINITIONS_YAML)["labels"]
    check_definitions(rules)
    ctx = FactContext(experiments=experiment_refs(Path(args.exp) if args.exp else None))
    for exp in ctx.experiments:
        print_info(f"result facts from {exp.name}: {exp.baseline}, {exp.mode}")

    LABELS_DIR.mkdir(exist_ok=True)
    index = next_index(LABELS_DIR)
    facts_csv, labels_csv = LABELS_DIR / f"facts_{index:03d}.csv", LABELS_DIR / f"labels_{index:03d}.csv"

    facts = collect_facts(pairs, ctx)
    fact_names = _columns([f.facts for f in facts])
    write_csv_rows(facts_csv, ["dataset", "sequence"] + fact_names, [f.row(fact_names) for f in facts])
    print_info(f"{facts_csv.relative_to(REPO_ROOT)}: {len(facts)} sequences x {len(fact_names)} facts")

    labels = label_sequences(facts, rules)
    label_names = _columns([l.labels for l in labels])
    write_csv_rows(labels_csv, ["dataset", "sequence"] + label_names, [l.row(label_names) for l in labels])
    print_info(f"{labels_csv.relative_to(REPO_ROOT)}: {len(labels)} sequences x {len(label_names)} labels")


if __name__ == "__main__":
    main()
