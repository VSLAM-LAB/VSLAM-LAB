"""
Module: VSLAM-LAB - Gym - select_sequences.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 0.1
- Created: 2026-09-27
- Updated: 2026-09-27
- License: GPLv3 License

Selects the sequences of a labels csv that match every given label and writes them as a config
yaml (`dataset: [sequence, ...]`, the shape of configs/config_*.yaml), ready to be the `Config:`
of an experiment yaml.

    pixi run select-sequences Gym/labels/labels_000.csv --labels length=short [--output Gym/Configs/gym_short.yaml]

Each --labels item is <label><op><value> with <op> one of = <= < >= >. `=` compares as text.
The others compare by position in the ordered `values` list of the label's entry in
Gym/label_definitions.yaml (the entry is the column name up to its first ':', so
ate:droidslam:mono uses the `ate` entry): 'length<=medium' keeps short and medium. Quote items
with < or > in the shell. A sequence is kept only when all items match; an empty cell matches
nothing. Without --output the yaml is printed.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utilities import load_yaml_file, make_printers  # noqa: E402

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

DEFINITIONS_YAML = REPO_ROOT / "Gym" / "label_definitions.yaml"
_ITEM = re.compile(r"^(?P<name>[^<>=]+)(?P<op><=|>=|<|>|=)(?P<value>.+)$")
_ORDER_OPS = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}


def parse_labels(items: list[str]) -> dict[str, tuple[str, str]]:
    """['length<=medium', ...] -> {'length': ('<=', 'medium'), ...}."""
    labels: dict[str, tuple[str, str]] = {}
    for item in items:
        m = _ITEM.match(item)
        if not m:
            sys.exit(f"{SCRIPT_LABEL}--labels items must be <label><op><value> with op = <= < >= >, got '{item}'")
        name = m.group("name")
        if name in labels:
            sys.exit(f"{SCRIPT_LABEL}label '{name}' given twice; a sequence has one value per label")
        labels[name] = (m.group("op"), m.group("value"))
    return labels


def tier_order(name: str, definitions: dict) -> list[str]:
    """The ordered `values` of the label's definition entry, for < <= > > comparisons."""
    entry = definitions.get(name.split(":")[0])
    values = entry.get("values") if isinstance(entry, dict) else None
    if not isinstance(values, list):
        sys.exit(f"{SCRIPT_LABEL}'{name}' has no ordered values in {DEFINITIONS_YAML.name}; only = applies")
    return values


def matches(cell: str, op: str, value: str, order: list[str] | None) -> bool:
    if op == "=":
        return cell == value
    assert order is not None
    if value not in order:
        sys.exit(f"{SCRIPT_LABEL}'{value}' is not one of {order}")
    return cell in order and _ORDER_OPS[op](order.index(cell), order.index(value))


def select_sequences(labels_csv: Path, labels: dict[str, tuple[str, str]], definitions: dict) -> dict[str, list[str]]:
    """{dataset: [sequence, ...]} of the rows whose every named column matches its (op, value)."""
    orders = {name: tier_order(name, definitions) if op != "=" else None for name, (op, _) in labels.items()}
    with labels_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = [name for name in labels if name not in columns]
        if missing:
            sys.exit(f"{SCRIPT_LABEL}{labels_csv}: no column {missing}; columns are {columns}")
        selected: dict[str, list[str]] = {}
        for row in reader:
            if all(matches(row[name], op, value, orders[name]) for name, (op, value) in labels.items()):
                selected.setdefault(row["dataset"], []).append(row["sequence"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the sequences matching every label and write them as a config yaml.")
    parser.add_argument("labels_csv", type=Path, help="a Gym/labels/labels_<n>.csv")
    parser.add_argument("--labels", nargs="+", required=True, metavar="LABEL<op>VALUE",
                        help="labels a sequence must all match, e.g. length=short 'length<=medium' (quote < and >)")
    parser.add_argument("--output", type=Path, help="config yaml to write; printed to stdout when omitted")
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    definitions = load_yaml_file(DEFINITIONS_YAML)["labels"]
    selected = select_sequences(args.labels_csv, labels, definitions)
    n = sum(len(seqs) for seqs in selected.values())
    header = (f"# Selected from {args.labels_csv} with "
              + " ".join(f"{k}{op}{v}" for k, (op, v) in labels.items()) + f": {n} sequences\n")
    body = yaml.safe_dump(selected, sort_keys=True, default_flow_style=False) if selected else "{}\n"

    if args.output is None:
        print(header + body, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + body)
    print_info(f"{args.output}: {n} sequences from {len(selected)} datasets")


if __name__ == "__main__":
    main()
