"""
Module: VSLAM-LAB - Datasets - extra-files - synch_gt.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-24
- Updated: 2026-07-25
- License: GPLv3 License

Synchronizes a sequence's rgb.csv and groundtruth.csv into a strict one-to-one match: each rgb
frame is paired with its nearest groundtruth pose (greedy nearest-neighbor, each timestamp used
at most once) within a threshold derived from the dataset's rgb_hz (half the nominal frame
period); rgb frames with no groundtruth pose inside that threshold are dropped. The pre-sync
files are preserved as rgb_raw.csv / groundtruth_raw.csv next to the synced ones.

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets): a bare <dataset> [<sequence> ...],
or --datasets/--sequences/--exp/--configs for every other shape.
Add --revert to restore the *_raw.csv originals instead of syncing.
"""

import argparse
import bisect
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from path_constants import GROUNTRUTH_FILE
from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv, raw_path, read_csv_rows,
    overwrite_csv_with_backup, revert_csv_from_backup,
)

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def dataset_rgb_hz(dataset_name: str) -> float | None:
    yaml_file = REPO_ROOT / "Datasets" / "dataset_files" / f"dataset_{dataset_name}.yaml"
    if not yaml_file.is_file():
        return None
    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    rgb_hz = cfg.get("rgb_hz")
    return float(rgb_hz) if rgb_hz else None


def associate(rgb_ts: list[int], gt_ts: list[int], threshold_ns: int) -> dict[int, int]:
    """Greedy nearest-neighbor bipartite match (the TUM RGB-D associate.py convention): every
    candidate pair within threshold_ns is considered closest-first, and each rgb/groundtruth
    timestamp is consumed by at most one match."""
    candidates = []
    for i, t in enumerate(rgb_ts):
        pos = bisect.bisect_left(gt_ts, t)
        for j in (pos - 1, pos):
            if 0 <= j < len(gt_ts):
                diff = abs(gt_ts[j] - t)
                if diff <= threshold_ns:
                    candidates.append((diff, i, j))
    candidates.sort(key=lambda c: c[0])

    used_rgb: set[int] = set()
    used_gt: set[int] = set()
    matches: dict[int, int] = {}
    for diff, i, j in candidates:
        if i in used_rgb or j in used_gt:
            continue
        used_rgb.add(i)
        used_gt.add(j)
        matches[i] = j
    return matches


def revert_pair(dataset_name: str, sequence_name: str) -> None:
    seq_path = sequence_path(dataset_name, sequence_name)
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    gt_csv = seq_path / GROUNTRUTH_FILE

    reverted_rgb = revert_csv_from_backup(rgb_csv)
    reverted_gt = revert_csv_from_backup(gt_csv)
    if not reverted_rgb and not reverted_gt:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - nothing to revert (no *_raw.csv found)")
        return

    print_info(f"Reverted {dataset_name}:{sequence_name}")


def synch_pair(dataset_name: str, sequence_name: str) -> None:
    seq_path = sequence_path(dataset_name, sequence_name)
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    gt_csv = seq_path / GROUNTRUTH_FILE
    rgb_raw = raw_path(rgb_csv)
    gt_raw = raw_path(gt_csv)

    if not rgb_csv.exists() or not gt_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv or groundtruth.csv (run 'pixi run download-sequence' first)")
        return

    if rgb_raw.exists() or gt_raw.exists():
        print_info(f"Skipping {dataset_name}:{sequence_name} - already synced (found *_raw.csv, use --revert first)")
        return

    rgb_hz = dataset_rgb_hz(dataset_name)
    if not rgb_hz:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - could not determine rgb_hz from dataset_{dataset_name}.yaml")
        return
    threshold_ns = int(0.5 * (1e9 / rgb_hz))

    rgb_header, rgb_rows = read_csv_rows(rgb_csv)
    gt_header, gt_rows = read_csv_rows(gt_csv)
    if not rgb_rows or not gt_rows:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - empty rgb.csv or groundtruth.csv")
        return

    rgb_ts = [int(row[0]) for row in rgb_rows]
    gt_ts = [int(row[0]) for row in gt_rows]
    matches = associate(rgb_ts, gt_ts, threshold_ns)

    if not matches:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - no rgb/groundtruth pairs matched within {threshold_ns / 1e9:.4f}s")
        return

    matched_indices = sorted(matches)
    new_rgb_rows = [rgb_rows[i] for i in matched_indices]
    new_gt_rows = [gt_rows[matches[i]] for i in matched_indices]

    overwrite_csv_with_backup(rgb_csv, rgb_header, new_rgb_rows)
    overwrite_csv_with_backup(gt_csv, gt_header, new_gt_rows)

    print_info(
        f"Synced {dataset_name}:{sequence_name} - {len(new_rgb_rows)}/{len(rgb_rows)} rgb frames matched "
        f"(threshold {threshold_ns / 1e9:.4f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize a sequence's rgb.csv and groundtruth.csv into a strict one-to-one match."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--revert", action="store_true", help="Restore the *_raw.csv originals instead of syncing")
    args = parser.parse_args()

    pairs = resolve_sequence_targets_or_exit(args, parser)

    action = revert_pair if args.revert else synch_pair
    for dataset_name, sequence_name in pairs:
        action(dataset_name, sequence_name)


if __name__ == "__main__":
    main()
