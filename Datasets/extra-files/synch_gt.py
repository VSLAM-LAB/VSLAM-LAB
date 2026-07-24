"""
Module: VSLAM-LAB - Datasets - extra-files - synch_gt.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-24
- License: GPLv3 License

Synchronizes a sequence's rgb.csv and groundtruth.csv into a strict one-to-one match: each rgb
frame is paired with its nearest groundtruth pose (greedy nearest-neighbor, each timestamp used
at most once) within a threshold derived from the dataset's rgb_hz (half the nominal frame
period); rgb frames with no groundtruth pose inside that threshold are dropped. The pre-sync
files are preserved as rgb_raw.csv / groundtruth_raw.csv next to the synced ones.

Supports three call shapes:
  pixi run synch-gt <dataset> <sequence>   - sync one dataset/sequence pair
  pixi run synch-gt <dataset>              - sync every downloaded sequence for that dataset
  pixi run synch-gt <exp_yaml>             - sync every dataset/sequence pair referenced by the
                                              Config file(s) of an experiment yaml
Add --revert to any of the above to restore the *_raw.csv originals instead of syncing.
"""

import bisect
import csv
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from path_constants import VSLAMLAB_BENCHMARK, RGB_BASE_FOLDER, GROUNTRUTH_FILE

VSLAMLAB_BENCHMARK = Path(os.environ.get("VSLAMLAB_BENCHMARK_OVERRIDE", VSLAMLAB_BENCHMARK))

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


def print_info(msg: str) -> None:
    print(f"{SCRIPT_LABEL}{msg}")


def print_warning(msg: str) -> None:
    print(f"{SCRIPT_LABEL}\033[93m{msg}\033[0m")


def sequence_path(dataset_name: str, sequence_name: str) -> Path:
    return VSLAMLAB_BENCHMARK / dataset_name.upper() / sequence_name


def raw_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_raw{csv_path.suffix}")


def dataset_rgb_hz(dataset_name: str) -> float | None:
    yaml_file = REPO_ROOT / "Datasets" / "dataset_files" / f"dataset_{dataset_name}.yaml"
    if not yaml_file.is_file():
        return None
    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    rgb_hz = cfg.get("rgb_hz")
    return float(rgb_hz) if rgb_hz else None


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    rows.sort(key=lambda row: int(row[0]))
    return header, rows


def write_csv_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


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
    rgb_csv = seq_path / f"{RGB_BASE_FOLDER}.csv"
    gt_csv = seq_path / GROUNTRUTH_FILE
    rgb_raw = raw_path(rgb_csv)
    gt_raw = raw_path(gt_csv)

    if not rgb_raw.exists() and not gt_raw.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - nothing to revert (no *_raw.csv found)")
        return

    if rgb_raw.exists():
        rgb_raw.replace(rgb_csv)
    if gt_raw.exists():
        gt_raw.replace(gt_csv)
    print_info(f"Reverted {dataset_name}:{sequence_name}")


def synch_pair(dataset_name: str, sequence_name: str) -> None:
    seq_path = sequence_path(dataset_name, sequence_name)
    rgb_csv = seq_path / f"{RGB_BASE_FOLDER}.csv"
    gt_csv = seq_path / GROUNTRUTH_FILE
    rgb_raw = raw_path(rgb_csv)
    gt_raw = raw_path(gt_csv)

    if not rgb_csv.exists() or not gt_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv or groundtruth.csv")
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

    # Preserve the pre-sync files before overwriting
    rgb_csv.rename(rgb_raw)
    gt_csv.rename(gt_raw)

    write_csv_rows(rgb_csv, rgb_header, new_rgb_rows)
    write_csv_rows(gt_csv, gt_header, new_gt_rows)

    print_info(
        f"Synced {dataset_name}:{sequence_name} - {len(new_rgb_rows)}/{len(rgb_rows)} rgb frames matched "
        f"(threshold {threshold_ns / 1e9:.4f}s)"
    )


def sequences_for_dataset(dataset_name: str) -> list[str]:
    dataset_path = VSLAMLAB_BENCHMARK / dataset_name.upper()
    if not dataset_path.is_dir():
        print_warning(f"Dataset folder not found: {dataset_path}")
        return []
    return sorted(
        p.name for p in dataset_path.iterdir()
        if p.is_dir() and (p / f"{RGB_BASE_FOLDER}.csv").exists()
    )


def pairs_from_config(config_yaml: Path) -> list[tuple[str, str]]:
    with open(config_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return [
        (dataset_name, sequence_name)
        for dataset_name, sequence_names in config.items()
        for sequence_name in sequence_names
    ]


def pairs_from_exp_yaml(exp_data: dict) -> list[tuple[str, str]]:
    configs_dir = REPO_ROOT / "configs"
    seen_configs: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for exp_name, settings in exp_data.items():
        config_name = (settings or {}).get("Config")
        if not config_name or config_name in seen_configs:
            continue
        seen_configs.add(config_name)

        config_path = configs_dir / config_name
        if not config_path.is_file():
            print_warning(f"Skipping experiment '{exp_name}' - config not found: {config_path}")
            continue

        for pair in pairs_from_config(config_path):
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(pair)
    return pairs


def pairs_from_file(path: Path) -> list[tuple[str, str]]:
    """Dispatch on the YAML's actual shape rather than just its filename:
      - exp_yaml:    top-level values are dicts with a 'Config' key, e.g. {exp_name: {Config: ...}}
      - config_yaml: top-level values are lists of sequence names, e.g. {dataset_name: [seq, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not data:
        print_warning(f"{path} is empty or not valid YAML")
        return []

    values = list(data.values())
    if all(isinstance(v, dict) for v in values):
        return pairs_from_exp_yaml(data)
    if all(isinstance(v, list) for v in values):
        return pairs_from_config(path)

    print_warning(
        f"{path} doesn't look like an exp_yaml ({{name: {{Config: ...}}}}) "
        f"or a config_yaml ({{dataset: [sequences]}}) - mixed/unrecognized entry types"
    )
    return []


def main() -> None:
    args = sys.argv[1:]
    revert = "--revert" in args
    args = [a for a in args if a != "--revert"]
    action = revert_pair if revert else synch_pair

    if len(args) == 2:
        action(*args)
    elif len(args) == 1:
        arg = args[0]
        if os.path.isfile(arg):
            pairs = pairs_from_file(Path(arg))
            if not pairs:
                print_warning(f"No dataset/sequence pairs found in {arg}")
                return
            for dataset_name, sequence_name in pairs:
                action(dataset_name, sequence_name)
        else:
            sequence_names = sequences_for_dataset(arg)
            if not sequence_names:
                print_warning(f"No sequences found for dataset '{arg}'")
                return
            for sequence_name in sequence_names:
                action(arg, sequence_name)
    else:
        print_info(
            "Usage: pixi run synch-gt <dataset> <sequence> [--revert] | "
            "pixi run synch-gt <dataset> [--revert] | "
            "pixi run synch-gt <exp_yaml> [--revert]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
