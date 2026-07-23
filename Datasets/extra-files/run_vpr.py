"""
Module: VSLAM-LAB - Datasets - extra-files - run_vpr.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-07-23
- Updated: 2026-07-23
- License: GPLv3 License

Dispatches VPR-LAB evaluation runs from the pixi `vpr` task. Supports three call shapes:
  pixi run vpr <dataset> <sequence>   - run one dataset/sequence pair
  pixi run vpr <dataset>              - run every sequence found for that dataset
  pixi run vpr <config.yaml>          - run every dataset/sequence pair listed in a config yaml
                                         (same `dataset: [sequences]` format as configs/config_*.yaml)

"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from path_constants import VSLAMLAB_BENCHMARK, RGB_BASE_FOLDER, HUGGINGFACE_TOKEN

VSLAMLAB_BENCHMARK = Path(os.environ.get("VSLAMLAB_BENCHMARK_OVERRIDE", VSLAMLAB_BENCHMARK))

VPR_LAB_DIR = REPO_ROOT / "Baselines" / "VPR-LAB"
DEFAULT_METHOD = "megaloc"

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


def print_info(msg: str) -> None:
    print(f"{SCRIPT_LABEL}{msg}")


def print_warning(msg: str) -> None:
    print(f"{SCRIPT_LABEL}\033[93m{msg}\033[0m")


def sequence_rgb_csv(dataset_name: str, sequence_name: str) -> Path:
    return VSLAMLAB_BENCHMARK / dataset_name.upper() / sequence_name / f"{RGB_BASE_FOLDER}.csv"


def sequence_log_dir(dataset_name: str, sequence_name: str) -> Path:
    return VSLAMLAB_BENCHMARK / dataset_name.upper() / sequence_name / "vpr-lab"


def run_vpr_pair(dataset_name: str, sequence_name: str, method: str = DEFAULT_METHOD) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    if not rgb_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing {rgb_csv}")
        return

    log_dir = sequence_log_dir(dataset_name, sequence_name)
    cmd = ["pixi", "run", "vpr-methods", method, "None", str(rgb_csv), str(rgb_csv), str(log_dir)]

    env = os.environ.copy()
    if HUGGINGFACE_TOKEN is not None:
        env["HF_TOKEN"] = HUGGINGFACE_TOKEN

    print_info(f"Running vpr-methods ({method}): {dataset_name}:{sequence_name}")
    subprocess.run(cmd, cwd=VPR_LAB_DIR, check=True, env=env)


def sequences_for_dataset(dataset_name: str) -> list[str]:
    dataset_path = VSLAMLAB_BENCHMARK / dataset_name.upper()
    if not dataset_path.is_dir():
        print_warning(f"Dataset folder not found: {dataset_path}")
        return []
    return sorted(
        p.name for p in dataset_path.iterdir()
        if p.is_dir() and (p / f"{RGB_BASE_FOLDER}.csv").exists()
    )


def run_from_config(config_yaml: str) -> None:
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f) or {}
    for dataset_name, sequence_names in config.items():
        for sequence_name in sequence_names:
            run_vpr_pair(dataset_name, sequence_name)


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 2:
        run_vpr_pair(*args)
    elif len(args) == 1:
        arg = args[0]
        if os.path.isfile(arg):
            run_from_config(arg)
        else:
            sequence_names = sequences_for_dataset(arg)
            if not sequence_names:
                print_warning(f"No sequences found for dataset '{arg}'")
                return
            for sequence_name in sequence_names:
                run_vpr_pair(arg, sequence_name)
    else:
        print_info("Usage: pixi run vpr <dataset> <sequence> | pixi run vpr <dataset> | pixi run vpr <config.yaml>")
        sys.exit(1)


if __name__ == "__main__":
    main()
