"""
Module: VSLAM-LAB - Datasets - extra-files - run_vpr.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-23
- Updated: 2026-07-25
- License: GPLv3 License

Dispatches VPR-LAB evaluation runs from the pixi `vpr` task. Target arguments follow
CLAUDE.md's sequence-target argument convention (see utilities.add_sequence_target_args /
resolve_sequence_targets):
  pixi run vpr <dataset> [<sequence> ...]      - one dataset, all or specific sequences of it
  pixi run vpr --datasets d1 d2 ...            - every downloaded sequence of each dataset
  pixi run vpr --sequences d1 s1 s2 --sequences d2 s1 ...   - repeatable, explicit sequences per dataset
  pixi run vpr --exp exp.yaml                  - every pair referenced by an exp yaml's Config file(s)
  pixi run vpr --configs config.yaml           - every pair listed in a config yaml (dataset: [sequences])

Sequences that already have a D.npy are skipped by default; add --overwrite to recompute them.
--method selects the VPR method to run (default: megaloc).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from path_constants import HUGGINGFACE_TOKEN
from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv,
)

VPR_LAB_DIR = REPO_ROOT / "Baselines" / "VPR-LAB"
DEFAULT_METHOD = "megaloc"

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def sequence_vpr_dir(dataset_name: str, sequence_name: str) -> Path:
    """<sequence_path>/vpr-lab - where `pixi run vpr` writes its D matrix and other VPR-LAB
    outputs. Shared with sample_vpr.py (which reads sequence_d_matrix() back), imported from
    here rather than utilities.py since it's specific to this pair of VPR scripts."""
    return sequence_path(dataset_name, sequence_name) / "vpr-lab"


def sequence_d_matrix(dataset_name: str, sequence_name: str) -> Path:
    return sequence_vpr_dir(dataset_name, sequence_name) / "D.npy"


def run_vpr_pair(dataset_name: str, sequence_name: str, *, method: str = DEFAULT_METHOD, overwrite: bool = False) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    if not rgb_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing {rgb_csv} (run 'pixi run download-sequence' first)")
        return

    d_matrix_path = sequence_d_matrix(dataset_name, sequence_name)
    if d_matrix_path.exists() and not overwrite:
        print_info(f"Skipping {dataset_name}:{sequence_name} - {d_matrix_path.name} already exists (use --overwrite to recompute)")
        return

    log_dir = sequence_vpr_dir(dataset_name, sequence_name)
    cmd = ["pixi", "run", "vpr-methods", method, "None", str(rgb_csv), str(rgb_csv), str(log_dir)]

    env = os.environ.copy()
    if HUGGINGFACE_TOKEN is not None:
        env["HF_TOKEN"] = HUGGINGFACE_TOKEN

    print_info(f"Running vpr-methods ({method}): {dataset_name}:{sequence_name}")
    subprocess.run(cmd, cwd=VPR_LAB_DIR, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch VPR-LAB evaluation runs.")
    add_sequence_target_args(parser)
    parser.add_argument("--method", default=DEFAULT_METHOD, help=f"VPR method to run (default: {DEFAULT_METHOD})")
    parser.add_argument("--overwrite", action="store_true", help="Recompute D.npy even if it already exists for a sequence")
    args = parser.parse_args()

    pairs = resolve_sequence_targets_or_exit(args, parser)
    for dataset_name, sequence_name in pairs:
        run_vpr_pair(dataset_name, sequence_name, method=args.method, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
