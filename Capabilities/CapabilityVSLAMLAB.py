"""
Module: VSLAM-LAB - Capabilities - CapabilityVSLAMLAB.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-10-10
- Updated: 2026-10-10
- License: GPLv3 License

CapabilityVSLAMLAB: a capability's own repository (github.com/VSLAM-LAB/<name>), cloned to
Capabilities/sources/<name>, with its own pixi.toml (model stack) and a `vslamlab_<name>.py` entry point
behind the `install` (weights prefetch) and `inference` pixi tasks. VSLAM-LAB keeps the sequence-target
handling and the run-side wiring; the capability only receives explicit --sequence-path folders. A capability
that shares a baseline's checkout (depth_anything in Baselines/Depth-Anything-3) passes its `path` and task.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from path_constants import VSLAM_LAB_DIR
from utilities import print_msg

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "
SOURCES_DIR = VSLAM_LAB_DIR / "Capabilities" / "sources"


class CapabilityVSLAMLAB:
    """One capability repository: fetch, install and run its `inference` task on sequence folders."""

    def __init__(self, name: str, github_repo: str, path: Path | None = None, inference_task: str = "inference") -> None:
        self.name = name
        self.github_repo = github_repo  # <owner>/<repo>
        self.path = path or SOURCES_DIR / name
        self.manifest = self.path / "pixi.toml"
        self.inference_task = inference_task

    def has_source(self) -> bool:
        return (self.path / ".git").exists()

    def is_installed(self) -> bool:
        return self.has_source() and (self.path / ".pixi" / "envs" / "default").exists()

    def fetch_source(self) -> None:
        if self.has_source():
            return
        print_msg(SCRIPT_LABEL, f"cloning {self.github_repo} into {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--recursive", f"https://github.com/{self.github_repo}.git", str(self.path)],
                       check=True)

    def install(self) -> None:
        """Clone if needed, then the capability's `install` task (environment + weights prefetch)."""
        self.fetch_source()
        print_msg(SCRIPT_LABEL, f"installing {self.name} ({self.manifest})")
        self._pixi("install")

    def run(self, sequence_paths: list[Path], extra_args: list[str] | None = None) -> None:
        """The capability's `inference` task on explicit sequence folders (installs first if needed)."""
        self.run_args(["--sequence-path", *map(str, sequence_paths), *(extra_args or [])])

    def run_args(self, args: list[str]) -> None:
        """The capability's `inference` task with arbitrary arguments, for a capability that does not work on
        sequence folders (placecell takes a matrix). Installs first if needed."""
        if not self.is_installed():
            self.install()
        self._pixi(self.inference_task, *args, frozen=True)

    def _pixi(self, task: str, *args: str, frozen: bool = False) -> None:
        # --manifest-path: run in the capability's own environment, not the calling (vslamlab) one
        cmd = ["pixi", "run", "--manifest-path", str(self.manifest), *(["--frozen"] if frozen else []), task, *args]
        subprocess.run(cmd, cwd=self.path, check=True)


def add_rgbd_modes(dataset_name: str) -> None:
    """Once a depth capability's output exists for a sequence, the dataset can run rgbd experiments: add 'rgbd' to
    the modes list in Datasets/dataset_files/dataset_<name>.yaml, plus 'rgbd-vi' when the dataset already supports
    'mono-vi' (imu present). Follows the existing ordering convention ('rgbd' after 'mono', 'rgbd-vi' after
    'mono-vi'); line-based and idempotent."""
    dataset_yaml = VSLAM_LAB_DIR / "Datasets" / "dataset_files" / f"dataset_{dataset_name}.yaml"
    if not dataset_yaml.exists():
        print_msg(SCRIPT_LABEL, f"{dataset_yaml} not found; cannot add 'rgbd' to the dataset modes", flag="warning")
        return

    lines = dataset_yaml.read_text().splitlines()
    idx = next((i for i, line in enumerate(lines) if line.startswith("modes:")), None)
    modes = yaml.safe_load(lines[idx].split("modes:", 1)[1]) if idx is not None else None
    if not isinstance(modes, list):
        print_msg(SCRIPT_LABEL, f"no parseable 'modes:' list in {dataset_yaml.name}; cannot add 'rgbd'", flag="warning")
        return

    def insert_after(mode: str, anchor: str) -> None:
        if mode not in new_modes:
            pos = new_modes.index(anchor) + 1 if anchor in new_modes else len(new_modes)
            new_modes.insert(pos, mode)

    new_modes = list(modes)
    insert_after("rgbd", "mono")
    if "mono-vi" in new_modes:
        insert_after("rgbd-vi", "mono-vi")
    if new_modes == modes:
        return  # already up to date

    lines[idx] = "modes: [" + ", ".join(f"'{mode}'" for mode in new_modes) + "]"
    dataset_yaml.write_text("\n".join(lines) + "\n")
    print_msg(SCRIPT_LABEL, f"{dataset_name} - modes updated to {new_modes} in {dataset_yaml.name}")
