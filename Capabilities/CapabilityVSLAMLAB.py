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
handling and the run-side wiring; the capability only receives explicit --sequence-path folders.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from path_constants import VSLAM_LAB_DIR
from utilities import print_msg

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "
SOURCES_DIR = VSLAM_LAB_DIR / "Capabilities" / "sources"


class CapabilityVSLAMLAB:
    """One capability repository: fetch, install and run its `inference` task on sequence folders."""

    def __init__(self, name: str, github_repo: str) -> None:
        self.name = name
        self.github_repo = github_repo  # <owner>/<repo>
        self.path = SOURCES_DIR / name
        self.manifest = self.path / "pixi.toml"

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
        if not self.is_installed():
            self.install()
        self._pixi("inference", "--sequence-path", *map(str, sequence_paths), *(extra_args or []), frozen=True)

    def _pixi(self, task: str, *args: str, frozen: bool = False) -> None:
        # --manifest-path: run in the capability's own environment, not the calling (vslamlab) one
        cmd = ["pixi", "run", "--manifest-path", str(self.manifest), *(["--frozen"] if frozen else []), task, *args]
        subprocess.run(cmd, cwd=self.path, check=True)
