"""
Module: VSLAM-LAB - Datasets - dataset_eth.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 2.0
- Created: 2024-07-13
- Updated: 2026-07-26
- License: GPLv3 License
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final
from urllib.parse import urljoin

import numpy as np

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows

MAX_NICKNAME_LEN: Final = 15


class EthDataset(DatasetVSLAMLAB):
    """ETH3D dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "eth") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [
            s.replace("mannequin", "mann.")
            .replace("einstein_global_light_changes_", "eins.gl.l.ch._")
            .replace("desk_global_light_changes", "desk.gl.l.ch.")
            .replace("einstein", "eins.")
            .replace("global", "gl.")
            .replace("camera", "cam.")
            .replace("_", " ")[:MAX_NICKNAME_LEN]
            for s in self.sequence_names
        ]

        # Depth factor
        self.depth_factor = self.cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        for mode in self.modes:
            compressed_name = f"{sequence_name}_{mode}.zip"
            download_url = urljoin(self.url_download_root, f"datasets/{compressed_name}")

            compressed_file = self.dataset_path / compressed_name
            decompressed_folder = self.sequence_path(sequence_name)

            if not compressed_file.exists():
                downloadFile(download_url, str(self.dataset_path))

            # Decompress only if needed
            needs_depth = (
                not (decompressed_folder / "depth").exists() and not (decompressed_folder / "depth_0").exists()
            )
            needs_mono = not decompressed_folder.exists()

            if (mode == "mono" and needs_mono) or (mode == "rgbd" and needs_depth):
                decompressFile(str(compressed_file), str(self.dataset_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for raw, tgt in (("rgb", self.rgb_path(sequence_name)), ("depth", self.depth_path(sequence_name))):
            src = sequence_path / raw
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb0_entries = list(self._iter_entries(sequence_path / "rgb.txt", "rgb/", "rgb_0/"))
        depth_entries = list(self._iter_entries(sequence_path / "depth.txt", "depth/", "depth_0/"))

        rows = []
        for (ts_r0, path_r0), (ts_d, path_d) in zip(rgb0_entries, depth_entries, strict=True):
            ts_r0_ns = int(float(ts_r0) * 1e9)
            ts_d_ns = int(float(ts_d) * 1e9)
            rows.append([ts_r0_ns, path_r0, ts_d_ns, path_d])

        write_csv_rows(
            self.rgb_csv_path(sequence_name), ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"], rows,
        )

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        with open(sequence_path / "calibration.txt", "r", encoding="utf-8") as f:
            first = f.readline().split()
            fx, fy, cx, cy = map(float, first[:4])
            rgbd0: dict[str, Any] = {
                "cam_name": "rgb_0",
                "cam_type": "rgb+depth",
                "depth_name": "depth_0",
                "cam_model": "pinhole",
                "focal_length": [fx, fy],
                "principal_point": [cx, cy],
                "depth_factor": float(self.depth_factor),
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4),
            }
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_txt = sequence_path / "groundtruth.txt"

        if not groundtruth_txt.exists():
            raise FileNotFoundError(f"Missing groundtruth: {groundtruth_txt}")

        rows = []
        with open(groundtruth_txt, "r", encoding="utf-8") as fin:
            for line in fin:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.split()
                ts_ns = int(float(parts[0]) * 1e9)
                rows.append([ts_ns] + parts[1:])

        write_csv_rows(self.groundtruth_csv_path(sequence_name), ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        if BENCHMARK_RETENTION != Retention.FULL:
            for name in ("calibration.txt", "groundtruth.txt", "rgb.txt", "depth.txt", "associated.txt"):
                (sequence_path / name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            for mode in self.modes:
                (self.dataset_path / f"{sequence_name}_{mode}.zip").unlink(missing_ok=True)

    @staticmethod
    def _iter_entries(txt_path: Path, old_prefix: str, new_prefix: str) -> Iterable[tuple[str, str]]:
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing file: {txt_path}")
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                ts, path = s.split(None, 1)
                if path.startswith(old_prefix):
                    path = new_prefix + path[len(old_prefix):]
                yield ts, path
