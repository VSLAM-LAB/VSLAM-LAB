"""
Module: VSLAM-LAB - Datasets - dataset_bonn_rgbd.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-27
- License: GPLv3 License
"""

from __future__ import annotations

from typing import Any, Final
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows

# Every zip / extracted folder on the server carries this prefix (rgbd_bonn_balloon.zip ->
# rgbd_bonn_balloon/); the yaml's sequence_names drop it, so it's re-added here.
RAW_PREFIX: Final = "rgbd_bonn_"

TIME_DIFF_THRESH: Final = 0.02  # seconds for RGB/Depth association (TUM RGB-D convention)

# One global RGB calibration for the whole dataset (ASUS Xtion Pro Live, 640x480), as published
# on the dataset page: (fx, fy, cx, cy, k1, k2, p1, p2, k3). Depth is already registered to RGB.
CAMERA_PARAMS: Final = (542.822841, 542.576870, 315.593520, 237.756098,
                        0.039903, -0.099343, -0.000730, -0.000144, 0.000000)


class BonnRgbdDataset(DatasetVSLAMLAB):
    """Bonn RGB-D Dynamic Dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "bonn-rgbd") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

        # Depth factor
        self.depth_factor = self.cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        raw_name = self._raw_name(sequence_name)

        compressed_file = self.dataset_path / f"{raw_name}.zip"
        decompressed_folder = self.dataset_path / raw_name
        download_url = urljoin(self.url_download_root.rstrip("/") + "/", compressed_file.name)

        if sequence_path.exists():
            return

        if not compressed_file.exists():
            downloadFile(download_url, str(self.dataset_path))

        # The zip unpacks to rgbd_bonn_<sequence>/; renaming it to sequence_path only once the
        # extraction has finished doubles as the completion marker - a crash mid-extraction
        # leaves the prefixed folder behind, never a half-filled sequence_path.
        decompressFile(str(compressed_file), str(self.dataset_path))
        decompressed_folder.replace(sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Source frames are already 640x480 (no target_resolution in the yaml), so the TUM-style
        # rgb/ and depth/ folders are renamed into place unchanged, as dataset_rgbdtum.py does.
        sequence_path = self.sequence_path(sequence_name)
        for raw, tgt in (("rgb", self.rgb_path(sequence_name)), ("depth", self.depth_path(sequence_name))):
            src = sequence_path / raw
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        """Associate RGB and depth by nearest timestamp within TIME_DIFF_THRESH.

        Like TUM RGB-D, the Xtion emits RGB and depth as two independently-timestamped streams
        (rgb.txt/depth.txt row counts differ by a frame or so per sequence), so the two are merged
        by nearest timestamp (pandas.merge_asof) rather than zipped by index - see
        dataset_rgbdtum.py.
        """
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        rgb_txt = sequence_path / "rgb.txt"
        depth_txt = sequence_path / "depth.txt"
        if not (rgb_txt.exists() and depth_txt.exists()):
            raise FileNotFoundError(f"Missing rgb.txt/depth.txt in {sequence_path}")

        rgb = pd.read_csv(rgb_txt, sep=r"\s+", comment="#", header=None, names=["ts", "rgb_path"])
        depth = pd.read_csv(depth_txt, sep=r"\s+", comment="#", header=None, names=["ts", "depth_path"])
        rgb = rgb.sort_values("ts").reset_index(drop=True)
        depth = depth.sort_values("ts").reset_index(drop=True)

        merged = pd.merge_asof(rgb, depth, on="ts", direction="nearest", tolerance=TIME_DIFF_THRESH)
        merged = merged.dropna(subset=["depth_path"]).copy()

        rgb_folder = self.rgb_path(sequence_name).name
        depth_folder = self.depth_path(sequence_name).name
        merged["ts_rgb_0 (ns)"] = (merged["ts"] * 1e9).astype(int)
        merged["ts_depth_0 (ns)"] = (merged["ts"] * 1e9).astype(int)
        merged["path_rgb_0"] = merged["rgb_path"].astype(str).str.replace(r"^rgb/", f"{rgb_folder}/", regex=True)
        merged["path_depth_0"] = merged["depth_path"].astype(str).str.replace(r"^depth/", f"{depth_folder}/", regex=True)

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"]
        write_csv_rows(rgb_csv, header, merged[header].values.tolist())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = (float(v) for v in CAMERA_PARAMS)

        rgbd0: dict[str, Any] = {
            "cam_name": self.rgb_path(sequence_name).name,
            "cam_type": "rgb+depth",
            "depth_name": self.depth_path(sequence_name).name,
            "cam_model": "pinhole",
            "distortion_type": "radtan5",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "distortion_coefficients": [k1, k2, p1, p2, k3],
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # Optitrack Prime 13 trajectory in TUM format: "# comment" lines, then
        # "timestamp tx ty tz qx qy qz qw" per row.
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_txt = sequence_path / "groundtruth.txt"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        if groundtruth_csv.exists():
            return
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
                rows.append([ts_ns] + parts[1:8])

        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        if BENCHMARK_RETENTION != Retention.FULL:
            for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
                (sequence_path / name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / f"{self._raw_name(sequence_name)}.zip").unlink(missing_ok=True)

    @staticmethod
    def _raw_name(sequence_name: str) -> str:
        """Name the source uses for this sequence's zip / extracted folder."""
        return f"{RAW_PREFIX}{sequence_name}"
