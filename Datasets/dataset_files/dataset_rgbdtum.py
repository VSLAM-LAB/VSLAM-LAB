"""
Module: VSLAM-LAB - Datasets - dataset_rgbdtum.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2024-07-13
- Updated: 2026-07-26
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

TIME_DIFF_THRESH: Final = 0.02  # seconds for RGB/Depth association

CAMERA_PARAMS = { # Camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2, k3)
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989, 0.262383, -0.953104, -0.005358,  0.002628, 1.163314),
    "freiburg2": (520.908620, 521.007327, 325.141443, 249.701764, 0.231222, -0.784899, -0.003257, -0.000105, 0.917205),
    "freiburg3": (535.4,      539.2,      320.1,      247.6,      0.0,      0.0,       0.0,       0.0,       0.0),
}


class RgbdtumDataset(DatasetVSLAMLAB):
    """TUM RGB-D dataset helper for VSLAM-LAB benchmark."""
    
    def __init__(self, dataset_name: str = "rgbdtum") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [self._nickname(s) for s in self.sequence_names]
        self.sequence_nicknames = [s.replace('near', 'n') for s in self.sequence_nicknames]
        self.sequence_nicknames = [s.replace('far', 'f') for s in self.sequence_nicknames]
        self.sequence_nicknames = [s.replace('with person', 'wp') for s in self.sequence_nicknames]
        self.sequence_nicknames = [s.replace('long office household', 'office') for s in self.sequence_nicknames]

        # Depth factor
        self.depth_factor = self.cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        camera = self._camera_from_sequence(sequence_name)

        # .tgz layout on server: <root>/<camera>/<sequence>.tgz
        compressed_name = f"{sequence_name}.tgz"
        download_url = urljoin(self.url_download_root.rstrip("/") + "/", f"{camera}/{compressed_name}")

        # Some archives unpack into a folder whose name differs (validation → secret for f1/f2)
        decompressed_name = sequence_name.replace("validation", "secret") if camera in ("freiburg1", "freiburg2") else sequence_name
        compressed_file = self.dataset_path / compressed_name
        decompressed_folder = self.dataset_path / decompressed_name

        if not compressed_file.exists():
            downloadFile(download_url, str(self.dataset_path))

        if not sequence_path.exists():
            decompressFile(str(compressed_file), str(self.dataset_path))
            # If archive expands to a different folder name, normalize it to sequence_path
            if decompressed_folder.exists() and decompressed_folder != sequence_path:
                decompressed_folder.replace(sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for raw, tgt in (("rgb", self.rgb_path(sequence_name)), ("depth", self.depth_path(sequence_name))):
            src = sequence_path / raw
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        """Associate RGB and Depth using nearest timestamp within tolerance.

        TUM's Kinect emits RGB and depth as two independently-timestamped streams (not a single
        hardware-synchronized capture), so unlike most rgbd datasets in this repo - which just
        list rgb_0/ and depth_0/ and zip the sorted filenames by index, assuming each pair is
        already the same frame (see dataset_eth.py/dataset_nuim.py/dataset_replica.py/
        dataset_7scenes.py) - a naive index-zip here would silently pair frames from different
        moments. This instead reads each stream's own timestamps from rgb.txt/depth.txt and
        associates them via a nearest-timestamp merge (pandas.merge_asof) within
        TIME_DIFF_THRESH, dropping any RGB frame with no depth frame close enough in time.
        """
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        rgb_txt = sequence_path / "rgb.txt"
        depth_txt = sequence_path / "depth.txt"
        if not (rgb_txt.exists() and depth_txt.exists()):
            raise FileNotFoundError(f"Missing rgb/depth txt in {sequence_path}")

        # Load monotonically sorted timestamps
        rgb = pd.read_csv(rgb_txt, sep=r"\s+", comment="#", header=None, names=["ts", "rgb_path"])
        depth = pd.read_csv(depth_txt, sep=r"\s+", comment="#", header=None, names=["ts", "depth_path"])
        rgb = rgb.sort_values("ts").reset_index(drop=True)
        depth = depth.sort_values("ts").reset_index(drop=True)

        # As-of merge finds nearest earlier match; we do symmetric by duplicating with reversed
        # but here TUM is dense and ordered, so a forward asof then post-check tolerance works well.
        merged = pd.merge_asof(rgb, depth, on="ts", direction="nearest", tolerance=TIME_DIFF_THRESH)
        merged = merged.dropna(subset=["depth_path"]).copy()

        # Format + path prefix fixes
        merged["ts_rgb_0 (ns)"] = (merged["ts"] * 1e9).astype(int)
        merged["ts_depth_0 (ns)"] = (merged["ts"] * 1e9).astype(int)
        merged["path_rgb_0"] = merged["rgb_path"].astype(str).str.replace(r"^rgb/", "rgb_0/", regex=True)
        merged["path_depth_0"] = merged["depth_path"].astype(str).str.replace(r"^depth/", "depth_0/", regex=True)

        out = merged[["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"]]
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"], out.values.tolist())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        camera = self._camera_from_sequence(sequence_name)
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = CAMERA_PARAMS[camera]

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
        if camera in ("freiburg1", "freiburg2"):
            rgbd0["distortion_type"] = "radtan5"
            rgbd0["distortion_coefficients"] = [k1, k2, p1, p2, k3]

        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_txt = sequence_path / "groundtruth.txt"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        if not groundtruth_txt.exists():
            return
        
        if groundtruth_csv.exists() and groundtruth_csv.stat().st_mtime >= groundtruth_txt.stat().st_mtime:
            return

        tmp = groundtruth_csv.with_suffix(".csv.tmp")
        with open(groundtruth_txt, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8", newline="") as fout:
            # Skip first 3 lines (header/comments), then write CSV header + values
            lines = fin.readlines()
            data_lines = [ln.strip() for ln in lines[3:] if ln.strip() and not ln.lstrip().startswith("#")]
            fout.write("ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n")
            for line in data_lines:
                s = line.strip()
                parts = s.split()
                ts_ns = int(float(parts[0]) * 1e9)
                new_line = f"{ts_ns}," + ",".join(parts[1:]) + "\n"    
                fout.write(new_line)
        tmp.replace(groundtruth_csv)
        tmp.unlink(missing_ok=True)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        if BENCHMARK_RETENTION != Retention.FULL:
            for name in ("accelerometer.txt", "depth.txt", "groundtruth.txt", "rgb.txt"):
                (sequence_path / name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / f"{sequence_name}.tgz").unlink(missing_ok=True)

    @staticmethod
    def _nickname(seq: str) -> str:
        s = seq.replace("rgbd_dataset_freiburg", "fr")
        s = s.replace("_", " ")
        s = s.replace("validation", "v").replace("structure", "st").replace("texture", "tx")
        s = s.replace("walking xyz", "walk")
        return s
    
    @staticmethod
    def _camera_from_sequence(name: str) -> str:
        for cam in ("freiburg1", "freiburg2", "freiburg3"):
            if cam in name:
                return cam
        raise ValueError(f"Cannot infer camera from sequence name: {name}")