from __future__ import annotations

import csv
import yaml
import numpy as np
from pathlib import Path
from urllib.parse import urljoin
from typing import Final, Any
from collections.abc import Iterable

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION

MAX_NICKNAME_LEN: Final = 15


class ETH_dataset(DatasetVSLAMLab):
    """ETH dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "eth") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace("mannequin", "mann.")
                                   .replace("einstein_global_light_changes_", "eins.gl.l.ch._")
                                   .replace("desk_global_light_changes","desk.gl.l.ch.")
                                   .replace("einstein", "eins.").replace("global", "gl.")
                                   .replace("camera", "cam.")
                                   .replace("_", " ")[:MAX_NICKNAME_LEN] for s in self.sequence_names]
        
        # Depth factor
        self.depth_factor = cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        for mode in self.modes:
            compressed_name = f"{sequence_name}_{mode}.zip"
            download_url = urljoin(self.url_download_root, f"datasets/{compressed_name}")

            compressed_file = self.dataset_path / compressed_name
            decompressed_folder = self.dataset_path / sequence_name

            if not compressed_file.exists():
                downloadFile(download_url, str(self.dataset_path))
            
            # Decompress only if needed
            needs_depth = not (decompressed_folder / "depth").exists() and not (decompressed_folder / "depth_0").exists()
            needs_mono = not decompressed_folder.exists()

            if (mode == "mono" and needs_mono) or (mode == "rgbd" and needs_depth):
                decompressFile(str(compressed_file), str(self.dataset_path))
     
    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for raw, dst in (("rgb", "rgb_0"), ("depth", "depth_0")):
            src = sequence_path / raw
            tgt = sequence_path / dst
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb0_entries = list(self._iter_entries(sequence_path / "rgb.txt",   "rgb/",   "rgb_0/"))
        depth_entries = list(self._iter_entries(sequence_path / "depth.txt", "depth/", "depth_0/"))

        rgb_csv = sequence_path / "rgb.csv"
        tmp = rgb_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])
            for (ts_r0, path_r0), (ts_d, path_d) in zip(rgb0_entries, depth_entries, strict=True):
                ts_r0_ns = int(float(ts_r0) * 1e9)
                ts_d_ns = int(float(ts_d) * 1e9)
                w.writerow([ts_r0_ns, path_r0, ts_d_ns, path_d])
        tmp.replace(rgb_csv)
        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        with open(sequence_path / "calibration.txt", "r", encoding="utf-8") as f:
            first = f.readline().split()
            fx, fy, cx, cy = map(float, first[:4])
            rgbd0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb+depth", "depth_name": "depth_0",
                    "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                    "depth_factor": float(self.depth_factor),
                    "fps": float(self.rgb_hz),
                    "T_BS": np.eye(4)}
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])
        
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_txt = sequence_path / "groundtruth.txt"
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        if not groundtruth_txt.exists():
            raise FileNotFoundError(f"Missing groundtruth: {groundtruth_txt}")

        with open(groundtruth_txt, "r", encoding="utf-8") as fin, open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"])
            for line in fin:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.split()
                ts_ns = int(float(parts[0]) * 1e9)
                w.writerow([ts_ns] + parts[1:])

        tmp.replace(groundtruth_csv)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

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