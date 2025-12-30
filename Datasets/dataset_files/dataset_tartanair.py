
from __future__ import annotations

import os
import csv
import yaml
import shutil
import numpy as np
from pathlib import Path
from urllib.parse import urljoin
from typing import Final, Any

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION, VSLAMLAB_BENCHMARK
from Datasets.DatasetVSLAMLab_issues import _get_dataset_issue

CAMERA_PARAMS: Final = [320.0, 320.0, 320.0, 240.0] # Camera intrinsics (fx, fy, cx, cy)


class TARTANAIR_dataset(DatasetVSLAMLab):
    """TARTANAIR dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "tartanair") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]
        self.url_download_gt_root: str = cfg["url_download_gt_root"]

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

    def download_sequence_data(self, sequence_name: str) -> None:

        # Variables
        compressed_name = 'tartanair-test-mono-release'
        compressed_name_ext = compressed_name + '.tar.gz'
        decompressed_name = compressed_name
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = self.dataset_path / compressed_name_ext
        decompressed_folder = self.dataset_path / decompressed_name

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not decompressed_folder.exists():
            decompressFile(compressed_file, str(self.dataset_path / compressed_name))

        # Download the gt
        if not (self.dataset_path / 'tartanair_cvpr_gt').exists():
            compressed_name = '3p1sf0eljfwrz4qgbpc6g95xtn2alyfk'
            compressed_name_ext = compressed_name + '.zip'
            decompressed_name = 'tartanair_cvpr_gt'

            compressed_file = self.dataset_path / compressed_name_ext
            decompressed_folder = self.dataset_path / decompressed_name

            download_url = self.url_download_gt_root
            if not compressed_file.exists():
                downloadFile(download_url, self.dataset_path)

            decompressFile(compressed_file, self.dataset_path / decompressed_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        rgb_path.mkdir(parents=True, exist_ok=True)

        rgb_path_0 = self.dataset_path / 'tartanair-test-mono-release' / 'mono' / sequence_name
        if not rgb_path_0.exists():
            return

        for png_file in os.listdir(rgb_path_0):
            if png_file.endswith(".png"):
                shutil.move(rgb_path_0 / png_file, rgb_path / png_file)

        shutil.rmtree(rgb_path_0)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        rgb_csv = sequence_path / 'rgb.csv'

        rgb_files = [f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()]
        rgb_files.sort()

        tmp_path = sequence_path / "rgb.csv.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                ts_ns = int(1e10 + ts * 1e9)
                w.writerow([ts_ns, f"rgb_0/{filename}"])

        os.replace(tmp_path, rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = CAMERA_PARAMS
        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}        
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_csv = sequence_path / "groundtruth.csv"
        groundtruth_txt = self.dataset_path / "tartanair_cvpr_gt" / "mono_gt" / f"{sequence_name}.txt"
        tmp_path = sequence_path / "groundtruth.csv.tmp"

        with open(groundtruth_txt, "r", encoding="utf-8") as fin, \
            open(tmp_path, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"])

            frame_idx = 0
            for line in fin:
                line = line.strip()
                parts = line.split()
                ts = frame_idx / float(self.rgb_hz)
                ts_ns = int(1e10 + ts * 1e9)
                frame_idx += 1
                tx, ty, tz, qx, qy, qz, qw = parts[:7]
                w.writerow([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        os.replace(tmp_path, groundtruth_csv)

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=8.2)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)
        
        dataset_folder = self.dataset_path / 'tartanair-test-mono-release'
        if dataset_folder.exists():
            shutil.rmtree(dataset_folder)

        gt_folder = self.dataset_path / 'tartanair_cvpr_gt'
        if gt_folder.exists():
            shutil.rmtree(gt_folder)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (VSLAMLAB_BENCHMARK / f"tartanair-test-mono-release.tar.gz").unlink(missing_ok=True)
            (VSLAMLAB_BENCHMARK / f"3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip").unlink(missing_ok=True)