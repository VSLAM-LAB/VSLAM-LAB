import csv
import os
import shutil
from typing import Any, Final
from zipfile import ZipFile

import gdown
import numpy as np
import yaml

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, Retention

CAMERA_PARAMS_1024: Final = [610.17789714, 915.2668457, 512.0, 512.0]  # Camera intrinsics (fx, fy, cx, cy)
CAMERA_PARAMS_320: Final = [190.68059285, 286.02088928, 160.0, 160.0]  # Camera intrinsics (fx, fy, cx, cy)


class DRUNKARDS_dataset(DatasetVSLAMLab):
    """DRUNKARDS dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "drunkards") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [f"{self.dataset_name}_{s[:1]}" for s in self.sequence_names]

        # Depth factor
        self.depth_factor = cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        folder_id = self._get_folder_id(sequence_name)
        color_zip = sequence_path / "color.zip"
        if not color_zip.exists():
            folder_url = f"{self.url_download_root}/{folder_id}"
            gdown.download_folder(
                url=folder_url,
                output=str(sequence_path),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,
                resume=True,
            )

        rgb_path = self.rgb_path(sequence_name)
        if not rgb_path.exists():
            with ZipFile(color_zip, "r") as zip_ref:
                zip_ref.extractall(sequence_path)
            color_path = sequence_path / "color"
            color_path.replace(rgb_path)

        depth_path = sequence_path / "depth_0"
        if not depth_path.exists():
            depth_zip = sequence_path / "depth.zip"
            with ZipFile(depth_zip, "r") as zip_ref:
                zip_ref.extractall(sequence_path)
            depth_path_ = sequence_path / "depth"
            depth_path_.replace(depth_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        rgb_files = [f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()]
        rgb_files.sort()

        with open(rgb_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])  # header
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                ts_ns = int(1e10 + ts * 1e9)
                writer.writerow([ts_ns, f"rgb_0/{filename}", ts_ns, f"depth_0/{filename}"])

    def create_calibration_yaml(self, sequence_name: str) -> None:
        if "1024" in sequence_name:
            fx, fy, cx, cy = CAMERA_PARAMS_1024
        if "320" in sequence_name:
            fx, fy, cx, cy = CAMERA_PARAMS_320

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
        groundtruth_txt = sequence_path / "pose.txt"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")
        with open(groundtruth_txt, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8", newline="") as fout:
            lines = fin.readlines()
            data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
            fout.write("ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n")
            for line in data_lines:
                s = line.strip()
                parts = s.split()
                ts_ns = 1e10 + int(float(parts[0]) * 1e9 / self.rgb_hz)
                new_line = f"{ts_ns}," + ",".join(parts[1:]) + "\n"
                fout.write(new_line)
        tmp.replace(groundtruth_csv)
        tmp.unlink(missing_ok=True)

    def remove_unused_files(self, sequence_name: str) -> None:
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            sequence_path = self.sequence_path(sequence_name)
            shutil.rmtree(sequence_path / "pose.txt", ignore_errors=True)
            for zip_file in self.dataset_path.rglob("*.zip"):
                zip_file.unlink(missing_ok=True)

    def _get_folder_id(self, sequence_name):
        if sequence_name == "00000_1024_level0":
            return "1Aa4Tz3ZX_x_RSgCzjCofhpwKuoGCGcP9"
        if sequence_name == "00001_1024_level0":
            return "1SzDa6EWlDLzM6gGzWIKI9f9obZRt36YF"
        if sequence_name == "00002_1024_level0":
            return "1T7qmjjo5tJlHrbA5SYaxmLTq2B0NUyO4"
        if sequence_name == "00000_320_level0":
            return "17TiQ7dWjjjJBMinOgNICyDdZtqvU_WYN"
        if sequence_name == "00001_320_level0":
            return "1iuG1DARacXrEdmmDiRJQk8DqixbwhEah"
