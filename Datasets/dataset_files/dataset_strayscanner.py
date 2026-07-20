from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas
import yaml
from huggingface_hub import HfApi, HfFileSystem, login
from huggingface_hub.utils import disable_progress_bars

from Datasets.dataset_files.dataset_videos import VIDEOS_dataset
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, HUGGINGFACE_TOKEN, Retention
from utilities import decompressFile


class StrayScanner_dataset(VIDEOS_dataset):
    """Iphone datasets from StrayScanner app"""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "strayscanner") -> None:
        DatasetVSLAMLab.__init__(self, dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.repo_id = cfg["repo_id"]

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

        # Depth factor
        self.depth_factor = cfg["depth_factor"]

        # RGB frequency
        self.rgb_hz = cfg["rgb_hz"]

        # Sequence location
        self.sequence_location = cfg["sequence_location"]

        # Get resolution size
        self.target_resolution = cfg.get("target_resolution", None)

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        sequence_location = self.sequence_location[self.sequence_names.index(sequence_name)]
        if sequence_location == "local":
            print(
                f"Sequence '{sequence_name}' is marked as 'local'. Please ensure the data is available at {self.dataset_path / sequence_name}."
            )
            return
        else:
            if HUGGINGFACE_TOKEN is not None:
                login(token=HUGGINGFACE_TOKEN)
                token = HUGGINGFACE_TOKEN
            else:
                token = os.environ.get("HF_TOKEN")

            api = HfApi(token=token)
            fs = HfFileSystem(token=token)
            disable_progress_bars()

            cache_file = self.dataset_path / "all_files_cache.json"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    all_files = json.load(f)
            else:
                all_files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(all_files, f, indent=2)
                print(f"Fetched and cached {len(all_files)} files")

            for f in all_files:
                if sequence_name in f:
                    local_file = self.dataset_path / f
                    if not local_file.exists():
                        fs.get_file(f"datasets/{self.repo_id}/{f}", str(local_file))
                    break

        if not sequence_path.exists():
            compressed_name = f"{sequence_name}.zip"
            compressed_file = self.dataset_path / compressed_name
            decompressFile(str(compressed_file), str(self.dataset_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for raw, dst in (("depth", "depth_0"),):
            src = sequence_path / raw
            tgt = sequence_path / dst
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

        if not (sequence_path / "rgb_0").exists():
            self.extract_png_frames(
                video_path=sequence_path / "rgb.mp4",
                output_dir=sequence_path / "rgb_0",
                target_resolution=self.target_resolution,
            )

        depth_dir = self.dataset_path / sequence_name / "depth_0"
        for depth_file in sorted(depth_dir.glob("*.png")):
            img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)  # keep 16-bit
            if img.shape[:2] != (self.target_resolution[1], self.target_resolution[0]):
                img = cv2.resize(img, (self.target_resolution[0], self.target_resolution[1]),
                                 interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(depth_file), img)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        rgb_csv = sequence_path / "rgb.csv"
        tmp = rgb_csv.with_suffix(".csv.tmp")

        odometry_csv = pandas.read_csv(sequence_path / "odometry.csv")
        assert isinstance(odometry_csv, pandas.DataFrame) and not odometry_csv.empty, "odometry.csv is missing or empty"

        rgb_path = "rgb_0/{idx:05d}.png"
        depth_path = "depth_0/{idx:06d}.png"

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])
            for i, (_, row) in enumerate(odometry_csv.iterrows()):
                ts = row["timestamp"]
                ts_r0_ns = int(float(ts) * 1e9)
                ts_d_ns = int(float(ts) * 1e9)

                rgb_file = sequence_path / rgb_path.format(idx=i)
                depth_file = sequence_path / depth_path.format(idx=i)
                if rgb_file.exists() and depth_file.exists():
                    w.writerow([ts_r0_ns, rgb_path.format(idx=i), ts_d_ns, depth_path.format(idx=i)])
        tmp.replace(rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        odometry_csv = pandas.read_csv(sequence_path / "odometry.csv")
        assert isinstance(odometry_csv, pandas.DataFrame) and not odometry_csv.empty, "odometry.csv is missing or empty"
        fx = odometry_csv[" fx"].iloc[0].item()
        fy = odometry_csv[" fy"].iloc[0].item()
        cx = odometry_csv[" cx"].iloc[0].item()
        cy = odometry_csv[" cy"].iloc[0].item()

        if self.target_resolution is not None:
            video_path = sequence_path / "rgb.mp4"
            cap = cv2.VideoCapture(video_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            scaled_width, scaled_height = self.estimate_new_resolution(
                original_width, original_height, self.target_resolution
            )
            scale_factor_x = scaled_width / original_width
            scale_factor_y = scaled_height / original_height
            fx *= scale_factor_x
            fy *= scale_factor_y
            cx *= scale_factor_x
            cy *= scale_factor_y

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
        sequence_path = self.dataset_path / sequence_name
        odometry_csv = pandas.read_csv(sequence_path / "odometry.csv")
        assert isinstance(odometry_csv, pandas.DataFrame) and not odometry_csv.empty, "odometry.csv is missing or empty"
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            for _, row in odometry_csv.iterrows():
                ts_ns = int(float(row["timestamp"]) * 1e9)
                w.writerow([ts_ns, row[" x"], row[" y"], row[" z"],
                            row[" qx"], row[" qy"], row[" qz"], row[" qw"]])
        tmp.replace(groundtruth_csv)

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
