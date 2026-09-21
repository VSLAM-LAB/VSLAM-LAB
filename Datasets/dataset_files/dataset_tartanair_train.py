"""
Module: VSLAM-LAB - Datasets - dataset_tartanair_train.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-24
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Final
from zipfile import ZipFile

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import hf_token, write_csv_rows

CAMERA_PARAMS: Final = [320.0, 320.0, 320.0, 240.0]  # Camera intrinsics (fx, fy, cx, cy)
STEREO_BASELINE_M: Final = 0.25  # Right camera sits 0.25 m along +x of the left camera frame

# Each <env>/<difficulty> group on HF ships one zip per modality, covering every trajectory of
# that group - these three are the ones the mono/stereo/rgbd modes need (pose_*.txt ship inside
# image_left.zip).
GROUP_ZIPS: Final = ("image_left.zip", "image_right.zip", "depth_left.zip")


def _split_sequence_name(sequence_name: str) -> tuple[str, str, str]:
    # Environment names contain underscores (abandonedfactory_night, seasonsforest_winter), so
    # parse from the right: <env>_<difficulty>_<PXXX>.
    env, difficulty, trajectory = sequence_name.rsplit("_", 2)
    return env, difficulty, trajectory


class TartanairTrainDataset(DatasetVSLAMLAB):
    """TartanAir dataset (training split) helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "tartanair-train") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.hf_repo_id: str = self.cfg["hf_repo_id"]

        # Depth factor
        self.depth_factor: float = self.cfg["depth_factor"]

    def _raw_trajectory_path(self, sequence_name: str) -> Path:
        env, difficulty, trajectory = _split_sequence_name(sequence_name)
        return self.dataset_path / env / difficulty / trajectory

    def download_sequence_data(self, sequence_name: str) -> None:
        raw_path = self._raw_trajectory_path(sequence_name)
        marker = raw_path / ".download_complete"
        if marker.exists():
            return

        env, difficulty, trajectory = _split_sequence_name(sequence_name)
        member_prefix = f"{env}/{difficulty}/{trajectory}/"
        for zip_name in GROUP_ZIPS:
            file_path = hf_hub_download(repo_id=self.hf_repo_id, filename=f"{env}/{difficulty}/{zip_name}",
                                        repo_type='dataset', token=hf_token())
            # Each group zip covers every trajectory of its <env>/<difficulty> - extract only
            # this sequence's members.
            with ZipFile(file_path, 'r') as zip_ref:
                members = [m for m in zip_ref.namelist() if m.startswith(member_prefix)]
                zip_ref.extractall(self.dataset_path, members=members)
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        raw_path = self._raw_trajectory_path(sequence_name)

        # Source images are exactly 640x480 (no target_resolution set), so the raw folders are
        # moved into place unresized.
        sequence_path.mkdir(parents=True, exist_ok=True)
        for raw, tgt in (
            ("image_left", self.rgb_path(sequence_name)),
            ("image_right", sequence_path / "rgb_1"),
        ):
            src = raw_path / raw
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

        # Depth ships as float32 NPY in meters (sky/far hits ~10000) - convert to 16-bit PNG,
        # zeroing values that don't fit the depth_factor range (0 = no depth).
        depth_path = self.depth_path(sequence_name)
        depth_path.mkdir(exist_ok=True)
        for npy_file in sorted((raw_path / "depth_left").glob("*.npy")):
            png_file = depth_path / f"{npy_file.stem}.png"
            if png_file.exists():
                continue
            scaled = np.load(npy_file).astype(np.float64) * self.depth_factor
            valid = np.isfinite(scaled) & (scaled > 0) & (scaled <= 65535)
            depth_png = np.where(valid, np.round(scaled), 0).astype(np.uint16)
            cv2.imwrite(str(png_file), depth_png)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_files = sorted(f.name for f in self.rgb_path(sequence_name).iterdir() if f.is_file())
        rgb1_files = sorted(f.name for f in (sequence_path / "rgb_1").iterdir() if f.is_file())
        depth_files = sorted(f.name for f in self.depth_path(sequence_name).iterdir() if f.is_file())

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1",
                  "ts_depth_0 (ns)", "path_depth_0"]
        rows = []
        # Frames are rendered synchronously (left/right/depth correspond 1:1 by index); the
        # source ships no timestamps, so synthesize them from the frame index at rgb_hz.
        for f0, f1, fd in zip(rgb_files, rgb1_files, depth_files, strict=True):
            frame_idx = int(f0.split("_")[0])
            ts_ns = int(1e10 + frame_idx / self.rgb_hz * 1e9)
            rows.append([ts_ns, f"rgb_0/{f0}", ts_ns, f"rgb_1/{f1}", ts_ns, f"depth_0/{fd}"])

        write_csv_rows(self.rgb_csv_path(sequence_name), header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = CAMERA_PARAMS

        T_BS_1 = np.eye(4)
        T_BS_1[0, 3] = STEREO_BASELINE_M

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
        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.rgb_hz),
            "T_BS": T_BS_1,
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb1], rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # pose_left.txt: one "tx ty tz qx qy qz qw" line per frame, NED world frame - written
        # as-is (same convention as dataset_tartanair.py), with timestamps synthesized to match
        # create_rgb_csv.
        pose_txt = self._raw_trajectory_path(sequence_name) / "pose_left.txt"

        rows = []
        with open(pose_txt, "r", encoding="utf-8") as fin:
            for frame_idx, line in enumerate(fin):
                parts = line.strip().split()
                ts_ns = int(1e10 + frame_idx / self.rgb_hz * 1e9)
                tx, ty, tz, qx, qy, qz, qw = parts[:7]
                rows.append([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        write_csv_rows(
            self.groundtruth_csv_path(sequence_name), ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], rows,
        )

    def remove_unused_files(self, sequence_name: str) -> None:
        raw_path = self._raw_trajectory_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            # Parsed into groundtruth.csv (pose_left) / never consumed (pose_right). The raw
            # depth_left NPYs stay at STANDARD: their PNG conversion is quantized, so they count
            # as original source data, not a pure reformat.
            (raw_path / "pose_left.txt").unlink(missing_ok=True)
            (raw_path / "pose_right.txt").unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(raw_path, ignore_errors=True)

        # Prune the now-empty <env>/<difficulty>/<traj> nesting; other trajectories of the same
        # group keep their own subfolders, so rmdir only succeeds when nothing is left.
        for folder in (raw_path, raw_path.parent, raw_path.parent.parent):
            if folder.is_dir() and next(folder.iterdir(), None) is None:
                folder.rmdir()

    def get_download_issues(self, _):
        # Zips are <env>/<difficulty>-scoped: requesting one trajectory still fetches its whole
        # group's image_left/image_right/depth_left zips (~10 GB typical, 29 GB worst case -
        # neighborhood/Easy).
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=29)]
