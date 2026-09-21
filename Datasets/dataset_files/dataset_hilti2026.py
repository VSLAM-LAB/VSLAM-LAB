"""
Module: VSLAM-LAB - Datasets - dataset_hilti2026.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-02
- License: GPLv3 License
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any, Final

import gdown
import numpy as np
import yaml
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import (
    compute_scaled_size, downloadFile, patch_ros2_qos_profiles_metadata, run_rosbag_frame_extraction,
    scale_intrinsics, write_csv_rows,
)

# Both cameras stream compressed RGB (README's own topic table: "Front/Back Camera data (RGB, ...)").
IMAGE_TOPIC_TEMPLATE = "/cam{cam}/image_raw/compressed"
IMU_TOPIC = "/imu/data_raw"
ROSBAG_NAME = "rosbag.db3"

GITHUB_RAW_ROOT: Final = "https://raw.githubusercontent.com/Hilti-Research/hilti-trimble-slam-challenge-2026/main"
CALIBRATION_IMUCAM_PATH: Final = "config/hilti_openvins/kalibr_imucam_chain.yaml"
CALIBRATION_IMU_PATH: Final = "config/hilti_openvins/kalibr_imu_chain.yaml"

# sequence_name -> (Drive/GT floor folder name, YYYY-MM-DD, run_N), e.g.
# "floor_UG1_2025_10_16_run_1" -> ("floor_UG1", "2025-10-16", "run_1"). Matches both the Drive
# folder layout (data/floor_X/YYYY-MM-DD/run_Z/rosbag/...) and the GitHub groundtruth/ filenames
# (floor_X_YYYY-MM-DD_run_Z.txt) documented in the challenge README - confirmed against a live
# Drive folder listing and the repo's groundtruth/ directory (30/30 matched).
_SEQUENCE_RE = re.compile(r"^(floor_[A-Za-z0-9]+)_(\d{4})_(\d{2})_(\d{2})_(run_\d+)$")


def _split_sequence_name(sequence_name: str) -> tuple[str, str, str]:
    m = _SEQUENCE_RE.match(sequence_name)
    if not m:
        raise ValueError(f"Unrecognized hilti2026 sequence name: {sequence_name}")
    floor, yyyy, mm, dd, run = m.groups()
    return floor, f"{yyyy}-{mm}-{dd}", run


def _gt_name(sequence_name: str) -> str:
    floor, date, run = _split_sequence_name(sequence_name)
    return f"{floor}_{date}_{run}.txt"


def _drive_rosbag_relpath(sequence_name: str) -> str:
    floor, date, run = _split_sequence_name(sequence_name)
    return f"data/{floor}/{date}/{run}/rosbag"


class Hilti2026Dataset(DatasetVSLAMLAB):
    """Hilti x Trimble 360 Visual-Inertial SLAM Challenge 2026 dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "hilti2026") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.google_drive_link: str = self.cfg["google_drive_link"]

    def _drive_manifest(self) -> dict[str, str]:
        # Per-sequence Drive folder IDs aren't published for most sequences (only 5 "early
        # release" ones appear as direct links in the README) and the old implementation's
        # hand-typed map for the other 25 didn't match those 5 real links at all. Instead, crawl
        # the one real, official root folder ID once (gdown's skip_download=True lists the whole
        # tree - id, relative path - without downloading any file content) and cache the
        # path->id mapping on disk, so every sequence resolves its own rosbag/metadata file IDs
        # from real, verified data instead of a guessed map.
        manifest_path = self.dataset_path / "drive_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        files = gdown.download_folder(url=self.google_drive_link, skip_download=True, quiet=False)
        manifest = {f.path: f.id for f in files}
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
        return manifest

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        # Calibration (one shared rig for every sequence in the challenge) - straight off GitHub,
        # no gdown/Drive involved, since the challenge also checks these into the repo itself.
        for calibration_path in (CALIBRATION_IMUCAM_PATH, CALIBRATION_IMU_PATH):
            calibration_file = self.dataset_path / calibration_path.rsplit("/", 1)[-1]
            if not calibration_file.exists():
                downloadFile(f"{GITHUB_RAW_ROOT}/{calibration_path}", str(self.dataset_path))

        # Ground truth - also straight off GitHub (all 30 sequences have one, released 2026-06-16;
        # the old implementation only covered 5/30 via hardcoded gdown URLs).
        gt_file = sequence_path / _gt_name(sequence_name)
        if not gt_file.exists():
            downloadFile(f"{GITHUB_RAW_ROOT}/groundtruth/{_gt_name(sequence_name)}", str(sequence_path))

        # Rosbag (Google Drive, resolved via the cached manifest - see _drive_manifest)
        bag_dir = sequence_path / "rosbag"
        if (bag_dir / ROSBAG_NAME).exists():
            return
        bag_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._drive_manifest()
        drive_relpath = _drive_rosbag_relpath(sequence_name)
        for filename in (ROSBAG_NAME, "metadata.yaml"):
            gdown.download(id=manifest[f"{drive_relpath}/{filename}"], output=str(bag_dir / filename), quiet=False)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        bag_dir = sequence_path / "rosbag"

        # Confirmed via a real downloaded bag's metadata.yaml (ros_distro: jazzy,
        # offered_qos_profiles: []) that hilti2026's bags ARE affected by the Jazzy/Humble
        # rosbag2 metadata incompatibility (VSLAM-LAB issue #96) - patch before reading.
        patched_bag_dir = patch_ros2_qos_profiles_metadata(bag_dir, sequence_path / "rosbag_patched")

        # extract-ros2bag-frames has no notion of self.target_resolution - it always writes at
        # the bag's native 1472x1440. Extract into a throwaway raw_path first, then resize into
        # the real rgb_0/rgb_1 below (same rgb_raw-then-resize shape as dataset_hilti2022.py).
        raw_path = sequence_path / "rgb_raw"
        for cam in ("0", "1"):
            run_rosbag_frame_extraction(
                "ros2", patched_bag_dir, raw_path, IMAGE_TOPIC_TEMPLATE.format(cam=cam), cam,
            )

        for cam, final_path in (("0", self.rgb_path(sequence_name)), ("1", sequence_path / "rgb_1")):
            if final_path.exists():
                continue
            final_path.mkdir(parents=True, exist_ok=True)
            for raw_image in sorted((raw_path / f"rgb_{cam}").glob("*.png")):
                with Image.open(raw_image) as img:
                    target_size = compute_scaled_size(img.size, self.target_resolution)
                    img.resize(target_size, Image.Resampling.LANCZOS).save(final_path / raw_image.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # extract-ros2bag-frames already wrote rgb_raw/rgb.csv with paths like "rgb_0/<ts>.png" -
        # identical to the final rgb_0 layout (same filenames, just resized in place above), so
        # the raw csv is valid as-is.
        raw_csv = self.sequence_path(sequence_name) / "rgb_raw" / "rgb.csv"
        shutil.copy2(raw_csv, self.rgb_csv_path(sequence_name))

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        bag_dir = sequence_path / "rosbag_patched"
        imu_csv = self.imu_csv_path(sequence_name)
        if imu_csv.exists():
            return

        inputs = f"--rosbag_path {bag_dir} --sequence_path {sequence_path} --imu_topic {IMU_TOPIC}"
        subprocess.run(f"pixi run -e ros2 extract-ros2bag-imu {inputs}", shell=True, check=True)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        def _load_kalibr_yaml(name: str) -> dict[str, Any]:
            with (self.dataset_path / name).open("r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines and lines[0].lstrip().startswith("%YAML:1.0"):
                lines = lines[1:]
            return yaml.safe_load("%YAML 1.2\n---\n" + "".join(lines))

        cam_data = _load_kalibr_yaml(CALIBRATION_IMUCAM_PATH.rsplit("/", 1)[-1])
        imu_data = _load_kalibr_yaml(CALIBRATION_IMU_PATH.rsplit("/", 1)[-1])
        imu0 = imu_data["imu0"]

        rgb: list[dict[str, Any]] = []
        for cam, cam_name in (("0", "rgb_0"), ("1", "rgb_1")):
            cam_cfg = cam_data[f"cam{cam}"]
            fx, fy, cx, cy = (float(v) for v in cam_cfg["intrinsics"])
            focal_length, principal_point = scale_intrinsics(
                (fx, fy), (cx, cy), tuple(cam_cfg["resolution"]), self.target_resolution
            )
            T_cam_imu = np.array(cam_cfg["T_cam_imu"], dtype=float).reshape(4, 4)

            rgb.append({
                "cam_name": cam_name,
                "cam_type": "rgb",
                "cam_model": "pinhole",
                "focal_length": focal_length,
                "principal_point": principal_point,
                "distortion_type": "equid4",
                "distortion_coefficients": [float(c) for c in cam_cfg["distortion_coeffs"]],
                "fps": float(self.rgb_hz),
                "T_BS": np.linalg.inv(T_cam_imu),
            })

        # Noise-density spec is the real Kalibr-calibrated value for this exact unit (confirmed
        # exact match against config/hilti_openvins/kalibr_imu_chain.yaml on the challenge repo -
        # contrary to VSLAM-LAB issue #97's suspicion, which only checked the unrelated Alphasense
        # Core manual). g is HILTI's own OpenVINS gravity_mag (config/hilti_openvins/
        # estimator_config.yaml), not a generic textbook constant. No published saturation-limit
        # (a_max/g_max) spec exists anywhere for this camera's internal IMU - kept as a generic
        # MEMS-IMU estimate, unverified.
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": float(imu0["gyroscope_noise_density"]),
            "sigma_a_c": float(imu0["accelerometer_noise_density"]),
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": float(imu0["gyroscope_random_walk"]),
            "sigma_aw_c": float(imu0["accelerometer_random_walk"]),
            "g": 9.80766,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(imu0["update_rate"]),
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgb, imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        gt_txt = self.sequence_path(sequence_name) / _gt_name(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        rows = []
        with open(gt_txt, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid groundtruth line {line_num} in {gt_txt}: expected 8 columns, got {len(parts)}"
                    )
                ts_s, tx, ty, tz, qx, qy, qz, qw = parts
                ts_ns = int(round(float(ts_s) * 1e9))
                rows.append([ts_ns, float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])
        write_csv_rows(self.groundtruth_csv_path(sequence_name), header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "rgb_raw", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(sequence_path / "rosbag", ignore_errors=True)
            shutil.rmtree(sequence_path / "rosbag_patched", ignore_errors=True)
            # calibration files are dataset-wide and re-read by every sequence's
            # create_calibration_yaml - never delete them here.
