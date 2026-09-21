"""
Module: VSLAM-LAB - Datasets - dataset_newer_college_stereo_vi.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-15
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import gdown
import numpy as np
import yaml
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import (
    compute_scaled_size, decompressFile, run_rosbag_frame_extraction, scale_intrinsics,
    write_csv_rows,
)

IMAGE_TOPIC_TEMPLATE = "/camera/infra{cam}/image_rect_raw"
# The bags carry no fused RealSense IMU topic (confirmed via a real dynamic_spinning bag: only
# these two split streams exist, at 250 Hz accel / 400 Hz gyro, plus the Ouster's own IMU) - the
# fused 250 Hz stream only exists in short_experiment's raw_format data.csv. create_imu_csv
# extracts both and reproduces the same fused convention.
IMU_TOPICS = {"accel": "/camera/accel/sample", "gyro": "/camera/gyro/sample"}

# Native resolution of the D435i infrared streams, at which the Kalibr camchain below was
# calibrated - confirmed against the camchain's own resolution field and the dataset's platform
# page (https://ori-drs.github.io/newer-college-dataset/stereo-cam/platform-stereo/).
_NATIVE_SIZE = (848, 480)

# Real measured rate of the RealSense fused IMU stream (382114 samples over the 1530 s
# short_experiment raw_format data.csv ~ 250 Hz, matching the documented 250 Hz nominal rate) -
# not the calibration imu.yaml's update_rate: 650, which describes the calibration-time sensor
# configuration, not the released sequences.
_IMU_FPS = 250.0

_GT_HEADER = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
_IMU_HEADER = [
    "ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)",
    "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)",
]


class NewerCollegeStereoViDataset(DatasetVSLAMLAB):
    """Newer College Dataset (stereo-inertial collection) helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "newer-college-stereo-vi") -> None:
        super().__init__(dataset_name)

        # Per-sequence and dataset-wide Google Drive file ids - see the yaml's
        # google_drive_files/calibration_files comments for the upstream folder layout.
        self.google_drive_files: dict[str, dict[str, Any]] = self.cfg["google_drive_files"]
        self.calibration_files: dict[str, str] = self.cfg["calibration_files"]

    def _calibration_path(self, name: str) -> Path:
        return self.dataset_path / "calibration" / f"{name}.yaml"

    def _download_calibration_files(self) -> None:
        # Dataset-wide Kalibr camchain + IMU noise yaml, shared by every sequence (same shape as
        # dataset_hilti2022.py's shared calibration archive).
        for name, file_id in self.calibration_files.items():
            target = self._calibration_path(name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(id=file_id, output=str(target), quiet=False)

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        self._download_calibration_files()

        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        files = self.google_drive_files[sequence_name]
        targets: list[tuple[str, str]] = []
        if "infra1_zips" in files:
            # raw_format sequence (short_experiment): per-camera image zips + EuRoC-style IMU csv.
            for cam in ("infra1", "infra2"):
                for i, file_id in enumerate(files[f"{cam}_zips"], start=1):
                    targets.append((file_id, f"{cam}_{i:02d}.zip"))
            targets.append((files["imu_csv"], "imu_data.csv"))
        else:
            # rosbag-only sequence: full sensor bags (several consecutive parts).
            for i, file_id in enumerate(files["rosbags"], start=0):
                targets.append((file_id, f"rosbag_{i:02d}.bag"))

        targets.append((files["ground_truth"], "registered_poses.csv"))
        if "time_offsets" in files:
            targets.append((files["time_offsets"], "time_offsets.csv"))

        for file_id, filename in targets:
            target = sequence_path / filename
            if not target.exists():
                gdown.download(id=file_id, output=str(target), quiet=False)

        marker.touch()

    def _rosbag_paths(self, sequence_name: str) -> list[Path]:
        files = self.google_drive_files[sequence_name]
        sequence_path = self.sequence_path(sequence_name)
        return [sequence_path / f"rosbag_{i:02d}.bag" for i in range(len(files.get("rosbags", [])))]

    @staticmethod
    def _ts_ns_from_stem(stem: str) -> int:
        # raw_format zip images are named "infra<N>_<sec>_<nsec>.png" (confirmed on a real
        # downloaded infra1 zip: flat archive, e.g. "infra1_1583836591_152386717.png");
        # rosbag-extracted frames are named "<ts_ns>.png". Normalize both to integer nanoseconds.
        if "_" in stem:
            parts = stem.split("_")
            return int(parts[-2]) * 10**9 + int(parts[-1])
        return int(stem)

    def _resize_into(self, image_files: list[Path], final_path: Path) -> None:
        # Write into a tmp dir, then atomically rename into place, so a crash partway through
        # can't leave a plausible-looking half-filled rgb_0/rgb_1.
        tmp_path = final_path.with_name(final_path.name + ".tmp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True)
        for image_file in image_files:
            target_name = f"{self._ts_ns_from_stem(image_file.stem)}.png"
            if self.target_resolution is None:
                shutil.copy2(image_file, tmp_path / target_name)
            else:
                with Image.open(image_file) as img:
                    target_size = compute_scaled_size(img.size, self.target_resolution)
                    img.resize(target_size, Image.Resampling.LANCZOS).save(tmp_path / target_name)
        tmp_path.rename(final_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        files = self.google_drive_files[sequence_name]

        if "infra1_zips" in files:
            # Decompress each per-camera zip part into one shared raw folder per camera.
            for cam_idx, cam in enumerate(("infra1", "infra2")):
                final_path = sequence_path / f"rgb_{cam_idx}"
                if final_path.exists():
                    continue
                raw_path = sequence_path / f"{cam}_raw"
                for i in range(1, len(files[f"{cam}_zips"]) + 1):
                    part_marker = raw_path / f".extract_complete_{i:02d}"
                    if part_marker.exists():
                        continue
                    decompressFile(sequence_path / f"{cam}_{i:02d}.zip", raw_path)
                    part_marker.touch()
                self._resize_into(sorted(raw_path.rglob("*.png")), final_path)
            return

        # rosbag-only sequence: extract each consecutive bag part into its own sub-folder (the
        # extraction script writes one rgb.csv per output folder and refuses to append to an
        # existing camera column), then merge the parts.
        raw_path = sequence_path / "rgb_raw"
        for cam_idx, cam in enumerate(("1", "2")):
            final_path = sequence_path / f"rgb_{cam_idx}"
            if final_path.exists():
                continue
            image_topic = IMAGE_TOPIC_TEMPLATE.format(cam=cam)
            part_paths = []
            for i, rosbag_path in enumerate(self._rosbag_paths(sequence_name)):
                part_path = raw_path / f"part_{i:02d}"
                run_rosbag_frame_extraction("ros1", rosbag_path, part_path, image_topic, cam_idx)
                part_paths.append(part_path)
            image_files = sorted(
                (f for part in part_paths for f in (part / f"rgb_{cam_idx}").glob("*.png")),
                key=lambda f: f.name,
            )
            self._resize_into(image_files, final_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # The two infrared streams are triggered together by the D435i (identical frame counts and
        # per-frame timestamps in the raw_format zips) - sort both folders and zip by index.
        rgb_0_files = sorted(self.rgb_path(sequence_name).glob("*.png"))
        rgb_1_files = sorted((self.sequence_path(sequence_name) / "rgb_1").glob("*.png"))

        n = min(len(rgb_0_files), len(rgb_1_files))
        rows = []
        for rgb_0_file, rgb_1_file in zip(rgb_0_files[:n], rgb_1_files[:n]):
            rows.append([
                int(rgb_0_file.stem), f"rgb_0/{rgb_0_file.name}",
                int(rgb_1_file.stem), f"rgb_1/{rgb_1_file.name}",
            ])

        write_csv_rows(
            self.rgb_csv_path(sequence_name),
            ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"],
            rows,
        )

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        files = self.google_drive_files[sequence_name]

        if "infra1_zips" in files:
            # raw_format IMU csv: "#counter,sec,nsec,wx,wy,wz,ax,ay,az" (comma-separated).
            rows = []
            with (sequence_path / "imu_data.csv").open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    _, sec, nsec, wx, wy, wz, ax, ay, az = line.split(",")
                    ts_ns = int(sec) * 10**9 + int(nsec)
                    rows.append([ts_ns, float(wx), float(wy), float(wz), float(ax), float(ay), float(az)])
            write_csv_rows(self.imu_csv_path(sequence_name), _IMU_HEADER, rows)
            return

        # rosbag-only sequence: extract the split accel/gyro streams from each bag part (separate
        # sub-folders - each extraction call overwrites its output csv), then fuse them the same
        # way the dataset's own raw_format data.csv does: one row per accel sample (250 Hz), gyro
        # interpolated onto the accel timestamps.
        streams: dict[str, list[list[float]]] = {"accel": [], "gyro": []}
        for i, rosbag_path in enumerate(self._rosbag_paths(sequence_name)):
            part_path = sequence_path / "imu_raw" / f"part_{i:02d}"
            for stream, imu_topic in IMU_TOPICS.items():
                part_csv = part_path / f"imu_{stream}.csv"
                if not part_csv.exists():
                    part_path.mkdir(parents=True, exist_ok=True)
                    # No run_rosbag_imu_extraction-style utility exists yet (unlike frame
                    # extraction) - same hand-rolled call as dataset_hilti2022.py. The extraction
                    # script writes its csv via tmp-then-replace, so the exists() guard above is
                    # crash-safe.
                    inputs = (
                        f"--rosbag_path {rosbag_path} --sequence_path {part_path}"
                        f" --imu_topic {imu_topic} --imu_name {stream}"
                    )
                    subprocess.run(f"pixi run -e ros1 extract-rosbag-imu {inputs}", shell=True, check=True)
                with part_csv.open("r", encoding="utf-8") as fin:
                    next(fin)  # header
                    for line in fin:
                        parts = line.strip().split(",")
                        if len(parts) == 7:
                            streams[stream].append([int(parts[0])] + [float(v) for v in parts[1:]])

        accel = np.array(sorted(streams["accel"], key=lambda r: r[0]), dtype=float)
        gyro = np.array(sorted(streams["gyro"], key=lambda r: r[0]), dtype=float)
        # Keep only accel samples inside the gyro time range - np.interp would otherwise clamp to
        # the gyro endpoints and fabricate constant readings there.
        mask = (accel[:, 0] >= gyro[0, 0]) & (accel[:, 0] <= gyro[-1, 0])
        accel = accel[mask]
        rows = []
        for row in accel:
            ts_ns = int(row[0])
            w = [float(np.interp(row[0], gyro[:, 0], gyro[:, axis])) for axis in (1, 2, 3)]
            a = [float(v) for v in row[4:7]]
            rows.append([ts_ns] + w + a)
        write_csv_rows(self.imu_csv_path(sequence_name), _IMU_HEADER, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        with self._calibration_path("camchain_imucam").open("r", encoding="utf-8") as f:
            camchain = yaml.safe_load(f)
        with self._calibration_path("imu_noise").open("r", encoding="utf-8") as f:
            imu_noise = yaml.safe_load(f)

        rgb: list[dict[str, Any]] = []
        for cam_key, cam_name in (("cam0", "rgb_0"), ("cam1", "rgb_1")):
            cam = camchain[cam_key]
            # Kalibr intrinsics are at the native 848x480 - rescale to match the resized rgb_0/
            # rgb_1 create_rgb_folder wrote (VSLAM-LAB issue #99).
            focal_length, principal_point = scale_intrinsics(
                [float(v) for v in cam["intrinsics"][0:2]],
                [float(v) for v in cam["intrinsics"][2:4]],
                tuple(cam["resolution"]),
                self.target_resolution,
            )
            T_cam_imu = np.array(cam["T_cam_imu"], dtype=float).reshape(4, 4)
            rgb.append({
                "cam_name": cam_name,
                "cam_type": "gray",
                "cam_model": "pinhole",
                "focal_length": focal_length,
                "principal_point": principal_point,
                "distortion_type": "radtan4",
                "distortion_coefficients": [float(v) for v in cam["distortion_coeffs"]],
                "fps": float(self.rgb_hz),
                "T_BS": np.linalg.inv(T_cam_imu),
            })

        # Noise densities from the dataset's own published Kalibr imu.yaml; saturation limits are
        # the BMI055 spec (+-16 g, +-2000 deg/s), not covered by that file.
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 156.9,
            "g_max": 34.9,
            "sigma_g_c": float(imu_noise["gyroscope_noise_density"]),
            "sigma_a_c": float(imu_noise["accelerometer_noise_density"]),
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": float(imu_noise["gyroscope_random_walk"]),
            "sigma_aw_c": float(imu_noise["accelerometer_random_walk"]),
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": _IMU_FPS,
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgb, imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # registered_poses.csv: "#sec,nsec,x,y,z,qx,qy,qz,qw" - 10 Hz poses of the left camera
        # frame (RS_C1, the rig's base frame per the dataset paper, Sec. III), from ICP
        # registration of each Ouster scan against the BLK360 prior map.
        #
        # Known clock caveat (timestamps written as published, deliberately uncorrected): the
        # camera/IMU clock leads the groundtruth timestamps by a per-sequence amount, measured by
        # cross-correlating each sequence's gyro magnitude against the GT angular rate -
        # ~59 ms on dynamic_spinning (corr 0.99, matching its time_offsets.csv of 53-57 ms) but
        # ~130 ms on short_experiment (corr 0.82, ~2x its time_offsets.csv of 56-79 ms; March vs
        # July 2020 campaigns appear to have handled the lidar-vs-camera clock differently when
        # generating registered_poses.csv). No documented upstream convention exists to correct
        # this cleanly, so poses ship as published - the same choice the NCD literature evaluates
        # against. See the GitHub issue filed with this integration for the full analysis.
        src = self.sequence_path(sequence_name) / "registered_poses.csv"

        rows = []
        with src.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sec, nsec, tx, ty, tz, qx, qy, qz, qw = line.split(",")
                ts_ns = int(sec) * 10**9 + int(nsec)
                rows.append([ts_ns, float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])

        write_csv_rows(self.groundtruth_csv_path(sequence_name), _GT_HEADER, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        files = self.google_drive_files[sequence_name]

        if BENCHMARK_RETENTION != Retention.FULL:
            for raw_name in ("infra1_raw", "infra2_raw", "rgb_raw", "imu_raw"):
                shutil.rmtree(sequence_path / raw_name, ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            for cam in ("infra1", "infra2"):
                for i in range(1, len(files.get(f"{cam}_zips", [])) + 1):
                    (sequence_path / f"{cam}_{i:02d}.zip").unlink(missing_ok=True)
            for rosbag_path in self._rosbag_paths(sequence_name):
                rosbag_path.unlink(missing_ok=True)
            for filename in ("imu_data.csv", "registered_poses.csv", "time_offsets.csv"):
                (sequence_path / filename).unlink(missing_ok=True)
            # dataset_path/"calibration" is dataset-wide and re-read by every sequence's
            # create_calibration_yaml - never delete it.
