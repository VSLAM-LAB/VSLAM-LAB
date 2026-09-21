"""
Module: VSLAM-LAB - Datasets - dataset_pupil_labs.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-28
- License: GPLv3 License
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import compute_scaled_size, make_printers, scale_intrinsics, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# A Pupil Cloud "Timeseries Data + Scene Video" export (raw-data-exporter, format version 2 -
# https://docs.pupil-labs.com/neon/data-collection/data-format/) is one folder per recording,
# named <recording name>-<recording id prefix>, holding the files below plus the scene video
# (<section id>_<start s>-<end s>.mp4, H.264 + audio) and gaze/fixation/blink/event CSVs this
# dataset doesn't use. The recording folder is the sequence folder: its files sit directly in
# self.sequence_path(sequence_name), next to the rgb_0/ etc. that the hooks below derive.
_EXPORT_DIR_NAME: Final = "Timeseries Data + Scene Video"
_RAW_FILES: Final[tuple[str, ...]] = ("info.json", "scene_camera.json", "world_timestamps.csv", "imu.csv")

# imu.csv units: gyro in deg/s, acceleration in g ("1 g = 9.80665 m/s^2" per the data-format docs).
_G: Final = 9.80665

# Pixel-value std below which a decoded frame is a uniform placeholder, not a camera image.
_PLACEHOLDER_STD: Final = 0.5

# Scene camera -> IMU extrinsics (T_BS of rgb_0 with the IMU as body frame).
#
# Pupil Labs documents (https://docs.pupil-labs.com/alpha-lab/imu-transformations/) the scene
# camera as rotated -102 deg about x with respect to their "IMU coordinate system" (x right,
# y forward, z up), with the camera origin at (0.0, -1.3, -6.62) mm in that frame - i.e. 1.3 mm
# behind and 6.62 mm below the IMU, the camera looking 12 deg down. The raw gyro/accel columns in
# imu.csv, however, are NOT expressed in that documented frame: fitting the accelerometer's
# gravity direction against the export's own drift-free roll/pitch columns on a real recording
# shows the raw axes are that frame rotated by exactly +90 deg about x (0.4 deg residual), i.e.
# x right, y down, z forward, with a right-handed gyro in the same axes. Composing both:
# R_imu_cam = Rx(+90) * Rx(-102) = Rx(-12 deg), t_imu_cam = Rx(+90) * (0, -1.3, -6.62) mm =
# (0, +6.62, -1.3) mm.
_IMU_CAM_TILT_DEG: Final = -12.0
_IMU_CAM_TRANSLATION_M: Final[tuple[float, float, float]] = (0.0, 0.00662, -0.0013)


def _rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _t_imu_cam() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rot_x(_IMU_CAM_TILT_DEG)
    T[:3, 3] = _IMU_CAM_TRANSLATION_M
    return T


class PupilLabsDataset(DatasetVSLAMLAB):
    """Pupil Labs Neon eye-tracking glasses dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "pupil-labs") -> None:
        super().__init__(dataset_name)

        # All sequences are local (scalar in the yaml) - user-recorded Pupil Cloud exports.
        self.sequence_location = self.cfg["sequence_location"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        if self._has_raw_export(sequence_name):
            return

        # Adopt a recording folder still nested inside an extracted Pupil Cloud export directory
        # (VSLAM-LAB-Benchmark/PUPIL-LABS/<export dir>/<sequence_name>/) by moving it up to be
        # the sequence folder itself.
        if not sequence_path.exists():
            nested = sorted(p.parent for p in self.dataset_path.glob(f"*/{sequence_name}/info.json"))
            if nested:
                print_info(f"Adopting recording folder {nested[0]} -> {sequence_path}")
                shutil.move(str(nested[0]), str(sequence_path))
                if self._has_raw_export(sequence_name):
                    return

        missing = [name for name in _RAW_FILES if not (sequence_path / name).is_file()]
        if self._find_video(sequence_name) is None:
            missing.append("<section id>_<start>-<end>.mp4")
        print_info(
            f"Sequence '{sequence_name}' is marked as 'local'. Please place its Pupil Cloud "
            f"'{_EXPORT_DIR_NAME}' recording folder at {sequence_path} (missing: {', '.join(missing)})."
        )

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return
        if not self._has_raw_export(sequence_name):
            raise FileNotFoundError(
                f"Pupil Cloud export for '{sequence_name}' not found at {self.sequence_path(sequence_name)} "
                f"(sequence marked as 'local')."
            )

        # Every frame is remapped onto the distortion-free pinhole camera create_calibration_yaml
        # writes (rational 8-coefficient source distortion -> pinhole), directly at the resized
        # output size: initUndistortRectifyMap takes the output camera matrix and size
        # independently of the source's, so undistortion and target_resolution scaling are one
        # remap instead of a remap plus a LANCZOS resize.
        K, D = self._scene_camera(sequence_name)
        (fx, fy), (cx, cy), (out_w, out_h) = self._pinhole_intrinsics(sequence_name)
        K_out = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, K_out, (out_w, out_h), cv2.CV_32FC1)

        video_path = self._find_video(sequence_name)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Decode into a staging folder and rename at the end, so a crash mid-way never leaves a
        # partial rgb_0/ that the exists() check above would take for a finished one.
        staging = rgb_path.with_name(f".{rgb_path.name}_tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        # Filenames carry the decoded-frame index (row of world_timestamps.csv), not a running
        # count, so create_rgb_csv can pair frames with their timestamps even with gaps: Pupil
        # Cloud pads the video from recording.begin until the scene camera delivers its first
        # frame with uniform placeholder frames (a run of gray then one black, ~1.75 s in the
        # first recording), which are skipped here - a real camera frame always has sensor noise,
        # so zero variance can only be a placeholder.
        idx = 0
        skipped = 0
        pbar = tqdm(total=total_frames, desc="    undistorting frames", unit="frame")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.std() < _PLACEHOLDER_STD:
                skipped += 1
            else:
                undistorted = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(staging / f"{idx:05d}.png"), undistorted)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        if skipped:
            print_info(f"{sequence_name}: skipped {skipped} uniform placeholder frames of {idx}")
        staging.rename(rgb_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        # world_timestamps.csv has one UTC-ns row per decoded video frame, in frame order (the
        # mp4 is variable-frame-rate, so these real timestamps are the only correct source - never
        # rgb_hz). Paired with rgb_0/ by the decoded-frame index in each filename (see
        # create_rgb_folder - placeholder frames were skipped, so a plain zip by order would
        # misalign every frame after the first gap).
        timestamps = pd.read_csv(self.sequence_path(sequence_name) / "world_timestamps.csv")["timestamp [ns]"]
        timestamps = timestamps.astype(np.int64).tolist()
        rgb_path = self.rgb_path(sequence_name)
        rgb_files = sorted(p.name for p in rgb_path.iterdir() if p.suffix.lower() == ".png")
        rows = []
        for name in rgb_files:
            idx = int(Path(name).stem)
            if idx >= len(timestamps):
                print_warning(f"{sequence_name}: frame {name} has no row in world_timestamps.csv ({len(timestamps)} rows) - dropped")
                continue
            rows.append([int(timestamps[idx]), f"{rgb_path.name}/{name}"])

        header = ["ts_rgb_0 (ns)", "path_rgb_0"]
        write_csv_rows(rgb_csv, header, rows)

    def create_imu_csv(self, sequence_name: str) -> None:
        imu_csv = self.imu_csv_path(sequence_name)
        if imu_csv.exists():
            return

        df = pd.read_csv(self.sequence_path(sequence_name) / "imu.csv")
        df = df.sort_values("timestamp [ns]").drop_duplicates("timestamp [ns]")
        out = pd.DataFrame()
        out["ts (ns)"] = df["timestamp [ns]"].astype(np.int64)
        for axis in "xyz":
            out[f"w{axis} (rad s^-1)"] = np.deg2rad(df[f"gyro {axis} [deg/s]"].astype(float))
        for axis in "xyz":
            out[f"a{axis} (m s^-2)"] = df[f"acceleration {axis} [g]"].astype(float) * _G

        # .astype(object) before .values - the int64 ns timestamp column would otherwise be
        # upcast to float64 alongside the other columns, losing precision past 2^53.
        write_csv_rows(imu_csv, list(out.columns), out.astype(object).values.tolist())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        (fx, fy), (cx, cy), _ = self._pinhole_intrinsics(sequence_name)
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [float(fx), float(fy)],
            "principal_point": [float(cx), float(cy)],
            "fps": float(self.rgb_hz),
            "T_BS": _t_imu_cam(),
        }

        # Neon's IMU is a TDK InvenSense ICM-20948 (per Pupil Labs' data-streams docs). Noise
        # densities and bias priors are the chip datasheet's typical figures (DS-000189 v1.3,
        # tables 1-2): gyro noise 0.015 dps/sqrt(Hz), accel noise 230 ug/sqrt(Hz), initial
        # zero-rate output +-5 dps, board-level zero-g output +-50 mg. a_max/g_max are the chip's
        # widest programmable full-scale ranges (+-16 g, +-2000 dps) - Pupil Labs doesn't
        # document which range the Companion app configures. Bias random-walk densities aren't in
        # the datasheet - same generic defaults as dataset_aria_digital_twin.py/dataset_madmax.py.
        # The sample rate is measured from this recording's imu.csv (the docs' nominal 110 Hz
        # doesn't match real exports, ~200 Hz here).
        imu0: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": float(16.0 * _G),
            "g_max": float(np.deg2rad(2000.0)),
            "sigma_g_c": float(np.deg2rad(0.015)),
            "sigma_a_c": float(230e-6 * _G),
            "sigma_bg": float(np.deg2rad(5.0)),
            "sigma_ba": float(50e-3 * _G),
            "sigma_gw_c": 20.0e-5,
            "sigma_aw_c": 20.0e-3,
            "g": _G,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": self._imu_rate_hz(sequence_name),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0], imu=[imu0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # Neon ships no pose groundtruth (only a fused orientation estimate) - header-only file.
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        write_csv_rows(self.groundtruth_csv_path(sequence_name), header, [])

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberate no-op at every retention tier, including MINIMAL: the export files are the
        # user's own recording with no re-download path (same rationale as
        # dataset_malaysia_jul2026.py/dataset_aria_digital_twin.py), and imu.csv carries columns
        # (roll/pitch estimates) the standardized layout doesn't keep.
        return

    def _has_raw_export(self, sequence_name: str) -> bool:
        sequence_path = self.sequence_path(sequence_name)
        return all((sequence_path / name).is_file() for name in _RAW_FILES) and self._find_video(sequence_name) is not None

    def _find_video(self, sequence_name: str) -> Path | None:
        sequence_path = self.sequence_path(sequence_name)
        if not sequence_path.is_dir():
            return None
        return next(iter(sorted(sequence_path.glob("*.mp4"))), None)

    def _scene_camera(self, sequence_name: str) -> tuple[np.ndarray, np.ndarray]:
        """(K 3x3, dist_coefs 8) of this recording's scene camera, from scene_camera.json -
        factory intrinsics of the specific Neon module, at the recording's video resolution."""
        with open(self.sequence_path(sequence_name) / "scene_camera.json", "r", encoding="utf-8") as f:
            cal = json.load(f)
        K = np.array(cal["camera_matrix"], dtype=np.float64)
        D = np.array(cal["dist_coefs"], dtype=np.float64).reshape(-1)
        return K, D

    def _native_size(self, sequence_name: str) -> tuple[int, int]:
        cap = cv2.VideoCapture(str(self._find_video(sequence_name)))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        return size

    def _pinhole_intrinsics(self, sequence_name: str) -> tuple[list[float], list[float], tuple[int, int]]:
        """(focal_length, principal_point, (width, height)) of the distortion-free pinhole camera
        rgb_0/ is rendered with - getOptimalNewCameraMatrix at alpha=0 (no invalid border pixels)
        on the native video size, then scaled to target_resolution. Recomputed here from the raw
        files by both create_rgb_folder and create_calibration_yaml so the two can never diverge."""
        K, D = self._scene_camera(sequence_name)
        native_size = self._native_size(sequence_name)
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, native_size, 0, native_size)
        focal_length, principal_point = scale_intrinsics(
            (K_new[0, 0], K_new[1, 1]), (K_new[0, 2], K_new[1, 2]), native_size, self.target_resolution
        )
        return focal_length, principal_point, compute_scaled_size(native_size, self.target_resolution)

    def _imu_rate_hz(self, sequence_name: str) -> float:
        ts = pd.read_csv(self.sequence_path(sequence_name) / "imu.csv")["timestamp [ns]"].astype(np.int64).to_numpy()
        ts = np.unique(ts)
        return float(round(1e9 / float(np.median(np.diff(ts))), 1))
