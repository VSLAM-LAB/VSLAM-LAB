"""
Module: VSLAM-LAB - Datasets - dataset_nsavp.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-31
- License: GPLv3 License
"""

from __future__ import annotations

from typing import Any, Final
from urllib.parse import urljoin

import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, downloadFile, scale_intrinsics, write_csv_rows

# Two mono cameras are triggered off the same hardware line (readme: TriggerMode On, shared
# trigger source), but frame counts can still differ by a frame or two at sequence boundaries -
# pair by nearest timestamp rather than assuming equal-length arrays and zipping by index.
STEREO_MATCH_TOLERANCE_NS: Final = 25_000_000  # ~half a frame period at 20.14 Hz


class NsavpDataset(DatasetVSLAMLAB):
    """Novel Sensors for Autonomous Vehicle Perception (NSAVP) dataset helper for VSLAM-LAB benchmark."""

    # Deep Blue Data's calibration/measured-extrinsics files carry a per-sequence session index
    # (e.g. "C1") baked into the filename that isn't derivable from sequence_name - hand-maintained
    # per sequence, same spirit as ROVER's DATES table.
    CALIBRATION_PREFIX: Final = {"R0_FA0": "C1"}

    def __init__(self, dataset_name: str = "nsavp") -> None:
        super().__init__(dataset_name)

        # Dict keyed by full sequence_name - see dataset_nsavp.yaml's url_download_root comment
        # for why this can't be a single templated root URL.
        self.url_download_root: dict[str, str] = self.cfg["url_download_root"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        sequence_path.mkdir(parents=True, exist_ok=True)
        root = self.url_download_root[sequence_name]
        calibration_prefix = self.CALIBRATION_PREFIX[sequence_name]

        filenames = [
            f"{sequence_name}_mono_left.h5",
            f"{sequence_name}_mono_right.h5",
            f"{sequence_name}_applanix.h5",
            f"{sequence_name}_{calibration_prefix}_calibration_results.yaml",
        ]
        for filename in filenames:
            if not (sequence_path / filename).exists():
                downloadFile(urljoin(root, filename), str(sequence_path))

        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path_0 = self.rgb_path(sequence_name)
        rgb_path_1 = sequence_path / "rgb_1"
        if rgb_path_0.exists() and rgb_path_1.exists():
            return

        rgb_path_0.mkdir(parents=True, exist_ok=True)
        rgb_path_1.mkdir(parents=True, exist_ok=True)

        with h5py.File(sequence_path / f"{sequence_name}_mono_left.h5", "r") as f_left, \
             h5py.File(sequence_path / f"{sequence_name}_mono_right.h5", "r") as f_right:
            # Cast from h5's native uint64 - merge_asof's tolerance arithmetic expects a signed
            # dtype, and ns-since-epoch values comfortably fit within int64 anyway.
            ts_left = f_left["/image_raw/timestamps"][:].astype(np.int64)
            ts_right = f_right["/image_raw/timestamps"][:].astype(np.int64)

            # Nearest-timestamp pairing (see STEREO_MATCH_TOLERANCE_NS above), then encode each
            # kept frame's own timestamp directly into its filename - create_rgb_csv re-derives
            # ts_rgb_0/ts_rgb_1 straight from the written filenames instead of re-opening these
            # (large, slow-to-stream) h5 files and redoing this pairing a second time.
            pairs = pd.merge_asof(
                pd.DataFrame({"ts_left": ts_left, "idx_left": np.arange(len(ts_left))}).sort_values("ts_left"),
                pd.DataFrame({"ts_right": ts_right, "idx_right": np.arange(len(ts_right))}).sort_values("ts_right"),
                left_on="ts_left", right_on="ts_right", direction="nearest",
                tolerance=STEREO_MATCH_TOLERANCE_NS,
            ).dropna(subset=["ts_right"])

            images_left = f_left["/image_raw/images"]
            images_right = f_right["/image_raw/images"]
            target_size = None
            for idx_left, idx_right, ts_l, ts_r in tqdm(
                zip(pairs["idx_left"], pairs["idx_right"].astype(int), pairs["ts_left"].astype(int), pairs["ts_right"].astype(int)),
                total=len(pairs), desc="    converting stereo mono frames",
            ):
                for image_dataset, idx, ts, rgb_path in (
                    (images_left, idx_left, ts_l, rgb_path_0),
                    (images_right, idx_right, ts_r, rgb_path_1),
                ):
                    img = Image.fromarray(image_dataset[idx], mode="L")
                    if target_size is None:
                        target_size = compute_scaled_size(img.size, self.target_resolution)
                    img.resize(target_size, Image.Resampling.LANCZOS).save(rgb_path / f"{ts}.png")

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_files_0 = sorted((self.rgb_path(sequence_name)).glob("*.png"))
        rgb_files_1 = sorted((sequence_path / "rgb_1").glob("*.png"))

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]
        rows = [
            [int(f0.stem), f"rgb_0/{f0.name}", int(f1.stem), f"rgb_1/{f1.name}"]
            for f0, f1 in zip(rgb_files_0, rgb_files_1, strict=True)
        ]
        write_csv_rows(self.rgb_csv_path(sequence_name), header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        calibration_prefix = self.CALIBRATION_PREFIX[sequence_name]
        calibration_file = sequence_path / f"{sequence_name}_{calibration_prefix}_calibration_results.yaml"
        with open(calibration_file, "r") as f:
            data = yaml.safe_load(f)

        cam_left, cam_right = data["cam2"], data["cam3"]  # mono_left, mono_right (see readme's stereo camera table)

        # Calibration intrinsics are computed at each camera's raw (pre-resize) resolution -
        # rescale to match the actual images written into rgb_0/rgb_1 (see create_rgb_folder's
        # self.target_resolution downscale).
        def _scaled_intrinsics(cam: dict) -> tuple[list[float], list[float]]:
            fx, fy, cx, cy = (float(v) for v in cam["intrinsics"])
            return scale_intrinsics((fx, fy), (cx, cy), tuple(cam["resolution"]), self.target_resolution)

        focal_length_0, principal_point_0 = _scaled_intrinsics(cam_left)
        focal_length_1, principal_point_1 = _scaled_intrinsics(cam_right)

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": focal_length_0,
            "principal_point": principal_point_0,
            "distortion_type": "radtan4",
            "distortion_coefficients": [float(v) for v in cam_left["distortion_coeffs"]],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": focal_length_1,
            "principal_point": principal_point_1,
            "distortion_type": "radtan4",
            "distortion_coefficients": [float(v) for v in cam_right["distortion_coeffs"]],
            "fps": float(self.rgb_hz),
            # cam3's T_cn_cnm1 (Kalibr camchain convention) maps a point from cam2 (mono_left)
            # into cam3 (mono_right) - i.e. the mono_left-to-mono_right extrinsic. T_BS wants the
            # inverse: mono_right's pose expressed in the mono_left (body/rgb_0) frame.
            "T_BS": np.linalg.inv(np.array(cam_right["T_cn_cnm1"], dtype=float)),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        applanix_file = sequence_path / f"{sequence_name}_applanix.h5"
        rows = []
        if applanix_file.exists():
            with h5py.File(applanix_file, "r") as f:
                positions = f["/pose_base_link/positions"][:]
                quaternions = f["/pose_base_link/quaternions"][:]
                timestamps = f["/pose_base_link/timestamps"][:]

            # Raw positions are ECEF meters (~6.3e6 magnitude) - translate to the first pose so
            # downstream tooling (and evo's Sim3 alignment against the estimate) work with
            # reasonably-scaled local coordinates. Orientation is left as base_link-wrt-ECEF; a
            # single global alignment absorbs the resulting constant rotation offset the same way
            # every other VSLAM-LAB dataset relies on it to absorb a body-vs-camera-frame offset.
            local_positions = positions - positions[0]
            for ts, pos, quat in zip(timestamps, local_positions, quaternions):
                rows.append([int(ts), float(pos[0]), float(pos[1]), float(pos[2]),
                             float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])

        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        calibration_prefix = self.CALIBRATION_PREFIX[sequence_name]

        if BENCHMARK_RETENTION != Retention.FULL:
            (sequence_path / f"{sequence_name}_{calibration_prefix}_calibration_results.yaml").unlink(missing_ok=True)
            (sequence_path / f"{sequence_name}_applanix.h5").unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (sequence_path / f"{sequence_name}_mono_left.h5").unlink(missing_ok=True)
            (sequence_path / f"{sequence_name}_mono_right.h5").unlink(missing_ok=True)
