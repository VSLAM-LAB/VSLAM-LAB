"""
Module: VSLAM-LAB - Datasets - dataset_minimal.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-18
- License: GPLv3 License
"""

from __future__ import annotations

import shutil

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import (
    compute_scaled_size, hf_token, run_rosbag_frame_extraction, scale_intrinsics, write_csv_rows,
)

COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"

# Global RealSense D435i color intrinsics, identical across both recording days (verified against
# /camera/color/camera_info in the 10-08 and 10-14 session bags; plumb_bob with all-zero
# distortion). The aligned depth stream shares these intrinsics by construction.
NATIVE_RESOLUTION = (1280, 720)
FOCAL_LENGTH = [908.479248046875, 907.8419189453125]
PRINCIPAL_POINT = [650.88916015625, 364.6926574707031]


class MinimalDataset(DatasetVSLAMLAB):
    """Minimal Texture Dataset for RGB-D SLAM helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "minimal") -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id: str = self.cfg["hf_repo_id"]

        # Get depth factor
        self.depth_factor: float = self.cfg["depth_factor"]

    def _remote_names(self, sequence_name: str) -> tuple[str, str]:
        """The HF repo (subfolder, base_name) for a sequence: '1008_triangle_01' -> ('10-08',
        'triangle_01')."""
        prefix, base_name = sequence_name.split("_", 1)
        return f"{prefix[:2]}-{prefix[2:]}", base_name

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return
        sequence_path.mkdir(parents=True, exist_ok=True)

        day, base_name = self._remote_names(sequence_name)
        for remote_file, local_name in (
            (f"{day}/{base_name}.bag", f"{sequence_name}.bag"),
            (f"{day}/{base_name}_groundtruth.txt", f"{sequence_name}_groundtruth.txt"),
        ):
            downloaded = hf_hub_download(repo_id=self.hf_repo_id, filename=remote_file,
                                         repo_type="dataset", token=hf_token(),
                                         local_dir=sequence_path)
            shutil.move(downloaded, sequence_path / local_name)
        shutil.rmtree(sequence_path / day, ignore_errors=True)
        shutil.rmtree(sequence_path / ".cache", ignore_errors=True)
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_path = sequence_path / f"{sequence_name}.bag"
        raw_path = sequence_path / "rgb_raw"

        # extract-rosbag-frames writes the color topic to rgb_0/ and the aligned-depth topic to
        # rgb_1/ (the script only knows rgb_{cam} names) - rgb_1/ becomes depth_0/ below.
        run_rosbag_frame_extraction("ros1", rosbag_path, raw_path, COLOR_TOPIC, 0)
        run_rosbag_frame_extraction("ros1", rosbag_path, raw_path, DEPTH_TOPIC, 1)

        rgb_path = self.rgb_path(sequence_name)
        if not rgb_path.exists():
            rgb_path.mkdir(parents=True, exist_ok=True)
            for raw_image in sorted((raw_path / "rgb_0").glob("*.png")):
                with Image.open(raw_image) as img:
                    # extract-rosbag-frames decodes rgb8 messages with 'passthrough' and saves via
                    # cv2.imwrite (which assumes BGR), so the raw PNGs have R/B swapped - undo it.
                    rgb = Image.fromarray(np.asarray(img)[:, :, ::-1])
                    target_size = compute_scaled_size(rgb.size, self.target_resolution)
                    rgb.resize(target_size, Image.Resampling.LANCZOS).save(rgb_path / raw_image.name)

        depth_path = self.depth_path(sequence_name)
        if not depth_path.exists():
            depth_path.mkdir(parents=True, exist_ok=True)
            for raw_image in sorted((raw_path / "rgb_1").glob("*.png")):
                with Image.open(raw_image) as img:
                    target_size = compute_scaled_size(img.size, self.target_resolution)
                    # Nearest-neighbor only for depth - interpolation would blend depth values
                    # across object boundaries and corrupt the metric data.
                    img.resize(target_size, Image.Resampling.NEAREST).save(depth_path / raw_image.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        raw_path = sequence_path / "rgb_raw"

        # Color and aligned depth are two topics of the same hardware capture (identical header
        # stamps, one message each per frame), so the sorted streams correspond 1:1 by index -
        # zip() also drops a trailing unpaired frame if one stream is one message longer.
        rgb_names = sorted(p.name for p in (raw_path / "rgb_0").glob("*.png"))
        depth_names = sorted(p.name for p in (raw_path / "rgb_1").glob("*.png"))

        rgb_dir = self.rgb_path(sequence_name).name
        depth_dir = self.depth_path(sequence_name).name
        rows = []
        for rgb_name, depth_name in zip(rgb_names, depth_names):
            rows.append([int(rgb_name.removesuffix(".png")), f"{rgb_dir}/{rgb_name}",
                         int(depth_name.removesuffix(".png")), f"{depth_dir}/{depth_name}"])

        write_csv_rows(self.rgb_csv_path(sequence_name),
                       ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        focal_length, principal_point = scale_intrinsics(
            FOCAL_LENGTH, PRINCIPAL_POINT, NATIVE_RESOLUTION, self.target_resolution,
        )

        rgbd0 = {
            "cam_name": "rgb_0",
            "cam_type": "rgb+depth",
            "depth_name": "depth_0",
            "cam_model": "pinhole",
            "focal_length": focal_length,
            "principal_point": principal_point,
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_tum = sequence_path / f"{sequence_name}_groundtruth.txt"

        rows = []
        with open(groundtruth_tum, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid groundtruth line {line_num} in {groundtruth_tum}: "
                        f"expected 8 columns, got {len(parts)}"
                    )
                ts_s, tx, ty, tz, qx, qy, qz, qw = parts
                ts_ns = int(round(float(ts_s) * 1e9))
                rows.append([ts_ns, float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])

        write_csv_rows(self.groundtruth_csv_path(sequence_name),
                       ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "rgb_raw", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (sequence_path / f"{sequence_name}.bag").unlink(missing_ok=True)
