"""
Module: VSLAM-LAB - Datasets - dataset_pamir.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-01
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, hf_token, patch_ros2_qos_profiles_metadata, run_rosbag_frame_extraction, write_csv_rows

IMAGE_TOPIC = "/gopro/image_raw/compressed"
IMU_TOPIC = "/gopro/imu"


class PamirBagsMixin:
    """Shared ROS2-mcap-bag handling for the pamir/pamir-rig datasets (afrl-uw/Pamir_Visual-Inertial_Data).

    Each subclass sets SEQUENCE_BAGS: dict[sequence_name, [bag_name, ...]] - one bag name per
    camera, in cam-index order (bag_names[0] is always cam 0 / rgb_0's own IMU). No verified
    calibration exists anywhere in the source (confirmed by parsing the mcap schema directly - no
    camera_info topic, no calibration file was ever uploaded despite the source README
    documenting one) - every camera is written as cam_model 'unknown'.
    """

    SEQUENCE_BAGS: ClassVar[dict[str, list[str]]] = {}

    def _bag_dir(self, sequence_name: str, bag_name: str) -> Path:
        # Fetches (or verifies, if already local) both the .mcap data file and its metadata.yaml
        # into a real rosbag2 directory layout, so rosbag2_py can open it the standard way rather
        # than relying on undocumented single-file-without-metadata behavior.
        local_root = self.sequence_path(sequence_name) / "bags"
        for filename in (f"Ros2Bags/{bag_name}/{bag_name}.mcap", f"Ros2Bags/{bag_name}/metadata.yaml"):
            hf_hub_download(repo_id=self.hf_repo_id, filename=filename, repo_type="dataset",
                             token=hf_token(), local_dir=local_root)
        hf_bag_dir = local_root / "Ros2Bags" / bag_name

        # Patch into a separate directory rather than editing the HF-downloaded metadata.yaml in
        # place - hf_hub_download re-verifies that file against the repo on every call (see
        # download_sequence_data/create_imu_csv, which also call this method), and would either
        # ignore or overwrite an in-place edit.
        patched_dir = local_root.parent / "bags_patched" / bag_name
        return patch_ros2_qos_profiles_metadata(hf_bag_dir, patched_dir)

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        for bag_name in self.SEQUENCE_BAGS[sequence_name]:
            self._bag_dir(sequence_name, bag_name)

        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        bag_names = self.SEQUENCE_BAGS[sequence_name]
        raw_path = sequence_path / "rgb_raw"

        # extract-ros2bag-frames has no notion of self.target_resolution - it always writes at
        # the bag's native resolution. Extract into a throwaway raw_path first, then resize into
        # the real rgb_0/rgb_1 below (same rgb_0_raw/-then-resize shape as HFColmapDatasetMixin).
        for cam, bag_name in enumerate(bag_names):
            bag_dir = self._bag_dir(sequence_name, bag_name)
            run_rosbag_frame_extraction("ros2", bag_dir, raw_path, IMAGE_TOPIC, cam, storage_id="mcap")

        for cam in range(len(bag_names)):
            final_path = self.rgb_path(sequence_name) if cam == 0 else sequence_path / f"rgb_{cam}"
            if final_path.exists():
                continue
            final_path.mkdir(parents=True, exist_ok=True)
            for raw_image in sorted((raw_path / f"rgb_{cam}").glob("*.png")):
                with Image.open(raw_image) as img:
                    target_size = compute_scaled_size(img.size, self.target_resolution)
                    img.resize(target_size, Image.Resampling.LANCZOS).save(final_path / raw_image.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # extract-ros2bag-frames already wrote rgb_raw/rgb.csv with paths like "rgb_0/<ts>.png" -
        # identical to the final rgb_0/rgb_1 layout (same filenames, just resized in place above),
        # so the raw csv is valid as-is.
        raw_csv = self.sequence_path(sequence_name) / "rgb_raw" / "rgb.csv"
        shutil.copy2(raw_csv, self.rgb_csv_path(sequence_name))

    def create_imu_csv(self, sequence_name: str) -> None:
        # Each bag carries only its own camera's onboard IMU - use cam 0's bag (the same camera
        # rgb_0 comes from), never another rig camera's.
        bag_dir = self._bag_dir(sequence_name, self.SEQUENCE_BAGS[sequence_name][0])
        inputs = (f"--rosbag_path {bag_dir} --sequence_path {self.sequence_path(sequence_name)} "
                  f"--imu_topic {IMU_TOPIC} --imu_name 0 --storage_id mcap")
        subprocess.run(f"pixi run -e ros2 extract-ros2bag-imu {inputs}", shell=True)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        rgb: list[dict[str, Any]] = [
            {
                "cam_name": f"rgb_{cam}",
                "cam_type": "rgb",
                "cam_model": "unknown",
                "focal_length": [0.0, 0.0],
                "principal_point": [0.0, 0.0],
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4),
            }
            for cam in range(len(self.SEQUENCE_BAGS[sequence_name]))
        ]
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgb)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # COLMAP_SVIn2_Trajectories/ is documented in the source's own README but was never
        # uploaded to the repo - header only, no rows.
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        write_csv_rows(groundtruth_csv, header, [])

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "rgb_raw", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(sequence_path / "bags", ignore_errors=True)
            shutil.rmtree(sequence_path / "bags_patched", ignore_errors=True)


class PamirDataset(PamirBagsMixin, DatasetVSLAMLAB):
    """Pamir shipwreck underwater visual-inertial dataset helper for VSLAM-LAB benchmark."""

    SEQUENCE_BAGS: ClassVar[dict[str, list[str]]] = {
        "2025_Pamir_1": ["2025_Pamir_1"],
        "2025_Pamir_2": ["2025_Pamir_2"],
    }

    def __init__(self, dataset_name: str = "pamir") -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id: str = self.cfg["hf_repo_id"]
