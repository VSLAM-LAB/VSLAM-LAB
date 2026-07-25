from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from huggingface_hub import HfApi, HfFileSystem, login
from huggingface_hub.utils import disable_progress_bars

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import HUGGINGFACE_TOKEN, VSLAMLAB_VIDEOS


class VideosDataset(DatasetVSLAMLAB):
    """VIDEOS dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "videos") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get videso path
        self.videos_path = VSLAMLAB_VIDEOS

        # Get download url
        self.repo_id = cfg["repo_id"]

        # Create sequence_nicknames
        self.sequence_nicknames = self.sequence_names

        # Get resolution size
        self.target_resolution = cfg.get("target_resolution", None)

    def download_sequence_data(self, sequence_name: str) -> None:
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

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        for p in self.videos_path.iterdir():
            if p.is_file() and sequence_name in p.name:
                video_path = p
                break

        if not rgb_path.exists():
            self.extract_png_frames(video_path=video_path, output_dir=rgb_path, target_resolution=self.target_resolution)  # extract at 30Hz

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(rgb_path / f)]
        rgb_files.sort()

        tmp = rgb_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            idx = 0
            for filename in rgb_files:
                path_r0 = "rgb_0/" + filename
                ts_r0_ns = int(float(idx) / self.rgb_hz * 1e9)
                w.writerow([ts_r0_ns, path_r0])
                idx += 1
        tmp.replace(rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        model, fx, fy, cx, cy = self._get_calibration_parameters(sequence_name)
        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": model,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        tmp.replace(groundtruth_csv)

    def estimate_new_resolution(self, original_width: int, original_height: int, target_resolution: list[int] | None = None) -> tuple[int, int]:
        if target_resolution is None:
            return original_width, original_height

        scaled_height = np.sqrt(
            target_resolution[0] * target_resolution[1] * original_height / original_width
        )
        scaled_width = target_resolution[0] * target_resolution[1] / scaled_height
        return int(scaled_width), int(scaled_height)

    def extract_png_frames(self, video_path: Path, output_dir: Path, target_resolution: list[int] | None = None, ti: float = 0.0, tf: float = None):
        """
        Extract frames from a video based on a frequency in Hertz (frames per second) and save as PNG images.
        Also creates an rgb.txt file with timestamps and image paths.
        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save the PNG files.
            target_resolution (list[int] | None): Target resolution for the output frames.
            ti (float): Start time in seconds. Defaults to 0.
            tf (float): End time in seconds. Defaults to end of video.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Failed to get FPS from video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Validate and clamp ti/tf
        ti = max(0.0, ti)
        tf = min(tf, video_duration) if tf is not None else video_duration
        if ti >= tf:
            raise ValueError(f"ti ({ti}s) must be less than tf ({tf}s).")

        # Seek to start frame
        start_frame = int(round(ti * fps))
        end_frame = int(round(tf * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_interval = int(round(fps / self.rgb_hz))
        print(f"Video opened: {video_path}")
        print(f"Video FPS: {fps:.2f}")
        print(f"Extracting {self.rgb_hz} frames per second (every {frame_interval} frames).")
        print(f"Time range: {ti:.2f}s to {tf:.2f}s (frames {start_frame} to {end_frame})")

        frame_idx = start_frame
        saved_idx = 0
        timestamp_list = []
        scale_image = target_resolution is not None
        estimate_new_resolution = True

        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_interval == 0:
                # Compute timestamp from the beginning of the video
                timestamp_nsec = int(1e9 * frame_idx / fps)

                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if estimate_new_resolution and scale_image:
                    rgb_frame_height, rgb_frame_width = rgb_frame.shape[:2]
                    scaled_height = np.sqrt(
                        target_resolution[0] * target_resolution[1] * rgb_frame_height / rgb_frame_width
                    )
                    scaled_width = target_resolution[0] * target_resolution[1] / scaled_height
                    scaled_height = int(scaled_height)
                    scaled_width = int(scaled_width)
                    estimate_new_resolution = False
                if scale_image:
                    resized_img = cv2.resize(rgb_frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LANCZOS4)
                else:
                    resized_img = rgb_frame

                # Save as PNG with 5-digit padded integer filename
                filename = output_dir / f"{saved_idx:05d}.png"
                cv2.imwrite(str(filename), cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

                # Save timestamp and image path
                image_relative_path = output_dir / f"{saved_idx:05d}.png"
                timestamp_list.append((timestamp_nsec, str(image_relative_path)))
                saved_idx += 1

            frame_idx += 1

        cap.release()

    def _get_calibration_parameters(self, sequence_name: str) -> tuple[str, float, float, float, float]:
        if sequence_name in ["GX010213", "GX010214", "GX010215", "GX010216", "GX010217"]:
            return "pinhole", 494.82475772566221, 494.82475772566221, 369.5, 207.5
        return "unknown", 0.0, 0.0, 0.0, 0.0