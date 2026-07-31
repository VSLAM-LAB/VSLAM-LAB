"""
Module: VSLAM-LAB - Datasets - dataset_videos.py
- Author: Alejandro Fontan
- Assisted by: None
- Version: 1.0
- Created: 2026-02-14
- Updated: 2026-07-30
- License: GPLv3 License
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, hf_token, make_printers, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


class VideosDataset(DatasetVSLAMLAB):
    """Videos dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "videos") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.hf_repo_id = self.cfg["hf_repo_id"]

    def download_sequence_data(self, sequence_name: str) -> None:
        token = hf_token()
        api = HfApi(token=token)
        fs = HfFileSystem(token=token)
        disable_progress_bars()

        remote_file = self._find_remote_file(sequence_name, api)
        if remote_file is None:
            raise FileNotFoundError(
                f"No file matching sequence '{sequence_name}' found in Hugging Face repo '{self.hf_repo_id}'."
            )

        local_file = self.dataset_path / remote_file
        marker = local_file.with_name(local_file.name + ".download_complete")
        if marker.exists():
            return

        fs.get_file(f"datasets/{self.hf_repo_id}/{remote_file}", str(local_file))
        marker.touch()

    def _find_remote_file(self, sequence_name: str, api: HfApi) -> str | None:
        # Cached listing of the repo's files, refreshed on a cache miss (rather than trusted
        # forever) so a file/sequence added to the repo after the cache was written is still found.
        cache_file = self.dataset_path / "all_files_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_files = json.load(f)
            match = next((f for f in cached_files if sequence_name in f), None)
            if match is not None:
                return match

        all_files = api.list_repo_files(repo_id=self.hf_repo_id, repo_type="dataset")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(all_files, f, indent=2)
        print_info(f"Fetched and cached {len(all_files)} files")

        return next((f for f in all_files if sequence_name in f), None)

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return

        video_path = next(
            (p for p in self.dataset_path.iterdir()
             if p.is_file() and sequence_name in p.name and not p.name.endswith(".download_complete")),
            None,
        )
        if video_path is None:
            raise FileNotFoundError(f"No downloaded video file found for sequence '{sequence_name}' in {self.dataset_path}")

        self.extract_png_frames(video_path=video_path, output_dir=rgb_path, target_resolution=self.target_resolution)  # extract at 30Hz

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        rgb_files = sorted(p.name for p in rgb_path.iterdir() if p.is_file())

        header = ["ts_rgb_0 (ns)", "path_rgb_0"]
        rows = [
            [int(idx / self.rgb_hz * 1e9), f"rgb_0/{filename}"]
            for idx, filename in enumerate(rgb_files)
        ]
        write_csv_rows(rgb_csv, header, rows)

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
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        write_csv_rows(groundtruth_csv, header, [])

    def remove_unused_files(self, sequence_name: str) -> None:
        # Each sequence's downloaded video file is exclusively its own (unlike youtube, where two
        # sequences can share one physical video) - safe to delete at MINIMAL retention once the
        # standardized rgb_0/ layout already has what it needs. all_files_cache.json is left alone
        # at every tier - it's a dataset-wide resource every future sequence's download_sequence_data
        # re-reads, not something scoped to this one sequence.
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            video_path = next(
                (p for p in self.dataset_path.iterdir()
                 if p.is_file() and sequence_name in p.name and not p.name.endswith(".download_complete")
                 and p.name != "all_files_cache.json"),
                None,
            )
            if video_path is not None:
                marker = video_path.with_name(video_path.name + ".download_complete")
                video_path.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)

    def estimate_new_resolution(self, original_width: int, original_height: int, target_resolution: list[int] | None = None) -> tuple[int, int]:
        return compute_scaled_size(
            (original_width, original_height), tuple(target_resolution) if target_resolution else None
        )

    def extract_png_frames(
        self,
        video_path: Path,
        output_dir: Path,
        target_resolution: list[int] | None = None,
        ti: float = 0.0,
        tf: float = None,
        crop: list[int] | None = None,
    ):
        """
        Extract frames from a video based on a frequency in Hertz (frames per second) and save as PNG images.
        Also creates an rgb.txt file with timestamps and image paths.
        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save the PNG files.
            target_resolution (list[int] | None): Target resolution for the output frames.
            ti (float): Start time in seconds. Defaults to 0.
            tf (float): End time in seconds. Defaults to end of video.
            crop (list[int] | None): [top, bottom, left, right] pixels to trim from each edge of
                the raw frame, applied before target_resolution resizing. Defaults to no crop.
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

        # Validate crop against the video's actual frame size up front, before processing any
        # frames, rather than failing deep inside the per-frame loop.
        if crop is not None:
            crop_top, crop_bottom, crop_left, crop_right = crop
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if crop_top + crop_bottom >= frame_height or crop_left + crop_right >= frame_width:
                raise ValueError(
                    f"crop {crop} leaves no pixels for a {frame_width}x{frame_height} frame."
                )

        # Seek to start frame
        start_frame = int(round(ti * fps))
        end_frame = int(round(tf * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_interval = int(round(fps / self.rgb_hz))
        print_info(f"Video opened: {video_path}")
        print_info(f"Video FPS: {fps:.2f}")
        print_info(f"Extracting {self.rgb_hz} frames per second (every {frame_interval} frames).")
        print_info(f"Time range: {ti:.2f}s to {tf:.2f}s (frames {start_frame} to {end_frame})")

        frame_idx = start_frame
        saved_idx = 0
        timestamp_list = []
        scale_image = target_resolution is not None
        needs_scaled_size = True

        pbar = tqdm(total=end_frame - start_frame + 1, desc="    extracting frames", unit="frame")
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            if (frame_idx - start_frame) % frame_interval == 0:
                # Compute timestamp from the beginning of the video
                timestamp_nsec = int(1e9 * frame_idx / fps)

                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if crop is not None:
                    h, w = rgb_frame.shape[:2]
                    rgb_frame = rgb_frame[crop_top:h - crop_bottom, crop_left:w - crop_right]

                if needs_scaled_size and scale_image:
                    rgb_frame_height, rgb_frame_width = rgb_frame.shape[:2]
                    scaled_width, scaled_height = compute_scaled_size(
                        (rgb_frame_width, rgb_frame_height), tuple(target_resolution)
                    )
                    needs_scaled_size = False
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

        pbar.close()
        cap.release()

    def _get_calibration_parameters(self, sequence_name: str) -> tuple[str, float, float, float, float]:
        if sequence_name in ["GX010213", "GX010214", "GX010215", "GX010216", "GX010217"]:
            return "pinhole", 494.82475772566221, 494.82475772566221, 369.5, 207.5
        return "unknown", 0.0, 0.0, 0.0, 0.0