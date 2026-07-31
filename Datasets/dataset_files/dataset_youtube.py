"""
Module: VSLAM-LAB - Datasets - dataset_youtube.py
- Author: Alejandro Fontan
- Assisted by: None
- Version: 1.0
- Created: 2026-04-11
- Updated: 2026-07-30
- License: GPLv3 License
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yt_dlp

from Datasets.dataset_files.dataset_videos import VideosDataset
from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention


class YoutubeDataset(VideosDataset):
    """YouTube dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "youtube") -> None:
        # Calls DatasetVSLAMLAB.__init__ directly (not super().__init__()) to skip
        # VideosDataset.__init__, which requires an hf_repo_id this dataset's yaml doesn't carry.
        DatasetVSLAMLAB.__init__(self, dataset_name)

        # Get sequence download urls
        self.url_download_sequences: dict[str, str] = self.cfg["url_download_sequences"]

        # Get time windows - sparse dict, only sequences needing a non-default window are listed
        self.time_windows: dict[str, list] = self.cfg.get("time_windows", {})

        # Get crop settings - sparse dict, only sequences that need cropping are listed
        self.crop_settings: dict[str, list[int]] = self.cfg.get("crop_settings", {})

        # Get calibration parameters - sparse dict, only sequences with real calibration are listed
        self.calibration_parameters: dict[str, dict] = self.cfg.get("calibration_parameters", {})

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        video_path = self._get_video_path(sequence_name)
        # A marker (not video_path.exists()) tracks completion, since a crashed/partial yt-dlp
        # download would otherwise leave a corrupt .mp4 that looks "already downloaded". This also
        # means two sequences sharing the same video (see _get_video_path) only download it once.
        marker = video_path.with_name(video_path.name + ".download_complete")
        if marker.exists():
            return
        sequence_path.mkdir(parents=True, exist_ok=True)

        url = self.url_download_sequences[sequence_name]
        ydl_opts = {
            "outtmpl": str(video_path),
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",  # ensure merged output is mp4
            "source_address": "0.0.0.0",  # force IPv4 - some networks can't route to
                                           # YouTube's IPv6-only CDN edges (yt-dlp's -4/--force-ipv4)
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return

        video_path = self._get_video_path(sequence_name)
        ti, tf = self._get_time_window(sequence_name)
        self.extract_png_frames(
            video_path=video_path,
            output_dir=rgb_path,
            target_resolution=self.target_resolution,
            ti=ti,
            tf=tf,
            crop=self._get_crop(sequence_name),
        )

    def create_calibration_yaml(self, sequence_name: str) -> None:
        cam_model, fx, fy, cx, cy, k1, k2, p1, p2 = self._get_calibration_parameters(sequence_name)
        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": cam_model,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        if cam_model != "unknown":
            rgb["distortion_type"] = cam_model
            rgb["distortion_coefficients"] = [k1, k2, p1, p2]
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def remove_unused_files(self, sequence_name: str) -> None:
        # video_path can be shared by more than one sequence (see _get_video_path) - only delete
        # it once every sequence sharing it already has its own rgb_0/ extracted, so deleting it
        # for one sequence never breaks a not-yet-processed sibling still needing it.
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            video_path = self._get_video_path(sequence_name)
            sibling_sequences = [s for s in self.sequence_names if self._get_video_path(s) == video_path]
            if all(self.rgb_path(s).exists() for s in sibling_sequences):
                marker = video_path.with_name(video_path.name + ".download_complete")
                video_path.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)

    def _get_crop(self, sequence_name: str) -> list[int] | None:
        """[top, bottom, left, right] pixels to trim from this sequence's frames, or None if the
        sequence isn't listed in crop_settings (i.e. no cropping needed)."""
        return self.crop_settings.get(sequence_name)

    def _get_time_window(self, sequence_name: str) -> tuple[float, float | None]:
        """(ti, tf) seconds to extract for this sequence, or (0, None) - the whole video - if the
        sequence isn't listed in time_windows."""
        ti, tf = self.time_windows.get(sequence_name, [0, None])
        return ti, tf

    def _get_video_path(self, sequence_name: str) -> str:
        if "fpv-drone-iceland" in sequence_name:
            return self.dataset_path / "fpv-drone-iceland.mp4"

        return self.dataset_path / f"{sequence_name}.mp4"

    def _get_calibration_parameters(
        self, sequence_name: str
    ) -> tuple[str, float, float, float, float, float, float, float, float]:
        """(cam_model, fx, fy, cx, cy, k1, k2, p1, p2) for this sequence, or all-zeroed
        "unknown" if the sequence isn't listed in calibration_parameters (i.e. no verified
        calibration exists for it)."""
        entry = self.calibration_parameters.get(sequence_name)
        if entry is None:
            return "unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        fx, fy = entry["focal_length"]
        cx, cy = entry["principal_point"]
        k1, k2, p1, p2 = entry["distortion_coefficients"]
        return entry["cam_model"], fx, fy, cx, cy, k1, k2, p1, p2
