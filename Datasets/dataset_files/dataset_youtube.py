from pathlib import Path
from typing import Any

import numpy as np
import yaml
import yt_dlp

from Datasets.dataset_files.dataset_videos import VIDEOS_dataset
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab


class YOUTUBE_dataset(VIDEOS_dataset):
    """YOUTUBE dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "youtube") -> None:
        DatasetVSLAMLab.__init__(self, dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get sequence download urls
        self.url_download_sequences: dict[str, str] = cfg["url_download_sequences"]

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

        # Get target resolution
        self.target_resolution = cfg.get("target_resolution", None)

        # Get time windows
        self.time_windows = cfg.get("time_windows", [[0, None] for _ in self.sequence_names])

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        video_path = self._get_video_path(sequence_name)
        if video_path.exists():
            return
        sequence_path.mkdir(parents=True, exist_ok=True)

        url = self.url_download_sequences[sequence_name]
        ydl_opts = {
            "outtmpl": str(video_path),
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",  # ensure merged output is mp4
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        video_path = self._get_video_path(sequence_name)

        if not rgb_path.exists():
            seq_index = self.sequence_names.index(sequence_name)
            self.extract_png_frames(
                video_path=video_path,
                output_dir=rgb_path,
                target_resolution=self.target_resolution,
                ti=self.time_windows[seq_index][0],
                tf=self.time_windows[seq_index][1],
            )

    def create_calibration_yaml(self, sequence_name: str) -> None:
        model, dist_type, fx, fy, cx, cy, k1, k2, p1, p2 = self._get_calibration_parameters(sequence_name)
        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": model,
            "distortion_type": dist_type,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "distortion_coefficients": [k1, k2, p1, p2],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def _get_video_path(self, sequence_name: str) -> str:
        if "fpv-drone-iceland" in sequence_name:
            return self.dataset_path / "fpv-drone-iceland.mp4"

        return self.dataset_path / f"{sequence_name}.mp4"

    def _get_calibration_parameters(
        self, sequence_name: str
    ) -> tuple[str, str, float, float, float, float, float, float, float, float]:
        if "fpv-drone-iceland" in sequence_name:
            return "pinhole", "radtan4", 231.7, 229.7, 369.5, 207.5, -0.004616, 0.000414, 0.001282, 0.001458
        return "unknown", "unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
