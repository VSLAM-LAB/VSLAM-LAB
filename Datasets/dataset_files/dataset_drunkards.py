"""
Module: VSLAM-LAB - Datasets - dataset_drunkards.py
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
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import gdown
import numpy as np
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, scale_intrinsics, write_csv_rows

# sequence_name -> (scene, resolution, level), e.g. "00000_320_level0" -> ("00000", "320", "level0").
# Matches the real Drive folder layout <resolution>/<scene>/<level>/{color,depth}.zip,pose.txt -
# confirmed via a live gdown.download_folder(skip_download=True) crawl of the dataset's own share
# link: 19 scenes total, numbered 00000-00016 plus 00018 and 00019 (00017 does not exist).
_SEQUENCE_RE = re.compile(r"^(\d{5})_(320|1024)_level(\d)$")


def _split_sequence_name(sequence_name: str) -> tuple[str, str, str]:
    m = _SEQUENCE_RE.match(sequence_name)
    if not m:
        raise ValueError(f"Unrecognized drunkards sequence name: {sequence_name}")
    scene, resolution, level = m.groups()
    return scene, resolution, f"level{level}"


class DrunkardsDataset(DatasetVSLAMLAB):
    """The Drunkard's Dataset synthetic exploratory-trajectory RGB-D benchmark helper for VSLAM-LAB."""

    def __init__(self, dataset_name: str = "drunkards") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.google_drive_link: str = self.cfg["google_drive_link"]

        # Depth factor: metric depth (m) = pixel_value / depth_factor
        self.depth_factor: float = self.cfg["depth_factor"]

    def _drive_manifest(self) -> dict[str, str]:
        # 152 sequences (19 scenes x 2 resolutions x 4 levels) aren't individually linked
        # anywhere - crawl the one real, official root folder ID once (gdown's skip_download=True
        # lists the whole tree - id, relative path - without downloading any file content) and
        # cache the path->id mapping on disk, so every sequence resolves its own file IDs from
        # real, verified data instead of a hand-typed map (the old implementation only covered
        # 5/152 sequences this way, via hardcoded per-sequence folder IDs). Model: dataset_hilti2026.py.
        manifest_path = self.dataset_path / "drive_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        files = gdown.download_folder(url=self.google_drive_link, skip_download=True, quiet=False)
        manifest = {f.path: f.id for f in files}
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
        return manifest

    @staticmethod
    def _native_size(resolution: str) -> tuple[int, int]:
        size = int(resolution)
        return size, size

    def _effective_target(self, resolution: str) -> tuple[int, int] | None:
        # Only the 1024x1024 sequences are bigger than target_resolution by pixel area - the
        # 320x320 sequences are already smaller and are left at their native size. A single
        # target_resolution value from the yaml, applied conditionally per sequence based on the
        # resolution encoded in its own sequence_name.
        if self.target_resolution is None:
            return None
        native_w, native_h = self._native_size(resolution)
        target_w, target_h = self.target_resolution
        if native_w * native_h <= target_w * target_h:
            return None
        return self.target_resolution

    def _wrong_frame_indices(self, sequence_name: str) -> set[int]:
        wrong_frames_file = self.sequence_path(sequence_name) / "wrong_frames.txt"
        if not wrong_frames_file.exists():
            return set()
        with wrong_frames_file.open("r", encoding="utf-8") as f:
            return {int(line) for line in f if line.strip()}

    def download_sequence_data(self, sequence_name: str) -> None:
        scene, resolution, level = _split_sequence_name(sequence_name)
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        manifest = self._drive_manifest()
        drive_relpath = f"{resolution}/{scene}/{level}"

        for filename in ("color.zip", "depth.zip", "pose.txt"):
            target = sequence_path / filename
            if not target.exists():
                gdown.download(id=manifest[f"{drive_relpath}/{filename}"], output=str(target), quiet=False)

        # Optional per-sequence corrupted-frame list (30/152 sequences have one)
        wrong_frames_key = f"{drive_relpath}/wrong_frames.txt"
        if wrong_frames_key in manifest:
            wrong_frames_file = sequence_path / "wrong_frames.txt"
            if not wrong_frames_file.exists():
                gdown.download(id=manifest[wrong_frames_key], output=str(wrong_frames_file), quiet=False)

        # One intrinsics.txt per resolution (dataset-wide, shared by every scene/level of that
        # resolution) - fetched into self.dataset_path, not the per-sequence sequence_path.
        intrinsics_file = self.dataset_path / f"intrinsics_{resolution}.txt"
        if not intrinsics_file.exists():
            gdown.download(id=manifest[f"{resolution}/intrinsics.txt"], output=str(intrinsics_file), quiet=False)

        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        _, resolution, _ = _split_sequence_name(sequence_name)
        sequence_path = self.sequence_path(sequence_name)
        effective_target = self._effective_target(resolution)

        rgb_path = self.rgb_path(sequence_name)
        if not rgb_path.exists():
            color_raw = sequence_path / "color_raw"
            if not color_raw.exists():
                with ZipFile(sequence_path / "color.zip") as zf:
                    zf.extractall(color_raw)
            rgb_path.mkdir(parents=True, exist_ok=True)
            for image_file in sorted(color_raw.rglob("*.png")):
                if effective_target is None:
                    shutil.copy2(image_file, rgb_path / image_file.name)
                else:
                    with Image.open(image_file) as img:
                        target_size = compute_scaled_size(img.size, effective_target)
                        img.resize(target_size, Image.Resampling.LANCZOS).save(rgb_path / image_file.name)

        depth_path = self.depth_path(sequence_name)
        if not depth_path.exists():
            depth_raw = sequence_path / "depth_raw"
            if not depth_raw.exists():
                with ZipFile(sequence_path / "depth.zip") as zf:
                    zf.extractall(depth_raw)
            depth_path.mkdir(parents=True, exist_ok=True)
            for depth_file in sorted(depth_raw.rglob("*.png")):
                if effective_target is None:
                    shutil.copy2(depth_file, depth_path / depth_file.name)
                else:
                    with Image.open(depth_file) as img:
                        target_size = compute_scaled_size(img.size, effective_target)
                        img.resize(target_size, Image.Resampling.NEAREST).save(depth_path / depth_file.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        wrong_frames = self._wrong_frame_indices(sequence_name)
        rgb_files = sorted(
            p for p in self.rgb_path(sequence_name).glob("*.png") if int(p.stem) not in wrong_frames
        )
        depth_files = sorted(
            p for p in self.depth_path(sequence_name).glob("*.png") if int(p.stem) not in wrong_frames
        )

        rows = []
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            frame_idx = int(rgb_file.stem)
            ts_ns = int(1e10 + frame_idx / self.rgb_hz * 1e9)
            rows.append([ts_ns, f"rgb_0/{rgb_file.name}", ts_ns, f"depth_0/{depth_file.name}"])

        write_csv_rows(
            self.rgb_csv_path(sequence_name),
            ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"],
            rows,
        )

    def create_calibration_yaml(self, sequence_name: str) -> None:
        _, resolution, _ = _split_sequence_name(sequence_name)
        intrinsics_file = self.dataset_path / f"intrinsics_{resolution}.txt"
        with intrinsics_file.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        fx, fy, cx, cy = (float(v) for v in lines[1].split(":", 1)[1].split(","))

        native_size = self._native_size(resolution)
        effective_target = self._effective_target(resolution)
        focal_length, principal_point = scale_intrinsics((fx, fy), (cx, cy), native_size, effective_target)

        rgbd0: dict[str, Any] = {
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
        wrong_frames = self._wrong_frame_indices(sequence_name)
        pose_txt = self.sequence_path(sequence_name) / "pose.txt"

        rows = []
        with pose_txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                frame_idx = int(parts[0])
                if frame_idx in wrong_frames:
                    continue
                tx, ty, tz, qx, qy, qz, qw = (float(v) for v in parts[1:8])
                ts_ns = int(1e10 + frame_idx / self.rgb_hz * 1e9)
                rows.append([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        write_csv_rows(
            self.groundtruth_csv_path(sequence_name),
            ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"],
            rows,
        )

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "color_raw", ignore_errors=True)
            shutil.rmtree(sequence_path / "depth_raw", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (sequence_path / "color.zip").unlink(missing_ok=True)
            (sequence_path / "depth.zip").unlink(missing_ok=True)
            (sequence_path / "pose.txt").unlink(missing_ok=True)
            (sequence_path / "wrong_frames.txt").unlink(missing_ok=True)
            # intrinsics_<resolution>.txt is dataset-wide and re-read by every sequence of that
            # resolution's create_calibration_yaml - never delete it here.
