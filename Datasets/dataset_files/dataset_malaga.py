"""
Module: VSLAM-LAB - Datasets - dataset_malaga.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-15
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Final
from urllib.parse import urljoin

import numpy as np
from PIL import Image
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, decompressFile, downloadFile, scale_intrinsics, write_csv_rows

EXTRACT_PREFIX: Final = "malaga-urban-dataset-extract-"
# Rectified images: stereo centers coincide and distortion is zero (see the dataset's
# camera_params_rectified_a=0_1024x768.txt), so the calibration is a plain pinhole model.
RECTIFIED_CALIBRATION_TXT: Final = "camera_params_rectified_a=0_1024x768.txt"
DOWNLOAD_COMPLETE_MARKER: Final = ".download_complete"


def _timestamp_ns(stamp: str) -> int:
    """Converts a 'seconds.fraction' UNIX stamp to integer nanoseconds without the precision
    loss of float(stamp) * 1e9 (these stamps exceed float64's integer range in ns)."""
    seconds, _, fraction = stamp.partition('.')
    return int(seconds) * 1_000_000_000 + int(fraction.ljust(9, '0')[:9])


def _image_timestamp(image_name: str) -> str:
    """'img_CAMERA1_1261228749.918590_left.jpg' -> '1261228749.918590'."""
    return image_name.split('_')[2]


class MalagaDataset(DatasetVSLAMLAB):
    """Malaga Urban Dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "malaga") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [f"extract {s}" for s in self.sequence_names]

    def _extract_name(self, sequence_name: str) -> str:
        return EXTRACT_PREFIX + sequence_name

    def _raw_extract_path(self, sequence_name: str) -> Path:
        return self.dataset_path / self._extract_name(sequence_name)

    def _raw_rectified_path(self, sequence_name: str) -> Path:
        extract_name = self._extract_name(sequence_name)
        return self._raw_extract_path(sequence_name) / f"{extract_name}_rectified_1024x768_Images"

    def _paired_timestamps(self, sequence_name: str) -> list[str]:
        """Timestamps with both a left and a right rectified image (a handful of frames ship
        with only one side)."""
        raw_rectified_path = self._raw_rectified_path(sequence_name)
        stamps: dict[str, set[str]] = {'left': set(), 'right': set()}
        for side in stamps:
            for image_file in raw_rectified_path.glob(f"*_{side}.jpg"):
                stamps[side].add(_image_timestamp(image_file.name))
        return sorted(stamps['left'] & stamps['right'])

    def download_sequence_data(self, sequence_name: str) -> None:
        extract_name = self._extract_name(sequence_name)
        compressed_file = self.dataset_path / (extract_name + '.zip')
        marker = self._raw_extract_path(sequence_name) / DOWNLOAD_COMPLETE_MARKER
        if marker.exists():
            return

        # is_zipfile checks the central directory at the end of the file, so a zip truncated by
        # an interrupted earlier run fails it and gets re-downloaded instead of blocking forever.
        if not compressed_file.exists() or not zipfile.is_zipfile(compressed_file):
            compressed_file.unlink(missing_ok=True)
            downloadFile(urljoin(self.url_download_root, extract_name + '.zip'), self.dataset_path)

        decompressFile(compressed_file, self.dataset_path)
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        raw_rectified_path = self._raw_rectified_path(sequence_name)
        paired_stamps = set(self._paired_timestamps(sequence_name))

        for side, rgb_path in (('left', self.rgb_path(sequence_name)), ('right', sequence_path / 'rgb_1')):
            if not raw_rectified_path.is_dir() or rgb_path.exists():
                continue

            rgb_path.mkdir(parents=True, exist_ok=True)
            image_files = [f for f in sorted(raw_rectified_path.glob(f"*_{side}.jpg"))
                           if _image_timestamp(f.name) in paired_stamps]
            target_size = None
            for image_file in tqdm(image_files, desc=f"    resizing {side} images"):
                if self.target_resolution is None:
                    shutil.copy2(image_file, rgb_path / image_file.name)
                    continue
                with Image.open(image_file) as img:
                    img.load()
                    if target_size is None:
                        target_size = compute_scaled_size(img.size, self.target_resolution)
                    img.resize(target_size, Image.Resampling.LANCZOS).save(rgb_path / image_file.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        sequence_path = self.sequence_path(sequence_name)
        rgb_path_0 = self.rgb_path(sequence_name)
        rgb_path_1 = sequence_path / 'rgb_1'

        stamp_to_file: dict[str, dict[str, str]] = {}
        for side, rgb_path in (('left', rgb_path_0), ('right', rgb_path_1)):
            for image_file in rgb_path.glob("*.jpg"):
                stamp_to_file.setdefault(_image_timestamp(image_file.name), {})[side] = image_file.name

        header = ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1']
        rows = []
        for stamp in sorted(s for s, files in stamp_to_file.items() if len(files) == 2):
            ts_ns = _timestamp_ns(stamp)
            rows.append([ts_ns, f"rgb_0/{stamp_to_file[stamp]['left']}",
                         ts_ns, f"rgb_1/{stamp_to_file[stamp]['right']}"])
        write_csv_rows(rgb_csv, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_txt = self._raw_extract_path(sequence_name) / RECTIFIED_CALIBRATION_TXT

        section = ""
        params: dict[str, dict[str, str]] = {}
        with open(calibration_txt, 'r') as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1]
                elif '=' in line:
                    key, value = (part.strip() for part in line.split('=', 1))
                    params.setdefault(section, {})[key] = value

        native_size = tuple(int(v) for v in params['CAMERA_LEFT']['resolution'].strip('[]').split())
        # pose_quaternion=[x y z qr qx qy qz]: right camera pose wrt left; x is the baseline (m).
        baseline = float(params['CAMERA_LEFT2RIGHT_POSE']['pose_quaternion'].strip('[]').split()[0])

        cams = []
        for cam_idx, camera_section in enumerate(('CAMERA_LEFT', 'CAMERA_RIGHT')):
            cam = params[camera_section]
            focal_length, principal_point = scale_intrinsics(
                (float(cam['fx']), float(cam['fy'])), (float(cam['cx']), float(cam['cy'])),
                native_size, self.target_resolution)

            T_BS = np.eye(4)
            if cam_idx == 1:
                T_BS[0, 3] = baseline
            cams.append({
                "cam_name": f"rgb_{cam_idx}",
                "cam_type": "rgb",
                "cam_model": "pinhole",
                "focal_length": focal_length,
                "principal_point": principal_point,
                "fps": self.rgb_hz,
                "T_BS": T_BS,
            })

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=cams)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # Groundtruth is the onboard GPS: Local Cartesian X/Y/Z positions (meters, +Z up) with
        # respect to a point near the start. Position-only - orientation is written as identity.
        extract_name = self._extract_name(sequence_name)
        gps_txt = self._raw_extract_path(sequence_name) / f"{extract_name}_all-sensors_GPS.txt"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)

        header = ['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw']
        rows = []
        with open(gps_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                values = line.split()
                # Columns (README_extracts.txt): 0=Time, 4=fix (0 means signal lost), 8..10=Local X/Y/Z.
                if int(float(values[4])) == 0:
                    continue
                ts_ns = _timestamp_ns(values[0])
                tx, ty, tz = float(values[8]), float(values[9]), float(values[10])
                rows.append([ts_ns, tx, ty, tz, 0.0, 0.0, 0.0, 1.0])
        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        raw_extract_path = self._raw_extract_path(sequence_name)
        compressed_file = self.dataset_path / (self._extract_name(sequence_name) + '.zip')

        # rgb_0/rgb_1 hold real (resized) copies, never symlinks into the raw extract folder,
        # so the whole folder is re-derivable from the kept zip.
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(raw_extract_path, ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            compressed_file.unlink(missing_ok=True)
