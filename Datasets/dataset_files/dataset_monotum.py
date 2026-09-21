"""
Module: VSLAM-LAB - Datasets - dataset_monotum.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-03
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import cv2
import numpy as np

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows


def _read_fov_calibration(camera_txt: Path) -> dict[str, Any]:
    """Parses a TUM monoVO camera.txt: pinhole+FOV intrinsics (fx fy cx cy omega, normalized to
    in_width/in_height), the input resolution, the output-camera spec ("crop"/"full"/explicit
    values), and the output resolution. Model: DSO's dataset-format doc
    (https://github.com/JakobEngel/dso#31-dataset-format) and UndistorterFOV's constructor
    (mono_dataset_code/src/FOVUndistorter.cpp)."""
    lines = camera_txt.read_text(encoding="utf-8").strip().splitlines()
    fx_n, fy_n, cx_n, cy_n, omega = (float(v) for v in lines[0].split()[:5])
    in_width, in_height = (int(v) for v in lines[1].split())
    out_spec = lines[2].strip()
    out_width, out_height = (int(v) for v in lines[3].split())
    return {
        "fx_n": fx_n, "fy_n": fy_n, "cx_n": cx_n, "cy_n": cy_n, "omega": omega,
        "in_width": in_width, "in_height": in_height,
        "out_spec": out_spec, "out_width": out_width, "out_height": out_height,
    }


def _fov_output_camera(calib: dict[str, Any]) -> tuple[float, float, float, float]:
    """Resolves the rectified (pinhole, zero-distortion) output camera matrix (ofx, ofy, ocx, ocy)
    in pixel units at out_width/out_height, replicating UndistorterFOV's explicit-output-values
    case (FOVUndistorter.cpp) - every TUM monoVO sequence.zip inspected so far (sequence_01) ships
    this form. Raises if a sequence instead ships "crop"/"full" (auto-solved output camera matrix)
    - that branch involves real per-sequence geometry search and was never exercised against
    actual data, so it's better surfaced loudly than run unverified."""
    in_w, in_h = calib["in_width"], calib["in_height"]
    out_w, out_h = calib["out_width"], calib["out_height"]
    omega = calib["omega"]

    if omega == 0:
        fx, fy = calib["fx_n"] * out_w, calib["fy_n"] * out_h
        cx, cy = calib["cx_n"] * out_w - 0.5, calib["cy_n"] * out_h - 0.5
        return fx, fy, cx, cy

    out_spec = calib["out_spec"]
    if out_spec in ("crop", "full", "none"):
        raise NotImplementedError(
            f"camera.txt output spec '{out_spec}' is not implemented (only explicit output "
            "values have been observed in this dataset's sequences)"
        )

    ofx_n, ofy_n, ocx_n, ocy_n = (float(v) for v in out_spec.split()[:4])
    return ofx_n * out_w, ofy_n * out_h, ocx_n * out_w - 0.5, ocy_n * out_h - 0.5


def _fov_undistort_maps(calib: dict[str, Any], out_cam: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Builds the (map_x, map_y) remap tables that turn an in_width x in_height FOV-distorted image
    into an out_width x out_height rectified (pinhole) one - the same undistorted-to-distorted
    pixel mapping as UndistorterFOV::distortCoordinates (FOVUndistorter.cpp), for use with
    cv2.remap. Reimplemented in Python so this dataset needs no external C++ build (unlike the
    original mono_dataset_code-based approach)."""
    fx = calib["fx_n"] * calib["in_width"]
    fy = calib["fy_n"] * calib["in_height"]
    cx = calib["cx_n"] * calib["in_width"] - 0.5
    cy = calib["cy_n"] * calib["in_height"] - 0.5
    omega = calib["omega"]
    d2t = 2.0 * np.tan(omega / 2.0) if omega != 0 else 0.0
    ofx, ofy, ocx, ocy = out_cam

    x, y = np.meshgrid(np.arange(calib["out_width"], dtype=np.float64), np.arange(calib["out_height"], dtype=np.float64))
    ix = (x - ocx) / ofx
    iy = (y - ocy) / ofy
    r = np.hypot(ix, iy)
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = np.where((r == 0) | (omega == 0), 1.0, np.arctan(r * d2t) / (omega * r))

    map_x = (fx * fac * ix + cx).astype(np.float32)
    map_y = (fy * fac * iy + cy).astype(np.float32)
    return map_x, map_y


class MonotumDataset(DatasetVSLAMLAB):
    """TUM monoVO (Monocular Visual Odometry) dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "monotum") -> None:
        super().__init__(dataset_name)

        self.url_download_root: str = self.cfg["url_download_root"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        compressed_name = f"{sequence_name}.zip"
        download_url = urljoin(self.url_download_root.rstrip("/") + "/", compressed_name)
        compressed_file = self.dataset_path / compressed_name

        if not compressed_file.exists():
            downloadFile(download_url, str(self.dataset_path))

        if not (sequence_path / "camera.txt").exists():
            decompressFile(str(compressed_file), str(self.dataset_path))

        images_zip = sequence_path / "images.zip"
        images_raw = sequence_path / "images_raw"
        if images_zip.exists() and not images_raw.exists():
            decompressFile(str(images_zip), str(images_raw))

        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return

        sequence_path = self.sequence_path(sequence_name)
        calib = _read_fov_calibration(sequence_path / "camera.txt")
        out_cam = _fov_output_camera(calib)
        map_x, map_y = _fov_undistort_maps(calib, out_cam)

        rgb_path.mkdir(parents=True, exist_ok=True)
        for image_file in sorted((sequence_path / "images_raw").glob("*.jpg")):
            img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            rectified = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cv2.imwrite(str(rgb_path / image_file.name), rectified)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        times_txt = self.sequence_path(sequence_name) / "times.txt"
        rows = []
        with open(times_txt, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.split()
                if not parts:
                    continue
                idx, ts = parts[0], parts[1]
                rows.append([int(round(float(ts) * 1e9)), f"rgb_0/{idx}.jpg"])
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        calib = _read_fov_calibration(sequence_path / "camera.txt")
        ofx, ofy, ocx, ocy = _fov_output_camera(calib)

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": [float(ofx), float(ofy)],
            "principal_point": [float(ocx), float(ocy)],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        groundtruth_txt = self.sequence_path(sequence_name) / "groundtruthSync.txt"
        rows = []
        if groundtruth_txt.exists():
            with open(groundtruth_txt, "r", encoding="utf-8") as fin:
                for line in fin:
                    parts = line.split()
                    if not parts or "NaN" in parts:
                        continue
                    rows.append([int(round(float(parts[0]) * 1e9))] + parts[1:])
        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            for name in ("camera.txt", "times.txt", "groundtruthSync.txt"):
                (sequence_path / name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            for name in ("images.zip", "pcalib.txt", "vignette.png", "statistics.txt"):
                (sequence_path / name).unlink(missing_ok=True)
            shutil.rmtree(sequence_path / "images_raw", ignore_errors=True)
            (self.dataset_path / f"{sequence_name}.zip").unlink(missing_ok=True)
