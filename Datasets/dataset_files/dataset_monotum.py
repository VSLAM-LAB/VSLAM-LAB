from __future__ import annotations

import csv
import os
import shutil
import subprocess
from contextlib import suppress
from pathlib import Path
from urllib.parse import urljoin
from zipfile import ZipFile
from typing import Any

import numpy as np

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, Retention, VSLAM_LAB_DIR
from utilities import downloadFile, decompressFile, replace_string_in_files


class MONOTUM_dataset(DatasetVSLAMLab):
    """Monocular TUM dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "monotum") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        self.url_download_root: str = self.cfg_require("url_download_root")
        self.sequence_nicknames = [s.replace("sequence_", "seq ") for s in self.sequence_names]
        self.mono_dataset_code_dir = VSLAM_LAB_DIR / "Baselines" / "mono_dataset_code"

    def download_sequence_data(self, sequence_name: str) -> None:
        compressed_name_ext = f"{sequence_name}.zip"
        download_url = urljoin(self.url_download_root.rstrip("/") + "/", compressed_name_ext)

        compressed_file = self.dataset_path / compressed_name_ext
        sequence_path = self.dataset_path / sequence_name

        if not compressed_file.exists():
            downloadFile(download_url, str(self.dataset_path))

        if not sequence_path.exists():
            decompressFile(str(compressed_file), str(self.dataset_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_path.mkdir(parents=True, exist_ok=True)

        images_zip = sequence_path / "images.zip"
        if not images_zip.exists():
            return

        if self._run_official_undistort(sequence_path):
            return

        # Fallback: extract image files directly when undistorter is unavailable.
        with ZipFile(images_zip, "r") as zf:
            image_members = [m for m in zf.namelist()
                             if m.lower().endswith((".jpg", ".jpeg", ".png"))
                             and not m.endswith("/")]
            image_members.sort()
            for i, member in enumerate(image_members):
                ext = Path(member).suffix.lower() or ".jpg"
                out_name = f"{i:05d}{ext}"
                with zf.open(member) as src, open(rgb_path / out_name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        times_txt = sequence_path / "times.txt"
        rgb_csv = sequence_path / "rgb.csv"

        timestamps = self._read_timestamps(times_txt)
        rgb_files = sorted([f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()])

        n = min(len(rgb_files), len(timestamps))
        with open(rgb_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            for i in range(n):
                ts_ns = int(float(timestamps[i]) * 1e9)
                writer.writerow([ts_ns, f"rgb_0/{rgb_files[i]}"])

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        calibration_txt = sequence_path / "calibration.txt"
        camera_txt = sequence_path / "camera.txt"

        fx = fy = cx = cy = None
        if calibration_txt.exists():
            with open(calibration_txt, "r", encoding="utf-8") as f:
                vals = f.readline().split()
                if len(vals) >= 4:
                    fx, fy, cx, cy = map(float, vals[:4])
        elif camera_txt.exists():
            with open(camera_txt, "r", encoding="utf-8") as f:
                vals = f.readline().split()
                if len(vals) >= 4:
                    fx, fy, cx, cy = map(float, vals[:4])

        if None in (fx, fy, cx, cy):
            raise ValueError(f"Could not read intrinsics for sequence '{sequence_name}'")

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        src = sequence_path / "groundtruthSync.txt"
        dst = sequence_path / "groundtruth.csv"

        if not src.exists():
            return

        tmp = dst.with_suffix(".csv.tmp")
        with open(src, "r", encoding="utf-8") as fin, open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            for line in fin:
                s = line.strip()
                if not s or "NaN" in s:
                    continue
                parts = s.split()
                if len(parts) < 8:
                    continue
                ts_ns = int(float(parts[0]) * 1e9)
                w.writerow([ts_ns] + parts[1:8])
        tmp.replace(dst)
        with suppress(FileNotFoundError):
            tmp.unlink()

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        if BENCHMARK_RETENTION != Retention.FULL:
            for name in ("groundtruthSync.txt", "times.txt", "pcalib.txt", "statistics.txt", "vignette.png"):
                (sequence_path / name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / f"{sequence_name}.zip").unlink(missing_ok=True)

    @staticmethod
    def _read_timestamps(times_txt: Path) -> list[float]:
        if not times_txt.exists():
            raise FileNotFoundError(f"Missing file: {times_txt}")

        out: list[float] = []
        with open(times_txt, "r", encoding="utf-8") as f:
            for line in f:
                cols = line.split()
                if not cols:
                    continue
                ts = cols[1] if len(cols) > 1 else cols[0]
                out.append(float(ts))
        return out

    def _run_official_undistort(self, sequence_path: Path) -> bool:
        bin_file = self.mono_dataset_code_dir / "bin" / "playbackDataset"
        if not bin_file.exists() and not self._build_mono_dataset_code():
            return False

        rgb_dir = sequence_path / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(bin_file), str(sequence_path), str(sequence_path)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            return False

        if not rgb_dir.exists():
            return False

        target = sequence_path / "rgb_0"
        for img in sorted(rgb_dir.glob("*")):
            if img.is_file():
                img.rename(target / img.name)
        shutil.rmtree(rgb_dir, ignore_errors=True)
        return True

    def _build_mono_dataset_code(self) -> bool:
        if (self.mono_dataset_code_dir / "bin" / "playbackDataset").exists():
            return True

        if not self.mono_dataset_code_dir.exists():
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/tum-vision/mono_dataset_code.git", str(self.mono_dataset_code_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception:
                return False

        replace_string_in_files(str(self.mono_dataset_code_dir), "CV_LOAD_IMAGE_UNCHANGED", "cv::IMREAD_UNCHANGED")
        replace_string_in_files(str(self.mono_dataset_code_dir), "CV_LOAD_IMAGE_GRAYSCALE", "cv::IMREAD_GRAYSCALE")

        extra_dir = VSLAM_LAB_DIR / "Datasets" / "extra-files"
        with suppress(Exception):
            shutil.copy(extra_dir / "CMakeLists.txt", self.mono_dataset_code_dir / "CMakeLists.txt")
            shutil.copy(extra_dir / "main_playbackDataset.cpp", self.mono_dataset_code_dir / "src" / "main_playbackDataset.cpp")
            shutil.copy(extra_dir / "build.sh", self.mono_dataset_code_dir / "build.sh")

        try:
            subprocess.run(
                ["bash", str(self.mono_dataset_code_dir / "build.sh")],
                cwd=str(self.mono_dataset_code_dir),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            return False

        return (self.mono_dataset_code_dir / "bin" / "playbackDataset").exists()
