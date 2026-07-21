"""
Module: VSLAM-LAB - Datasets - dataset_soneva.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-21
- License: GPLv3 License
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.DatasetVSLAMLab_issues import _get_dataset_issue
from path_constants import HUGGINGFACE_TOKEN

# COLMAP binary camera models this dataset's per-sequence reconstructions are known to use.
# https://colmap.github.io/cameras.html
_COLMAP_MODEL_NUM_PARAMS = {0: 3, 1: 4}  # 0: SIMPLE_PINHOLE (f, cx, cy), 1: PINHOLE (fx, fy, cx, cy)

# sequence_names are kept short (e.g. "hb_20250710"); every survey's actual top-level folder in
# the HF repo carries this shared prefix (e.g. "maldives_soneva_hb_20250710").
_REMOTE_SEQUENCE_PREFIX = "maldives_soneva_"


class SONEVA_dataset(DatasetVSLAMLab):
    """SONEVA dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "soneva") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.repo_id = cfg["repo_id"]

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace("_", " ") for s in self.sequence_names]

        # Get resolution size
        self.target_resolution = tuple(cfg["target_resolution"])

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0_raw"

        if rgb_path.exists():
            return
        rgb_path.mkdir(parents=True, exist_ok=True)

        remote_name = self._remote_sequence_name(sequence_name)
        subfolder = self._lhs_subfolder(sequence_name)
        remote_dir = f"{remote_name}/raw/{subfolder}"

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(rgb_path),
            allow_patterns=[f"{remote_dir}/*"],
            max_workers=8,
            token=self._hf_token(),
        )

        # snapshot_download() mirrors the repo's folder structure under local_dir; flatten it
        # since VSLAM-LAB wants the images directly under rgb_0_raw/.
        nested_dir = rgb_path / remote_name / "raw" / subfolder
        for file_path in nested_dir.iterdir():
            file_path.rename(rgb_path / file_path.name)
        shutil.rmtree(rgb_path / remote_name)
        shutil.rmtree(rgb_path / ".cache", ignore_errors=True)

    def create_rgb_folder(self, sequence_name: str) -> None:
        IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_raw_path = sequence_path / "rgb_0_raw"

        if rgb_path.exists():
            return
        if not rgb_raw_path.exists():
            return

        rgb_path.mkdir(parents=True, exist_ok=True)
        target_size = None
        init_size = None
        for file_path in tqdm(sorted(rgb_raw_path.iterdir()), desc="    resizing images"):
            if file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            with Image.open(file_path) as img:
                img.load()
                if target_size is None:
                    init_size = img.size
                    target_size = self._compute_scaled_size(img.size)

                if img.size != init_size:
                    print(f"{file_path.name} {img.size} != {init_size}")

                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(rgb_path / file_path.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return

        rgb_files = sorted(file_path.name for file_path in rgb_path.iterdir() if file_path.is_file())

        rgb = pd.DataFrame(
            {
                "ts_rgb_0 (ns)": [int(i * 1e9 / self.rgb_hz) for i in range(len(rgb_files))],
                "path_rgb_0": [f"rgb_0/{filename}" for filename in rgb_files],
            }
        )

        out = rgb[["ts_rgb_0 (ns)", "path_rgb_0"]]
        tmp = rgb_csv.with_suffix(".csv.tmp")
        try:
            out.to_csv(tmp, index=False)
            tmp.replace(rgb_csv)
        finally:
            tmp.unlink(missing_ok=True)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"

        cameras = self._read_colmap_cameras(sequence_name)
        raw_to_colmap = self._read_image_mapping(sequence_name)
        images = self._read_colmap_images(sequence_name)

        # Any registered LHS frame tells us which COLMAP camera_id is the LHS camera.
        camera_id = next(images[name][0] for name in raw_to_colmap.values() if name in images)
        model_name, width, height, params = cameras[camera_id]

        if model_name == "SIMPLE_PINHOLE":
            f, cx, cy = params
            fx, fy = f, f
        else:  # PINHOLE
            fx, fy, cx, cy = params

        # Rescale intrinsics from COLMAP's reference image size to the resized rgb_0 image size.
        with Image.open(next(rgb_path.iterdir())) as img:
            resized_w, resized_h = img.size
        scale_x, scale_y = resized_w / width, resized_h / height

        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx * scale_x, fy * scale_y],
            "principal_point": [cx * scale_x, cy * scale_y],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        raw_to_colmap = self._read_image_mapping(sequence_name)
        images = self._read_colmap_images(sequence_name)

        rgb_files = sorted(file_path.name for file_path in rgb_path.iterdir() if file_path.is_file())

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            for i, filename in enumerate(rgb_files):
                colmap_name = raw_to_colmap.get(filename)
                if colmap_name is None or colmap_name not in images:
                    continue

                _, qvec, tvec = images[colmap_name]
                tx, ty, tz, qx, qy, qz, qw = self._world_to_camera_to_pose(qvec, tvec)

                ts_ns = int(i * 1e9 / self.rgb_hz)
                w.writerow([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        tmp.replace(groundtruth_csv)

    def get_download_issues(self, _):
        if self._hf_token() is not None:
            return []
        return [
            _get_dataset_issue(
                issue_id="huggingface_token",
                dataset_name=self.dataset_name,
                website="https://huggingface.co/settings/tokens",
                yaml_file=str(self.yaml_file),
            )
        ]

    def _compute_scaled_size(self, original_size: tuple[int, int]) -> tuple[int, int]:
        target_w, target_h = self.target_resolution
        orig_w, orig_h = original_size
        target_area = target_w * target_h

        scaled_h = int(np.sqrt(target_area * orig_h / orig_w))
        scaled_w = int(target_area / scaled_h)
        return scaled_w, scaled_h

    def _hf_token(self) -> str | None:
        if HUGGINGFACE_TOKEN is not None:
            login(token=HUGGINGFACE_TOKEN)
            return HUGGINGFACE_TOKEN
        return os.environ.get("HF_TOKEN")

    def _all_repo_files(self) -> list[str]:
        cache_file = self.dataset_path / "all_files_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        api = HfApi(token=self._hf_token())
        all_files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(all_files, f, indent=2)
        return all_files

    @staticmethod
    def _remote_sequence_name(sequence_name: str) -> str:
        return f"{_REMOTE_SEQUENCE_PREFIX}{sequence_name}"

    def _lhs_subfolder(self, sequence_name: str) -> str:
        prefix = f"{self._remote_sequence_name(sequence_name)}/raw/"
        subfolders = {f[len(prefix):].split("/")[0] for f in self._all_repo_files() if f.startswith(prefix)}
        for subfolder in subfolders:
            if "lhs" in subfolder.lower():
                return subfolder
        raise FileNotFoundError(f"No LHS subfolder found for sequence {sequence_name}")

    def _fetch_colmap_file(self, sequence_name: str, filename: str) -> Path:
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=f"{self._remote_sequence_name(sequence_name)}/colmap/{filename}",
            token=self._hf_token(),
        )
        return Path(local_path)

    def _read_colmap_cameras(self, sequence_name: str) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
        path = self._fetch_colmap_file(sequence_name, "cameras.bin")
        with open(path, "rb") as f:
            data = f.read()

        offset = 0
        num_cameras = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        cameras: dict[int, tuple[str, int, int, tuple[float, ...]]] = {}
        for _ in range(num_cameras):
            camera_id, model_id = struct.unpack_from("<ii", data, offset)
            offset += 8
            width, height = struct.unpack_from("<QQ", data, offset)
            offset += 16
            n = _COLMAP_MODEL_NUM_PARAMS[model_id]
            params = struct.unpack_from(f"<{n}d", data, offset)
            offset += 8 * n
            model_name = "SIMPLE_PINHOLE" if model_id == 0 else "PINHOLE"
            cameras[camera_id] = (model_name, width, height, params)
        return cameras

    def _read_colmap_images(self, sequence_name: str) -> dict[str, tuple[int, tuple[float, ...], tuple[float, ...]]]:
        path = self._fetch_colmap_file(sequence_name, "images.bin")
        with open(path, "rb") as f:
            data = f.read()

        offset = 0
        num_images = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        images: dict[str, tuple[int, tuple[float, ...], tuple[float, ...]]] = {}
        for _ in range(num_images):
            offset += 4  # image_id
            qvec = struct.unpack_from("<4d", data, offset)
            offset += 32
            tvec = struct.unpack_from("<3d", data, offset)
            offset += 24
            camera_id = struct.unpack_from("<i", data, offset)[0]
            offset += 4

            end = data.index(b"\x00", offset)
            name = data[offset:end].decode("utf-8")
            offset = end + 1

            num_points2d = struct.unpack_from("<Q", data, offset)[0]
            offset += 8 + num_points2d * 24  # x, y (double) + point3D_id (int64) per point

            images[name] = (camera_id, qvec, tvec)
        return images

    def _read_image_mapping(self, sequence_name: str) -> dict[str, str]:
        path = self._fetch_colmap_file(sequence_name, "image_mapping.csv")
        subfolder = self._lhs_subfolder(sequence_name)

        mapping: dict[str, str] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                raw_path = row["raw_path"]
                if raw_path.split("/")[1] != subfolder:
                    continue
                mapping[Path(raw_path).name] = row["colmap_image"]
        return mapping

    @staticmethod
    def _world_to_camera_to_pose(
        qvec: tuple[float, float, float, float], tvec: tuple[float, float, float]
    ) -> tuple[float, float, float, float, float, float, float]:
        # COLMAP's qvec/tvec transform world -> camera coordinates (X_cam = R * X_world + t).
        # Groundtruth trajectories are stored as the camera pose in the world frame instead.
        qw, qx, qy, qz = qvec
        rot_world_to_cam = Rotation.from_quat([qx, qy, qz, qw])
        rot_cam_to_world = rot_world_to_cam.inv()

        t_world = rot_cam_to_world.apply(-np.array(tvec))
        qx_w, qy_w, qz_w, qw_w = rot_cam_to_world.as_quat()

        return float(t_world[0]), float(t_world[1]), float(t_world[2]), float(qx_w), float(qy_w), float(qz_w), float(qw_w)
