"""
Module: VSLAM-LAB - Datasets - dataset_soneva.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-21
- Updated: 2026-07-25
- License: GPLv3 License
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import (
    compute_scaled_size, ensure_hf_sequence_download, hf_token, make_printers, read_colmap_cameras,
    read_colmap_images, world_to_camera_to_pose, write_csv_rows,
)

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# sequence_names are kept short (e.g. "hb_20250710"); every survey's actual top-level folder in
# the HF repo carries this shared prefix (e.g. "maldives_soneva_hb_20250710").
_REMOTE_SEQUENCE_PREFIX = "maldives_soneva_"


class HFColmapDatasetMixin:
    """Shared logic for a Hugging-Face-sourced dataset with a per-sequence COLMAP reconstruction,
    used by both SonevaDataset (below) and SweetcoralsDataset (dataset_sweetcorals.py, which
    imports this mixin from here rather than utilities.py, since it's specific to this pair of
    datasets). Concrete classes still define their own __init__, download_sequence_data,
    create_calibration_yaml, and create_groundtruth_csv - only the parts that were byte-identical
    between the two are here. _fetch_colmap_file() is also here (a template method), but concrete
    classes must provide _remote_sequence_name(sequence_name): the one piece that actually differs
    (a dynamic HfApi lookup in SonevaDataset vs. a hardcoded table in SweetcoralsDataset). The
    COLMAP binary-format parsing itself (read_colmap_cameras/read_colmap_images/
    world_to_camera_to_pose) lives in utilities.py instead, since it doesn't depend on either
    dataset - only fetching the file to parse does."""

    def create_rgb_folder(self, sequence_name: str) -> None:
        IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
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

            if self.target_resolution is None:
                shutil.copy2(file_path, rgb_path / file_path.name)
                continue

            with Image.open(file_path) as img:
                img.load()
                if target_size is None:
                    init_size = img.size
                    target_size = compute_scaled_size(img.size, self.target_resolution)

                if img.size != init_size:
                    print_warning(f"{file_path.name} {img.size} != {init_size}")

                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(rgb_path / file_path.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        rgb_files = sorted(file_path.name for file_path in rgb_path.iterdir() if file_path.is_file())
        rows = [[int(i * 1e9 / self.rgb_hz), f"rgb_0/{filename}"] for i, filename in enumerate(rgb_files)]
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def get_download_issues(self, _):
        if hf_token() is not None:
            return []
        return [
            _get_dataset_issue(
                issue_id="huggingface_token",
                dataset_name=self.dataset_name,
                website="https://huggingface.co/settings/tokens",
                yaml_file=str(self.yaml_file),
            )
        ]

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberately narrow: only rgb_0_raw/ (the resized-away raw images) is ever removed, even
        # at MINIMAL retention. Anything else a concrete subclass keeps around - e.g.
        # SonevaDataset's all_files_cache.json, a dataset-wide (not per-sequence) HfApi listing
        # cache reused across every sequence's download - is left alone here; override this method
        # if a subclass needs to clean up more.
        sequence_path = self.sequence_path(sequence_name)
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(sequence_path / "rgb_0_raw", ignore_errors=True)

    def _fetch_colmap_file(self, sequence_name: str, filename: str) -> Path:
        """Downloads a single file from this sequence's colmap/ folder in the HF repo. Concrete
        classes must provide _remote_sequence_name(sequence_name) - the only piece that actually
        differs between soneva/sweetcorals (a dynamic HfApi lookup vs. a hardcoded table)."""
        local_path = hf_hub_download(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            filename=f"{self._remote_sequence_name(sequence_name)}/colmap/{filename}",
            token=hf_token(),
        )
        return Path(local_path)


class SonevaDataset(HFColmapDatasetMixin, DatasetVSLAMLAB):
    """Soneva dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "soneva") -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id = self.cfg["hf_repo_id"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = sequence_path / "rgb_0_raw"

        remote_folder = self._remote_sequence_name(sequence_name)
        subfolder = self._lhs_subfolder(sequence_name)
        remote_dir = f"{remote_folder}/raw/{subfolder}"

        ensure_hf_sequence_download(self.hf_repo_id, [remote_dir], rgb_path, token=hf_token())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        cameras = read_colmap_cameras(self._fetch_colmap_file(sequence_name, "cameras.bin"))
        raw_to_colmap = self._read_image_mapping(sequence_name)
        images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))

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
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        raw_to_colmap = self._read_image_mapping(sequence_name)
        images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))
        rgb_files = sorted(file_path.name for file_path in rgb_path.iterdir() if file_path.is_file())

        rows = []
        for i, filename in enumerate(rgb_files):
            colmap_name = raw_to_colmap.get(filename)
            if colmap_name is None or colmap_name not in images:
                continue

            _, qvec, tvec = images[colmap_name]
            tx, ty, tz, qx, qy, qz, qw = world_to_camera_to_pose(qvec, tvec)
            ts_ns = int(i * 1e9 / self.rgb_hz)
            rows.append([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        write_csv_rows(groundtruth_csv, ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], rows)

    def _all_repo_files(self) -> list[str]:
        cache_file = self.dataset_path / "all_files_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        api = HfApi(token=hf_token())
        all_files = api.list_repo_files(repo_id=self.hf_repo_id, repo_type="dataset")
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
