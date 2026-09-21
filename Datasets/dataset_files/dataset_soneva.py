"""
Module: VSLAM-LAB - Datasets - dataset_soneva.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5, Fable 5.1)
- Version: 1.1
- Created: 2026-07-21
- Updated: 2026-09-04
- License: GPLv3 License
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
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

# Every survey is shot with the Wildflow two-GoPro rig: two cameras on a 60 cm bar, both pointing
# down and parallel, each firing its *own* 0.5 s photo time-lapse. There is no hardware sync (the
# EXIF capture times of the two sides start seconds to minutes apart), the two bodies are often
# different GoPro models with different raw sizes (HERO10 5568x4176 vs HERO11 5568x4872), and the
# per-side frame counts differ. The sides are therefore exposed as two INDEPENDENT MONOCULAR
# streams - never as a stereo pair: rgb_0 is the left camera, rgb_1 the right one, each with its
# own frame list (rgb.csv / rgb_1.csv), calibration entry (identity T_BS, no rig extrinsics) and
# groundtruth (groundtruth.csv / groundtruth_1.csv), and the dataset only advertises 'mono' (on
# rgb_0). The HF repo names the sides "<date> LHS" / "<date> RHS" under raw/ (with a varying
# survey-tag prefix, e.g. "T2 29.10.25 LHS"), so each side is found by its token.
_SIDE_TOKENS: tuple[str, ...] = ("lhs", "rhs")  # stream index i -> rgb_<i>

_GROUNDTRUTH_HEADER = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

# "<base>_comb" sequences: the base survey's two streams merged into ONE denser monocular rgb_0
# (with a single rgb.csv / groundtruth.csv), built from the base sequence's already-processed
# rgb_0 + rgb_1 + groundtruth files - so requesting a *_comb sequence first pulls in its base.
#
# - Merge order: the two GoPros are not synchronized and their EXIF clocks are ~100 s apart (checked
#   on hb_20250710: EXIF says RHS started 102 s after LHS, the COLMAP poses say 2 s), so the true
#   RHS-vs-LHS shutter offset is estimated per survey from the two groundtruths instead: the rig
#   is rigid (every RHS pose sits ~0.6 m - the bar - from one LHS pose), so each RHS frame's
#   nearest LHS pose, projected onto the LHS trajectory, gives a continuous LHS frame index; the
#   median difference to the RHS frame index is the offset (fractional, e.g. +3.7 frames). RHS
#   frame k then gets timestamp (k + offset) / rgb_hz and the two streams are sorted together,
#   alternating L, R, L, R at roughly twice rgb_hz. The estimate (and its spread) is saved as
#   comb_alignment.json in the comb sequence folder.
# - Geometry: rgb_0 and rgb_1 are different GoPro bodies with different intrinsics and resized
#   sizes, and a comb sequence has ONE calibration - the base rgb_0 camera's. Both are undistorted
#   pinholes, so rgb_1 frames are warped exactly (an affine scale + shift, K0 * K1^-1) into the
#   rgb_0 intrinsics, then both streams are cropped to the common canvas = the rgb_0 image
#   rectangle intersected with the warped rgb_1 rectangle (mostly a right/bottom crop; a few
#   pixels may also go from the left/top where the warped rgb_1 doesn't reach - that shift is
#   folded into the comb principal point). fx/fy are rgb_0's unchanged; image_dimension is the
#   canvas size. Frames are re-encoded (crop/warp), as lossless PNGs named
#   lhs_<stem>.png / rhs_<stem>.png.
_COMB_SUFFIX = "_comb"
_COMB_FRAME_PREFIXES: tuple[str, ...] = ("lhs_", "rhs_")  # comb rgb_0 frame name prefix per stream
_COMB_ALIGNMENT_FILE = "comb_alignment.json"


def _is_comb(sequence_name: str) -> bool:
    return sequence_name.endswith(_COMB_SUFFIX)


def _base_sequence(sequence_name: str) -> str:
    return sequence_name[: -len(_COMB_SUFFIX)]


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in HFColmapDatasetMixin.IMAGE_SUFFIXES)


class HFColmapDatasetMixin:
    """Shared logic for a Hugging-Face-sourced dataset with a per-sequence COLMAP reconstruction
    and one or more independent monocular RGB streams (rgb_0, rgb_1, ...), used by both
    SonevaDataset (below) and SweetcoralsDataset (dataset_sweetcorals.py, which imports this
    mixin from here rather than utilities.py, since it's specific to this pair of datasets).

    Per-stream layout produced under the sequence folder, for stream index i:
      rgb_<i>_raw/        the flat raw download (ensure_hf_sequence_download; deleted at MINIMAL)
      rgb_<i>/            the resized frames
      rgb.csv | rgb_<i>.csv          one frame list per stream (columns ts_rgb_<i> (ns), path_rgb_<i>);
                                     streams are unsynchronized with unequal frame counts, so they
                                     never share rows in one csv
      groundtruth.csv | groundtruth_<i>.csv   COLMAP poses of that stream's camera (same world frame)
    Stream 0 uses the base class's canonical rgb_0 / rgb.csv / groundtruth.csv names. Only rgb_0 is
    what 'mono' runs on - the extra streams ship alongside for whoever wants them (there is no
    framework consumer of rgb_<i>.csv / groundtruth_<i>.csv yet).

    Concrete classes still define their own __init__, download_sequence_data,
    create_calibration_yaml and create_groundtruth_csv (the per-stream COLMAP lookups differ), plus
    two hooks the mixin's shared methods rely on:
      _stream_indices(sequence_name) -> list[int]   the stream indexes this sequence ships
      _remote_sequence_name(sequence_name) -> str   the sequence's top-level folder in the HF repo
                                                    (a dynamic HfApi lookup in SonevaDataset vs. a
                                                    hardcoded table in SweetcoralsDataset)
    The COLMAP binary-format parsing itself (read_colmap_cameras/read_colmap_images/
    world_to_camera_to_pose) lives in utilities.py instead, since it doesn't depend on either
    dataset - only fetching the file to parse does."""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

    # --- per-stream path helpers ---------------------------------------------------------------
    def rgb_stream_path(self, sequence_name: str, cam_idx: int) -> Path:
        """rgb_0 via the base-class helper, rgb_<i> alongside it."""
        if cam_idx == 0:
            return self.rgb_path(sequence_name)
        return self.sequence_path(sequence_name) / f"rgb_{cam_idx}"

    def rgb_raw_path(self, sequence_name: str, cam_idx: int) -> Path:
        return self.sequence_path(sequence_name) / f"rgb_{cam_idx}_raw"

    def rgb_stream_csv_path(self, sequence_name: str, cam_idx: int) -> Path:
        """rgb.csv via the base-class helper for stream 0, rgb_<i>.csv for the others."""
        if cam_idx == 0:
            return self.rgb_csv_path(sequence_name)
        return self.sequence_path(sequence_name) / f"rgb_{cam_idx}.csv"

    def groundtruth_stream_csv_path(self, sequence_name: str, cam_idx: int) -> Path:
        """groundtruth.csv via the base-class helper for stream 0, groundtruth_<i>.csv for the others."""
        if cam_idx == 0:
            return self.groundtruth_csv_path(sequence_name)
        return self.sequence_path(sequence_name) / f"groundtruth_{cam_idx}.csv"

    # --- DatasetVSLAMLAB hooks shared by both datasets -----------------------------------------
    def create_rgb_folder(self, sequence_name: str) -> None:
        for cam_idx in self._stream_indices(sequence_name):
            rgb_dir = self.rgb_stream_path(sequence_name, cam_idx)
            raw_dir = self.rgb_raw_path(sequence_name, cam_idx)
            if rgb_dir.exists() or not raw_dir.exists():
                continue
            self._resize_raw_folder(raw_dir, rgb_dir)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # One csv per stream: the streams are independent time-lapses with unequal frame counts
        # (see the module comment), so their rows can't be paired. Timestamps are synthetic
        # (frame index / rgb_hz) per stream - each stream starts at 0.
        for cam_idx in self._stream_indices(sequence_name):
            rgb_csv = self.rgb_stream_csv_path(sequence_name, cam_idx)
            if rgb_csv.exists():
                continue
            rgb_dir = self.rgb_stream_path(sequence_name, cam_idx)
            rgb_files = sorted(file_path.name for file_path in rgb_dir.iterdir() if file_path.is_file())
            rows = [[int(i * 1e9 / self.rgb_hz), f"{rgb_dir.name}/{filename}"] for i, filename in enumerate(rgb_files)]
            write_csv_rows(rgb_csv, [f"ts_rgb_{cam_idx} (ns)", f"path_rgb_{cam_idx}"], rows)

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
        # Deliberately narrow: only the rgb_<i>_raw/ folders (the resized-away raw images) are
        # ever removed, even at MINIMAL retention. Anything else a concrete subclass keeps around -
        # e.g. SonevaDataset's all_files_cache.json, a dataset-wide (not per-sequence) HfApi
        # listing cache reused across every sequence's download - is left alone here; override
        # this method if a subclass needs to clean up more.
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            for cam_idx in self._stream_indices(sequence_name):
                shutil.rmtree(self.rgb_raw_path(sequence_name, cam_idx), ignore_errors=True)

    # --- shared building blocks for the per-dataset hooks --------------------------------------
    def _resize_raw_folder(self, raw_dir: Path, rgb_dir: Path) -> None:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        target_size = None
        init_size = None
        for file_path in tqdm(sorted(raw_dir.iterdir()), desc=f"    resizing images ({raw_dir.name} -> {rgb_dir.name})"):
            if file_path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue

            if self.target_resolution is None:
                shutil.copy2(file_path, rgb_dir / file_path.name)
                continue

            with Image.open(file_path) as img:
                img.load()
                if target_size is None:
                    init_size = img.size
                    target_size = compute_scaled_size(img.size, self.target_resolution)

                if img.size != init_size:
                    print_warning(f"{file_path.name} {img.size} != {init_size}")

                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(rgb_dir / file_path.name)

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

    def _pinhole_rgb_calibration(self, sequence_name: str, camera_id: int, cam_idx: int = 0) -> dict[str, Any]:
        """Builds the rgb_<cam_idx> pinhole calibration dict for write_calibration_yaml from an
        already-resolved COLMAP camera_id. Shared by SonevaDataset/SweetcoralsDataset, which
        differ only in how camera_id itself gets resolved - see each class's own
        create_calibration_yaml. T_BS is identity for every stream: the rig's inter-camera
        extrinsics are neither published nor wanted (independent monocular streams, not stereo)."""
        cameras = read_colmap_cameras(self._fetch_colmap_file(sequence_name, "cameras.bin"))
        model_name, width, height, params = cameras[camera_id]

        if model_name == "SIMPLE_PINHOLE":
            f, cx, cy = params
            fx, fy = f, f
        else:  # PINHOLE
            fx, fy, cx, cy = params

        # Rescale intrinsics from COLMAP's reference image size to the resized rgb_<i> image size.
        # Deliberately NOT scale_intrinsics(..., (width, height), self.target_resolution): verified
        # against a real downloaded sequence that COLMAP's declared camera (width, height) here
        # (5456, 4082) does not equal rgb_0_raw's actual JPEG pixel size (5568, 4176) - COLMAP was
        # evidently run against a resized copy of the originals. compute_scaled_size(width, height)
        # would therefore predict a resize target (641, 479) that create_rgb_folder's real resize -
        # driven by the raw JPEG's actual size - never produces (640, 480). Reading the real,
        # already-resized rgb_<i> image's size directly sidesteps that mismatch. (The same holds
        # for the right camera: a HERO11 side is 5568x4872 raw against e.g. 4986x4354 in COLMAP.)
        rgb_dir = self.rgb_stream_path(sequence_name, cam_idx)
        with Image.open(next(rgb_dir.iterdir())) as img:
            resized_w, resized_h = img.size
        scale_x, scale_y = resized_w / width, resized_h / height

        return {
            "cam_name": f"rgb_{cam_idx}",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx * scale_x, fy * scale_y],
            "principal_point": [cx * scale_x, cy * scale_y],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }

    def _write_colmap_groundtruth(
        self, sequence_name: str, cam_idx: int, images: dict[str, tuple], to_colmap_name: Callable[[str], str | None],
    ) -> None:
        """Writes stream cam_idx's groundtruth csv from an already-parsed COLMAP images.bin dict:
        one row per rgb_<cam_idx> frame that is registered in the reconstruction, with the same
        synthetic frame-index timestamps create_rgb_csv assigns that stream. to_colmap_name maps a
        frame's filename to its COLMAP image name (None if unmapped) - an image_mapping.csv lookup
        for soneva, the identity for sweetcorals."""
        rgb_dir = self.rgb_stream_path(sequence_name, cam_idx)
        rgb_files = sorted(file_path.name for file_path in rgb_dir.iterdir() if file_path.is_file())

        rows = []
        for i, filename in enumerate(rgb_files):
            colmap_name = to_colmap_name(filename)
            if colmap_name is None or colmap_name not in images:
                continue

            _, qvec, tvec = images[colmap_name]
            tx, ty, tz, qx, qy, qz, qw = world_to_camera_to_pose(qvec, tvec)
            ts_ns = int(i * 1e9 / self.rgb_hz)
            rows.append([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        write_csv_rows(self.groundtruth_stream_csv_path(sequence_name, cam_idx), _GROUNDTRUTH_HEADER, rows)

    def _write_empty_groundtruth(self, sequence_name: str, cam_idx: int) -> None:
        """Header-only groundtruth for a stream with no published poses - written rather than left
        missing, like every other pose-less dataset in this repo."""
        write_csv_rows(self.groundtruth_stream_csv_path(sequence_name, cam_idx), _GROUNDTRUTH_HEADER, [])


class SonevaDataset(HFColmapDatasetMixin, DatasetVSLAMLAB):
    """Soneva dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "soneva") -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id = self.cfg["hf_repo_id"]

    def download_sequence_data(self, sequence_name: str) -> None:
        if _is_comb(sequence_name):
            self._ensure_base_sequence(_base_sequence(sequence_name))
            return

        remote_folder = self._remote_sequence_name(sequence_name)
        for cam_idx in self._stream_indices(sequence_name):
            remote_dir = f"{remote_folder}/raw/{self._side_subfolder(sequence_name, cam_idx)}"
            ensure_hf_sequence_download(
                self.hf_repo_id, [remote_dir], self.rgb_raw_path(sequence_name, cam_idx), token=hf_token(),
            )

    def create_rgb_folder(self, sequence_name: str) -> None:
        if _is_comb(sequence_name):
            self._create_comb_rgb_folder(sequence_name)
            return
        super().create_rgb_folder(sequence_name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        if _is_comb(sequence_name):
            self._create_comb_rgb_csv(sequence_name)
            return
        super().create_rgb_csv(sequence_name)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        if _is_comb(sequence_name):
            rgb = [self._comb_geometry(_base_sequence(sequence_name))["calibration"]]
        else:
            rgb = self._stream_calibrations(sequence_name)
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgb)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        if _is_comb(sequence_name):
            self._create_comb_groundtruth_csv(sequence_name)
            return

        # Both sides are registered in the same COLMAP reconstruction, so groundtruth.csv (rgb_0)
        # and groundtruth_1.csv (rgb_1) share one world frame - only their timestamps are each
        # stream's own synthetic frame index.
        images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))
        for cam_idx in self._stream_indices(sequence_name):
            raw_to_colmap = self._read_image_mapping(sequence_name, cam_idx)
            self._write_colmap_groundtruth(sequence_name, cam_idx, images, raw_to_colmap.get)

    def _stream_calibrations(self, sequence_name: str) -> list[dict[str, Any]]:
        """One rgb_<i> pinhole calibration dict per stream of a base survey, from COLMAP."""
        images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))

        rgb = []
        for cam_idx in self._stream_indices(sequence_name):
            # Any registered frame of this side tells us which COLMAP camera_id is its camera.
            # (Every survey reconstructs both sides jointly with one camera_id per side, but which
            # id is which side varies between surveys - hence the lookup instead of a constant.)
            raw_to_colmap = self._read_image_mapping(sequence_name, cam_idx)
            camera_id = next(images[name][0] for name in raw_to_colmap.values() if name in images)
            rgb.append(self._pinhole_rgb_calibration(sequence_name, camera_id, cam_idx))
        return rgb

    # --- *_comb sequences (see the module comment above _COMB_SUFFIX) ---------------------------
    def _ensure_base_sequence(self, base: str) -> None:
        """A *_comb sequence is built from its base survey's rgb_0 + rgb_1 layout, so pull that in
        first. download_sequence() alone isn't enough: a base processed before rgb_1 existed
        (rgb_0 only) already counts as 'available' to check_sequence_availability (mono needs no
        rgb_1), so it is topped up by running its download_process directly - idempotent for
        everything already in place, it only fetches/resizes the missing stream and rewrites the
        calibration.yaml / groundtruth files."""
        self.download_sequence(base)
        needed = [
            self.rgb_stream_path(base, 1),
            self.rgb_stream_csv_path(base, 1),
            self.groundtruth_stream_csv_path(base, 1),
        ]
        if not all(path.exists() for path in needed):
            print_info(f"{base}: rgb_1 stream missing (processed before rgb_1 existed) - completing it for {base}{_COMB_SUFFIX}")
            self.download_process(base)

    def _comb_geometry(self, base: str) -> dict[str, Any]:
        """The common canvas a comb sequence's frames live on, in the base rgb_0 camera's pixel
        coordinates: 'crop' = (x0, y0, x1, y1) box applied to rgb_0 frames, 'rgb1_to_rgb0' =
        (sx, sy) scale of the exact affine warp taking rgb_1 pixels into rgb_0 intrinsics
        (u0 = cx0 + sx * (u1 - cx1), sx = fx0 / fx1; likewise v), 'cal0'/'cal1' = the base
        streams' calibration dicts, 'calibration' = the comb rgb_0 dict (rgb_0's fx/fy, principal
        point shifted by the crop origin)."""
        cal0, cal1 = self._stream_calibrations(base)
        (fx0, fy0), (cx0, cy0) = cal0["focal_length"], cal0["principal_point"]
        (fx1, fy1), (cx1, cy1) = cal1["focal_length"], cal1["principal_point"]
        with Image.open(_image_files(self.rgb_stream_path(base, 0))[0]) as img:
            w0, h0 = img.size
        with Image.open(_image_files(self.rgb_stream_path(base, 1))[0]) as img:
            w1, h1 = img.size

        # rgb_1's image rectangle expressed in rgb_0 pixel coordinates, intersected with rgb_0's
        # own rectangle: the region both streams cover once rgb_1 is warped.
        sx, sy = fx0 / fx1, fy0 / fy1
        u_lo, u_hi = cx0 - sx * cx1, cx0 + sx * (w1 - cx1)
        v_lo, v_hi = cy0 - sy * cy1, cy0 + sy * (h1 - cy1)
        x0, y0 = math.ceil(max(0.0, u_lo)), math.ceil(max(0.0, v_lo))
        x1, y1 = math.floor(min(float(w0), u_hi)), math.floor(min(float(h0), v_hi))
        if x1 - x0 < w0 // 2 or y1 - y0 < h0 // 2:
            raise RuntimeError(f"{base}: rgb_0 and rgb_1 overlap too little to merge ({x1 - x0}x{y1 - y0} of {w0}x{h0})")

        calibration = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx0, fy0],
            "principal_point": [cx0 - x0, cy0 - y0],
            # Two interleaved rgb_hz streams: the comb sequence's actual frame rate.
            "fps": float(self.rgb_hz * len(_COMB_FRAME_PREFIXES)),
            "T_BS": np.eye(4),
        }
        return {"crop": (x0, y0, x1, y1), "rgb1_to_rgb0": (sx, sy), "cal0": cal0, "cal1": cal1, "calibration": calibration}

    def _create_comb_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return
        base = _base_sequence(sequence_name)
        geometry = self._comb_geometry(base)
        x0, y0, x1, y1 = geometry["crop"]
        size = (x1 - x0, y1 - y0)
        rgb_path.mkdir(parents=True, exist_ok=True)

        # rgb_0: a pure crop (its intrinsics are the comb calibration, up to the crop origin).
        for frame in tqdm(_image_files(self.rgb_stream_path(base, 0)), desc=f"    cropping rgb_0 -> {sequence_name}/rgb_0 ({_COMB_FRAME_PREFIXES[0]}*)"):
            with Image.open(frame) as img:
                img.crop((x0, y0, x1, y1)).save(rgb_path / f"{_COMB_FRAME_PREFIXES[0]}{frame.stem}.png")

        # rgb_1: the exact affine warp into rgb_0 intrinsics, sampled directly on the comb canvas.
        # PIL's AFFINE maps each output pixel (x, y) to input (a*x + b*y + c, d*x + e*y + f) on
        # pixel-center-consistent continuous coordinates (checked: an integer translation moves
        # pixels exactly). Output x sits at rgb_0 coordinate u0 = x + x0, whose rgb_1 coordinate
        # is u1 = cx1 + (u0 - cx0) / sx.
        sx, sy = geometry["rgb1_to_rgb0"]
        (cx0, cy0), (cx1, cy1) = geometry["cal0"]["principal_point"], geometry["cal1"]["principal_point"]
        affine = (1.0 / sx, 0.0, cx1 + (x0 - cx0) / sx, 0.0, 1.0 / sy, cy1 + (y0 - cy0) / sy)
        for frame in tqdm(_image_files(self.rgb_stream_path(base, 1)), desc=f"    warping rgb_1 -> {sequence_name}/rgb_0 ({_COMB_FRAME_PREFIXES[1]}*)"):
            with Image.open(frame) as img:
                img.transform(size, Image.AFFINE, affine, resample=Image.BICUBIC).save(rgb_path / f"{_COMB_FRAME_PREFIXES[1]}{frame.stem}.png")

    def _comb_alignment(self, sequence_name: str) -> dict[str, Any]:
        """The RHS-vs-LHS shutter offset of a comb sequence's base survey, in LHS frames (see the
        module comment above _COMB_SUFFIX), estimated from the two groundtruths and cached as
        comb_alignment.json in the comb sequence folder."""
        alignment_file = self.sequence_path(sequence_name) / _COMB_ALIGNMENT_FILE
        if alignment_file.exists():
            with open(alignment_file, "r", encoding="utf-8") as f:
                return json.load(f)

        base = _base_sequence(sequence_name)
        frame_ns = 1e9 / self.rgb_hz
        gt0 = pd.read_csv(self.groundtruth_stream_csv_path(base, 0))
        gt1 = pd.read_csv(self.groundtruth_stream_csv_path(base, 1))
        idx0 = np.rint(gt0["ts (ns)"].to_numpy() / frame_ns).astype(int)
        idx1 = np.rint(gt1["ts (ns)"].to_numpy() / frame_ns).astype(int)
        p0 = gt0[["tx (m)", "ty (m)", "tz (m)"]].to_numpy()
        p1 = gt1[["tx (m)", "ty (m)", "tz (m)"]].to_numpy()
        if len(p0) < 3 or len(p1) < 1:
            raise RuntimeError(f"{base}: not enough groundtruth poses to align the two streams for {sequence_name}")

        # Nearest LHS pose of every RHS frame; a rigid rig puts it one bar-length away. Frames whose
        # nearest pose is much farther than typical (LHS gaps in the reconstruction) are left out.
        distances = np.linalg.norm(p1[:, None, :] - p0[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        nearest_distance = distances.min(axis=1)
        typical = float(np.median(nearest_distance))
        matched = np.nonzero(nearest_distance <= 1.5 * typical)[0]

        # Project the RHS position onto the LHS trajectory around its nearest pose for a continuous
        # (sub-frame) LHS index; the offset is that index minus the RHS frame index.
        offsets = []
        for k in matched:
            j = nearest[k]
            jm, jp = max(j - 1, 0), min(j + 1, len(p0) - 1)
            segment = p0[jp] - p0[jm]
            length_sq = float(segment @ segment)
            t = float(np.clip((p1[k] - p0[jm]) @ segment / length_sq, 0.0, 1.0)) if length_sq > 0 else 0.5
            continuous_index = idx0[jm] + t * (idx0[jp] - idx0[jm])
            offsets.append(continuous_index - idx1[k])
        offsets = np.asarray(offsets)
        median_offset = float(np.median(offsets))
        # The per-frame estimates have a tight mode plus tails (the diver hovering at the start,
        # turns between lanes, where the nearest LHS pose is ambiguous), so the median is what's
        # used and the mode's weight - not a spread - is the sanity measure (checked on
        # hb_20250710: ~80% of frames within half a frame of the median, no dependence on heading).
        consistent = float(np.mean(np.abs(offsets - median_offset) <= 0.5))

        alignment = {
            "base_sequence": base,
            "rhs_frame_offset": median_offset,  # RHS frame k <-> LHS frame k + offset
            "fraction_within_half_frame": consistent,
            "n_rhs_frames_matched": int(len(offsets)),
            "n_rhs_frames": int(len(p1)),
            "median_nearest_lhs_distance_m": typical,
        }
        summary = (
            f"{sequence_name}: RHS frames offset by {median_offset:+.2f} LHS frames "
            f"({consistent:.0%} of {len(offsets)}/{len(p1)} frames within half a frame, nearest LHS pose {typical:.2f} m)"
        )
        if consistent < 0.5:
            print_warning(summary + " - offset is poorly supported; the interleaving may be out of order in places")
        else:
            print_info(summary)
        self.sequence_path(sequence_name).mkdir(parents=True, exist_ok=True)
        with open(alignment_file, "w", encoding="utf-8") as f:
            json.dump(alignment, f, indent=2)
        return alignment

    def _comb_timestamp_ns(self, frame_index: int, cam_idx: int, rhs_frame_offset: float) -> int:
        """Comb timestamp of frame frame_index of stream cam_idx: LHS keeps its base frame-index
        clock, RHS is shifted by the estimated offset. The +cam_idx ns breaks the exact tie an
        integer-valued offset would otherwise produce, so the comb clock is strictly increasing."""
        offset = rhs_frame_offset if cam_idx == 1 else 0.0
        return int(round((frame_index + offset) * 1e9 / self.rgb_hz)) + cam_idx

    def _create_comb_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return
        base = _base_sequence(sequence_name)
        rhs_frame_offset = self._comb_alignment(sequence_name)["rhs_frame_offset"]

        rows: list[list[Any]] = []
        for cam_idx, prefix in enumerate(_COMB_FRAME_PREFIXES):
            # Same sorted order as the base stream's own csv/groundtruth, so frame_index matches.
            for frame_index, frame in enumerate(_image_files(self.rgb_stream_path(base, cam_idx))):
                ts_ns = self._comb_timestamp_ns(frame_index, cam_idx, rhs_frame_offset)
                rows.append([ts_ns, f"{self.rgb_path(sequence_name).name}/{prefix}{frame.stem}.png"])
        rows.sort(key=lambda row: row[0])
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def _create_comb_groundtruth_csv(self, sequence_name: str) -> None:
        base = _base_sequence(sequence_name)
        rhs_frame_offset = self._comb_alignment(sequence_name)["rhs_frame_offset"]
        frame_ns = 1e9 / self.rgb_hz

        rows: list[list[Any]] = []
        for cam_idx in range(len(_COMB_FRAME_PREFIXES)):
            gt = pd.read_csv(self.groundtruth_stream_csv_path(base, cam_idx))
            for row in gt.itertuples(index=False):
                frame_index = int(round(row[0] / frame_ns))
                rows.append([self._comb_timestamp_ns(frame_index, cam_idx, rhs_frame_offset), *row[1:]])
        rows.sort(key=lambda row: row[0])
        write_csv_rows(self.groundtruth_csv_path(sequence_name), _GROUNDTRUTH_HEADER, rows)

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

    def _stream_indices(self, sequence_name: str) -> list[int]:
        """Every survey published so far ships both sides; a side missing from the repo listing
        is simply not a stream (rather than an error), so a one-camera survey would still work.
        A *_comb sequence has the single merged rgb_0 stream."""
        if _is_comb(sequence_name):
            return [0]
        return [
            cam_idx for cam_idx in range(len(_SIDE_TOKENS))
            if self._find_side_subfolder(sequence_name, cam_idx) is not None
        ]

    def _find_side_subfolder(self, sequence_name: str, cam_idx: int) -> str | None:
        """The raw/ subfolder of stream cam_idx's side (e.g. 'T2 29.10.25 LHS' for rgb_0), or
        None if the repo has no folder carrying that side's token for this sequence."""
        prefix = f"{self._remote_sequence_name(sequence_name)}/raw/"
        subfolders = {f[len(prefix):].split("/")[0] for f in self._all_repo_files() if f.startswith(prefix)}
        token = _SIDE_TOKENS[cam_idx]
        return next((subfolder for subfolder in sorted(subfolders) if token in subfolder.lower()), None)

    def _side_subfolder(self, sequence_name: str, cam_idx: int) -> str:
        subfolder = self._find_side_subfolder(sequence_name, cam_idx)
        if subfolder is None:
            raise FileNotFoundError(f"No {_SIDE_TOKENS[cam_idx].upper()} subfolder found for sequence {sequence_name}")
        return subfolder

    def _read_image_mapping(self, sequence_name: str, cam_idx: int) -> dict[str, str]:
        """raw filename -> COLMAP image name, restricted to stream cam_idx's side (the repo's
        image_mapping.csv covers both sides in one file, keyed by 'raw/<side folder>/<file>')."""
        path = self._fetch_colmap_file(sequence_name, "image_mapping.csv")
        subfolder = self._side_subfolder(sequence_name, cam_idx)

        mapping: dict[str, str] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                raw_path = row["raw_path"]
                if raw_path.split("/")[1] != subfolder:
                    continue
                mapping[Path(raw_path).name] = row["colmap_image"]
        return mapping
