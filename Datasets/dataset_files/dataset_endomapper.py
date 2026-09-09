"""
Module: VSLAM-LAB - Datasets - dataset_endomapper.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5.1)
- Version: 1.0
- Created: 2026-09-10
- License: GPLv3 License
"""

from __future__ import annotations

import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, make_printers, read_colmap_images, scale_intrinsics, world_to_camera_to_pose, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# Seq_<NNN> is a whole procedure, Seq_<NNN>_<MM> its COLMAP sub-model MM (see the yaml).
_SEQUENCE_NAME_RE: Final = re.compile(r"^(?P<procedure>Seq_\d{3})(?:_(?P<submodel>\d{2}))?$")
# Video frames inside the zip's img_train/ folder, named by video frame index.
_FRAME_RE: Final = re.compile(r"^out(?P<index>\d+)\.png$")

# Per-sequence files download_sequence_data leaves in the sequence folder.
_RAW_ZIP_LINK: Final = "raw.zip"  # symlink onto raw_data_path/Seq_<NNN>.zip
_COLMAP_DIR: Final = "colmap"  # holds the sequence's sub-model images.bin

# The calibu camera type EndoMapper's geometrical xml declares: fx, fy, cx, cy, k1, k2, k3, k4
# (Kannala-Brandt), i.e. VSLAM-LAB's pinhole + equid4.
_KB4_CAMERA_TYPE: Final = "calibu_fu_fv_u0_v0_kb4"

_GROUNDTRUTH_HEADER: Final = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def _split_sequence_name(sequence_name: str) -> tuple[str, int | None]:
    """'Seq_001_11' -> ('Seq_001', 11); 'Seq_001' -> ('Seq_001', None)."""
    match = _SEQUENCE_NAME_RE.match(sequence_name)
    if match is None:
        raise ValueError(f"Unknown endomapper sequence '{sequence_name}' - expected Seq_<NNN> or Seq_<NNN>_<MM>")
    submodel = match["submodel"]
    return match["procedure"], (int(submodel) if submodel is not None else None)


def _frame_index(colmap_name: str) -> int:
    """'out7199.png' (the zip's / COLMAP's frame name) -> 7199 (video frame index)."""
    match = _FRAME_RE.match(Path(colmap_name).name)
    if match is None:
        raise ValueError(f"Unexpected frame name '{colmap_name}' - expected out<N>.png")
    return int(match["index"])


def _rgb_name(frame_index: int) -> str:
    """rgb_0 file name of a video frame: zero-padded so lexicographic order is video order."""
    return f"{frame_index:06d}.png"


def _colmap_name(rgb_name: str) -> str:
    """Inverse of _rgb_name: '007199.png' -> 'out7199.png'."""
    return f"out{int(Path(rgb_name).stem)}.png"


def _zip_root(zf: zipfile.ZipFile) -> str:
    """The zip's single top-level folder (Seq_001.zip: '33', the procedure's internal id)."""
    roots = {name.split("/", 1)[0] for name in zf.namelist() if "/" in name}
    if len(roots) != 1:
        raise ValueError(f"{zf.filename}: expected one top-level folder, found {sorted(roots)}")
    return roots.pop()


class EndomapperDataset(DatasetVSLAMLAB):
    """EndoMapper endoscopy dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "endomapper") -> None:
        super().__init__(dataset_name)

        # All sequences are local (scalar in the yaml): the raw folder is the only source, entered
        # through raw_data_path.
        self.sequence_location = self.cfg["sequence_location"]
        self.raw_data_path = Path(self.cfg["raw_data_path"])

    def download_sequence_data(self, sequence_name: str) -> None:
        procedure, _ = _split_sequence_name(sequence_name)
        raw_link = self._raw_zip(sequence_name)
        if not (raw_link.is_symlink() or raw_link.exists()):
            raw_zip = self.raw_data_path / f"{procedure}.zip"
            if not raw_zip.is_file():
                print_info(
                    f"Sequence '{sequence_name}' is marked as 'local'. Its raw zip was not found at {raw_zip} - "
                    f"place it there, or point raw_data_path in dataset_{self.dataset_name}.yaml at your copy of "
                    f"the EndoMapper raw folder."
                )
                return
            self.sequence_path(sequence_name).mkdir(parents=True, exist_ok=True)
            # Absolute target on purpose: the zip lives outside the benchmark folder.
            os.symlink(raw_zip.resolve(), raw_link)

        # The endoscope calibration: Seq_<NNN>_info.json names the endoscope, whose geometrical xml
        # holds the intrinsics. Both are tiny - copied in so the sequence folder is self-contained.
        for name in (self._info_json_name(procedure), self._geometrical_xml_name(sequence_name)):
            target = self.sequence_path(sequence_name) / name
            if not target.exists():
                shutil.copy2(self.raw_data_path / name, target)

        # This sequence's COLMAP poses: sparse/<M>/images.bin of its sub-model (Seq_<NNN>: the
        # largest one). Extracted into a temp folder and renamed once complete, so a crash midway
        # can't leave a colmap/ that later looks finished.
        colmap_dir = self._colmap_dir(sequence_name)
        if (colmap_dir / "images.bin").is_file():
            return
        tmp_dir = colmap_dir.with_name(colmap_dir.name + ".tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        with self._open_raw_zip(sequence_name) as zf:
            member = f"{_zip_root(zf)}/sparse/{self._submodel_index(sequence_name)}/images.bin"
            (tmp_dir / "images.bin").write_bytes(zf.read(member))
        tmp_dir.rename(colmap_dir)

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return

        tmp_path = rgb_path.with_name(rgb_path.name + ".tmp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True)

        target_size = None
        init_size = None
        with self._open_raw_zip(sequence_name) as zf:
            root = _zip_root(zf)
            for colmap_name in tqdm(self._frame_names(sequence_name, zf), desc=f"    resizing frames -> {rgb_path.name}"):
                data = zf.read(f"{root}/img_train/{colmap_name}")
                out_path = tmp_path / _rgb_name(_frame_index(colmap_name))
                if self.target_resolution is None:
                    out_path.write_bytes(data)  # the original PNG, byte for byte
                    continue
                with Image.open(BytesIO(data)) as img:
                    if target_size is None:
                        init_size = img.size
                        target_size = compute_scaled_size(img.size, self.target_resolution)
                    if img.size != init_size:
                        print_warning(f"{colmap_name} {img.size} != {init_size}")
                    img.resize(target_size, Image.Resampling.LANCZOS).save(out_path)

        tmp_path.rename(rgb_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return
        rgb_path = self.rgb_path(sequence_name)
        rows = [[self._ts_ns(int(p.stem)), f"{rgb_path.name}/{p.name}"] for p in self._rgb_frames(sequence_name)]
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # The endoscope's official EndoMapper calibration (geometrical xml), at the native
        # 1440x1080, rescaled to the size create_rgb_folder produced. Note the COLMAP sub-models
        # were reconstructed with a slightly different KB4 set (their cameras.bin: fx 717.21 vs
        # 717.69 here, etc.) - the published calibration is the one written.
        width, height, params = self._read_geometrical_xml(sequence_name)
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        self._check_calibration_resolution(sequence_name, (width, height))
        focal_length, principal_point = scale_intrinsics((fx, fy), (cx, cy), (width, height), self.target_resolution)
        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "distortion_type": "equid4",
            "distortion_coefficients": [k1, k2, k3, k4],
            "focal_length": focal_length,
            "principal_point": principal_point,
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # COLMAP poses of this sequence's sub-model (camera-in-world, the sub-model's own frame and
        # scale), one row per rgb_0 frame registered in it - every frame of a Seq_<NNN>_<MM>
        # sequence, only the largest sub-model's frames of a whole-procedure Seq_<NNN>.
        images = read_colmap_images(self._colmap_dir(sequence_name) / "images.bin")
        rows = []
        for frame in self._rgb_frames(sequence_name):
            registered = images.get(_colmap_name(frame.name))
            if registered is None:
                continue
            _, qvec, tvec = registered
            rows.append([self._ts_ns(int(frame.stem)), *world_to_camera_to_pose(qvec, tvec)])
        n_frames = len(self._rgb_frames(sequence_name))
        if len(rows) < n_frames:
            print_info(
                f"{sequence_name}: COLMAP poses (sub-model {self._submodel_index(sequence_name)}) cover "
                f"{len(rows)} of {n_frames} frames"
            )
        write_csv_rows(self.groundtruth_csv_path(sequence_name), _GROUNDTRUTH_HEADER, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        # colmap/images.bin is a copy of a zip member, fully turned into groundtruth.csv - gone at
        # STANDARD, re-extracted from the zip on demand. raw.zip is a symlink onto the raw folder
        # (the only copy of the frames) and is never deleted, at any tier.
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(self._colmap_dir(sequence_name), ignore_errors=True)

    # --- helpers, all recomputed from sequence_name (no per-sequence state on self) -------------
    def _raw_zip(self, sequence_name: str) -> Path:
        return self.sequence_path(sequence_name) / _RAW_ZIP_LINK

    def _colmap_dir(self, sequence_name: str) -> Path:
        return self.sequence_path(sequence_name) / _COLMAP_DIR

    def _open_raw_zip(self, sequence_name: str) -> zipfile.ZipFile:
        raw_link = self._raw_zip(sequence_name)
        if not raw_link.is_file():
            raise FileNotFoundError(
                f"Raw zip for '{sequence_name}' not found at {raw_link} (sequence marked as 'local'): run "
                f"download_sequence_data with the raw folder in place, and keep it in place while processing."
            )
        return zipfile.ZipFile(raw_link)

    def _submodel_index(self, sequence_name: str) -> int:
        """The COLMAP sub-model this sequence's poses come from: its own for Seq_<NNN>_<MM>, the
        largest cluster (most frames in cluster_list/<M>.txt, lowest M on a tie) for Seq_<NNN>."""
        _, submodel = _split_sequence_name(sequence_name)
        if submodel is not None:
            return submodel
        with self._open_raw_zip(sequence_name) as zf:
            prefix = f"{_zip_root(zf)}/cluster_list/"
            sizes = [
                (len(zf.read(name).split()), int(Path(name).stem))
                for name in zf.namelist()
                if name.startswith(prefix) and name.endswith(".txt")
            ]
        if not sizes:
            raise ValueError(f"{sequence_name}: no cluster_list/*.txt in {self._raw_zip(sequence_name)}")
        return max(sizes, key=lambda item: (item[0], -item[1]))[1]

    def _frame_names(self, sequence_name: str, zf: zipfile.ZipFile) -> list[str]:
        """COLMAP frame names ('out<N>.png') that make up rgb_0, in video order: every img_train
        frame for Seq_<NNN>, the frames registered in the sub-model's images.bin for Seq_<NNN>_<MM>."""
        _, submodel = _split_sequence_name(sequence_name)
        if submodel is None:
            prefix = f"{_zip_root(zf)}/img_train/"
            names = [Path(name).name for name in zf.namelist() if name.startswith(prefix) and name.endswith(".png")]
        else:
            names = list(read_colmap_images(self._colmap_dir(sequence_name) / "images.bin"))
        return sorted(names, key=_frame_index)

    def _rgb_frames(self, sequence_name: str) -> list[Path]:
        rgb_path = self.rgb_path(sequence_name)
        return sorted(p for p in rgb_path.iterdir() if p.is_file() and p.suffix.lower() == ".png")

    def _ts_ns(self, frame_index: int) -> int:
        """Video frame index -> timestamp in ns, at the nominal capture rate (no timestamps ship)."""
        return int(round(frame_index * 1e9 / self.rgb_hz))

    @staticmethod
    def _info_json_name(procedure: str) -> str:
        return f"{procedure}_info.json"

    def _geometrical_xml_name(self, sequence_name: str) -> str:
        """'Endoscope_<NN>_geometrical.xml' for this sequence's endoscope, from the procedure's info
        json - read from the sequence folder if already copied in, else from the raw folder."""
        procedure, _ = _split_sequence_name(sequence_name)
        for folder in (self.sequence_path(sequence_name), self.raw_data_path):
            info_json = folder / self._info_json_name(procedure)
            if info_json.is_file():
                with open(info_json, encoding="utf-8") as f:
                    return f"Endoscope_{int(json.load(f)['endoscope_number']):02d}_geometrical.xml"
        raise FileNotFoundError(f"{sequence_name}: {self._info_json_name(procedure)} not found in {self.raw_data_path}")

    def _read_geometrical_xml(self, sequence_name: str) -> tuple[int, int, list[float]]:
        """(width, height, [fx, fy, cx, cy, k1, k2, k3, k4]) from the endoscope's calibu xml."""
        xml_path = self.sequence_path(sequence_name) / self._geometrical_xml_name(sequence_name)
        camera_model = ET.parse(xml_path).getroot().find("./camera/camera_model")
        if camera_model is None or camera_model.get("type") != _KB4_CAMERA_TYPE:
            found = None if camera_model is None else camera_model.get("type")
            raise ValueError(f"{xml_path}: expected camera_model type '{_KB4_CAMERA_TYPE}', found {found!r}")
        params = [float(v) for v in camera_model.findtext("params").strip(" []\n\t").split(";")]
        if len(params) != 8:
            raise ValueError(f"{xml_path}: expected 8 kb4 params, found {len(params)}")
        return int(camera_model.findtext("width")), int(camera_model.findtext("height")), params

    def _check_calibration_resolution(self, sequence_name: str, native_size: tuple[int, int]) -> None:
        """Warn if the first rgb_0 frame's size differs from what the calibration is being scaled
        to - the written intrinsics would then describe the wrong image size (#99)."""
        frames = self._rgb_frames(sequence_name)
        if not frames:
            return
        expected_size = compute_scaled_size(native_size, self.target_resolution)
        with Image.open(frames[0]) as img:
            if img.size != expected_size:
                print_warning(
                    f"{sequence_name}: rgb_0/{frames[0].name} is {img.size}, but the calibration is scaled for "
                    f"{expected_size} - intrinsics may describe the wrong image size."
                )
