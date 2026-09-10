"""
Module: VSLAM-LAB - Datasets - dataset_endomapper_sim.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5.1)
- Version: 1.0
- Created: 2026-09-10
- License: GPLv3 License
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Final

# OpenCV's EXR decoder is disabled unless this is set before the first EXR is decoded.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB  # noqa: E402
from utilities import compute_scaled_size, make_printers, scale_intrinsics, write_csv_rows  # noqa: E402

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

_SIMULATED_DIR: Final = "Simulated Sequences"  # under raw_data_path
_RAW_ZIP_LINK: Final = "raw.zip"  # symlink onto raw_data_path/Simulated Sequences/Seq_<N>.zip
_TEXT_FILES: Final = ("calibration.txt", "info.txt", "trajectory.csv")  # copied out of the zip

_RGB_MEMBER_RE: Final = re.compile(r"(?:^|/)image_(?P<index>\d+)\.png$")
_DEPTH_MEMBER_RE: Final = re.compile(r"(?:^|/)aov_image_(?P<index>\d+)\.exr$")
_DEPTH_CHANNEL: Final = 2  # the EXR's R channel (cv2 decodes BGRA); G/B are zero, A is one
_DM_TO_M: Final = 0.1  # info.txt: depth and trajectory are in decimeters

_GROUNDTRUTH_HEADER: Final = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def _frame_name(frame_index: int) -> str:
    return f"{frame_index:04d}.png"


class EndomapperSimDataset(DatasetVSLAMLAB):
    """EndoMapper simulated colon dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "endomapper-sim") -> None:
        super().__init__(dataset_name)

        # All sequences are local (scalar in the yaml): the raw folder is the only source, entered
        # through raw_data_path.
        self.sequence_location = self.cfg["sequence_location"]
        self.raw_data_path = Path(self.cfg["raw_data_path"])
        # Depth in meters = depth_0 pixel value / depth_factor (see the yaml).
        self.depth_factor: float = float(self.cfg["depth_factor"])

    def download_sequence_data(self, sequence_name: str) -> None:
        raw_link = self._raw_zip(sequence_name)
        if not (raw_link.is_symlink() or raw_link.exists()):
            raw_zip = self.raw_data_path / _SIMULATED_DIR / f"{sequence_name}.zip"
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

        # The three small text files (calibration, deformation/units info, trajectory), copied
        # out so the sequence folder is self-contained; the frames stay in the zip.
        missing = [name for name in _TEXT_FILES if not (self.sequence_path(sequence_name) / name).is_file()]
        if not missing:
            return
        with self._open_raw_zip(sequence_name) as zf:
            members = {Path(name).name: name for name in zf.namelist() if Path(name).name in _TEXT_FILES}
            for name in missing:
                (self.sequence_path(sequence_name) / name).write_bytes(zf.read(members[name]))

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path, depth_path = self.rgb_path(sequence_name), self.depth_path(sequence_name)
        if rgb_path.exists() and depth_path.exists():
            return

        # Built in sibling temp folders and renamed once complete, so a crash midway can't leave
        # a partial rgb_0/depth_0 that later looks finished.
        tmp_rgb, tmp_depth = rgb_path.with_name("rgb_0.tmp"), depth_path.with_name("depth_0.tmp")
        for tmp in (tmp_rgb, tmp_depth):
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)

        with self._open_raw_zip(sequence_name) as zf:
            rgb_members, depth_members = self._frame_members(sequence_name, zf)
            frames = sorted(set(rgb_members) & set(depth_members))
            target_size = None
            for index in tqdm(frames, desc=f"    resizing frames -> {rgb_path.name}/{depth_path.name}"):
                with Image.open(BytesIO(zf.read(rgb_members[index]))) as img:
                    if target_size is None:
                        target_size = compute_scaled_size(img.size, self.target_resolution)
                    rgb = img.convert("RGB")  # drop the constant alpha channel
                    if self.target_resolution is not None:
                        rgb = rgb.resize(target_size, Image.Resampling.LANCZOS)
                    rgb.save(tmp_rgb / _frame_name(index))

                depth_dm = self._decode_depth(zf.read(depth_members[index]))
                if self.target_resolution is not None:
                    # Depth: nearest-neighbor only, never an interpolating resample.
                    depth_dm = cv2.resize(depth_dm, target_size, interpolation=cv2.INTER_NEAREST)
                depth_px = np.round(depth_dm.astype(np.float64) * _DM_TO_M * self.depth_factor)
                cv2.imwrite(str(tmp_depth / _frame_name(index)), np.clip(depth_px, 0, np.iinfo(np.uint16).max).astype(np.uint16))

        shutil.rmtree(rgb_path, ignore_errors=True)
        shutil.rmtree(depth_path, ignore_errors=True)
        tmp_rgb.rename(rgb_path)
        tmp_depth.rename(depth_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return
        # rgb_0 and depth_0 are rendered from the same camera at the same instants (one row per
        # frame, same timestamp), and create_rgb_folder only kept frames present in both.
        rgb_path, depth_path = self.rgb_path(sequence_name), self.depth_path(sequence_name)
        rows = []
        for frame in self._rgb_frames(sequence_name):
            ts_ns = self._ts_ns(int(frame.stem))
            rows.append([ts_ns, f"{rgb_path.name}/{frame.name}", ts_ns, f"{depth_path.name}/{frame.name}"])
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # calibration.txt: "fx: <v>", "fy:", "cx:", "cy:", blank, "cols: 960", "rows: 720" - the
        # rendered pinhole at native size, rescaled to what create_rgb_folder produced.
        values = self._read_calibration_txt(sequence_name)
        native_size = (int(values["cols"]), int(values["rows"]))
        self._check_calibration_resolution(sequence_name, native_size)
        focal_length, principal_point = scale_intrinsics(
            (values["fx"], values["fy"]), (values["cx"], values["cy"]), native_size, self.target_resolution
        )
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
        # trajectory.csv: "tX;tY;tZ;rX;rY;rZ;rW;time(s)", one row per rendered frame at 1/30 s,
        # position in dm (info.txt) -> m here, quaternion written as given (rX rY rZ rW = qx qy qz
        # qw). Seq_0's file ends in a truncated row (a cut-off upload) - malformed rows are
        # skipped and reported, as are frames left without a pose.
        poses: dict[int, list[float]] = {}
        skipped = 0
        with open(self.sequence_path(sequence_name) / "trajectory.csv", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(";")
                try:
                    tx, ty, tz, qx, qy, qz, qw, t = (float(v) for v in parts)
                except ValueError:
                    skipped += 1
                    continue
                poses[int(round(t * self.rgb_hz))] = [tx * _DM_TO_M, ty * _DM_TO_M, tz * _DM_TO_M, qx, qy, qz, qw]
        if skipped:
            print_warning(f"{sequence_name}: skipped {skipped} malformed trajectory.csv row(s)")

        frames = [int(frame.stem) for frame in self._rgb_frames(sequence_name)]
        rows = [[self._ts_ns(index), *poses[index]] for index in frames if index in poses]
        if len(rows) < len(frames):
            print_warning(f"{sequence_name}: {len(frames) - len(rows)} of {len(frames)} frames have no pose in trajectory.csv")
        write_csv_rows(self.groundtruth_csv_path(sequence_name), _GROUNDTRUTH_HEADER, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberate no-op at every retention tier: raw.zip is a symlink onto the raw folder (the
        # only copy of the frames), and the three copied text files are the calibration/pose
        # sources, a few KB in total.
        return

    # --- helpers, all recomputed from sequence_name (no per-sequence state on self) -------------
    def _raw_zip(self, sequence_name: str) -> Path:
        return self.sequence_path(sequence_name) / _RAW_ZIP_LINK

    def _open_raw_zip(self, sequence_name: str) -> zipfile.ZipFile:
        raw_link = self._raw_zip(sequence_name)
        if not raw_link.is_file():
            raise FileNotFoundError(
                f"Raw zip for '{sequence_name}' not found at {raw_link} (sequence marked as 'local'): run "
                f"download_sequence_data with the raw folder in place, and keep it in place while processing."
            )
        return zipfile.ZipFile(raw_link)

    def _frame_members(self, sequence_name: str, zf: zipfile.ZipFile) -> tuple[dict[int, str], dict[int, str]]:
        """(frame index -> rgb member, frame index -> depth member) of the zip, reporting frames
        that have only one of the two (dropped: rgbd needs the pair)."""
        rgb: dict[int, str] = {}
        depth: dict[int, str] = {}
        for name in zf.namelist():
            if match := _RGB_MEMBER_RE.search(name):
                rgb[int(match["index"])] = name
            elif match := _DEPTH_MEMBER_RE.search(name):
                depth[int(match["index"])] = name
        for label, only in (("depth map", set(rgb) - set(depth)), ("RGB image", set(depth) - set(rgb))):
            if only:
                print_warning(f"{sequence_name}: dropping {len(only)} frames with no {label} in the zip")
        return rgb, depth

    @staticmethod
    def _decode_depth(exr_bytes: bytes) -> np.ndarray:
        """The EXR's depth channel as a float32 (H, W) array, in the source units (dm)."""
        image = cv2.imdecode(np.frombuffer(exr_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("cv2 could not decode a depth EXR (is OPENCV_IO_ENABLE_OPENEXR set?)")
        return image[..., _DEPTH_CHANNEL] if image.ndim == 3 else image

    def _rgb_frames(self, sequence_name: str) -> list[Path]:
        rgb_path = self.rgb_path(sequence_name)
        return sorted(p for p in rgb_path.iterdir() if p.is_file() and p.suffix.lower() == ".png")

    def _ts_ns(self, frame_index: int) -> int:
        """Frame index -> timestamp in ns at the render rate (no timestamps ship with the frames)."""
        return int(round(frame_index * 1e9 / self.rgb_hz))

    def _read_calibration_txt(self, sequence_name: str) -> dict[str, float]:
        values: dict[str, float] = {}
        with open(self.sequence_path(sequence_name) / "calibration.txt", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    values[key.strip()] = float(value)
        missing = {"fx", "fy", "cx", "cy", "cols", "rows"} - set(values)
        if missing:
            raise ValueError(f"{sequence_name}: calibration.txt lacks {sorted(missing)}")
        return values

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
