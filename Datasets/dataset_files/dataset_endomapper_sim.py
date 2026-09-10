"""
Module: VSLAM-LAB - Datasets - dataset_endomapper_sim.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2026-09-10
- Updated: 2026-09-10
- License: GPLv3 License
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Final

# OpenCV's EXR decoder is disabled unless this is set before the first EXR is decoded.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB  # noqa: E402
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue  # noqa: E402
from utilities import (  # noqa: E402
    compute_scaled_size,
    make_printers,
    scale_intrinsics,
    synapse_client,
    synapse_download_folder,
    synapse_resolve_path,
    write_csv_rows,
)

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# Layout of raw_data_path (mirrors the Synapse project): Simulated Sequences/Seq_<N>/{rgb/
# image_<NNNN>.png, depth/aov_image_<NNNN>.exr, calibration.txt, info.txt, rgb.txt, depth.txt,
# trajectory.csv}. A marker records a completed Synapse folder download.
_SIMULATED_DIR: Final = "Simulated Sequences"
_RGB_DIR: Final = "rgb"
_DEPTH_DIR: Final = "depth"
_TEXT_FILES: Final = ("calibration.txt", "info.txt", "trajectory.csv")
_DOWNLOAD_MARKER: Final = ".download_complete"
_RAW_LINK_NAME: Final = "raw"  # symlink in the sequence folder onto the raw Seq_<N> folder

_RGB_FRAME_RE: Final = re.compile(r"^image_(?P<index>\d+)\.png$")
_DEPTH_FRAME_RE: Final = re.compile(r"^aov_image_(?P<index>\d+)\.exr$")
_DEPTH_CHANNEL: Final = 2  # the EXR's R channel (cv2 decodes BGRA); G/B are zero, A is one
_DM_TO_M: Final = 0.1  # info.txt: depth and trajectory are in decimeters

_GROUNDTRUTH_HEADER: Final = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def _frame_name(frame_index: int) -> str:
    return f"{frame_index:04d}.png"


def _indexed_files(folder: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    """frame index -> file, for the files in folder whose name matches pattern."""
    if not folder.is_dir():
        return {}
    files = {}
    for path in folder.iterdir():
        if match := pattern.match(path.name):
            files[int(match["index"])] = path
    return files


class EndomapperSimDataset(DatasetVSLAMLAB):
    """EndoMapper simulated colon dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "endomapper-sim") -> None:
        super().__init__(dataset_name)

        self.dataset_homepage: str = self.cfg["api_url"]
        self.synapse_project_id: str = self.cfg["synapse_project_id"]
        # Local mirror of the Synapse project (fetched on demand, reused as-is when present).
        self.raw_data_path = Path(self.cfg["raw_data_path"])
        # Depth in meters = depth_0 pixel value / depth_factor (see the yaml).
        self.depth_factor: float = float(self.cfg["depth_factor"])

    def download_sequence_data(self, sequence_name: str) -> None:
        raw_dir = self._raw_dir(sequence_name)
        if not self._raw_complete(raw_dir):
            syn = synapse_client()
            if syn is None:
                print_info(
                    f"Sequence '{sequence_name}' is not in {raw_dir.parent} and no Synapse credentials are configured "
                    f"(~/.synapseConfig or SYNAPSE_AUTH_TOKEN) - place its folder there yourself, or set up the "
                    f"credentials (see `pixi run get-resources`)."
                )
                return
            folder_id = synapse_resolve_path(syn, self.synapse_project_id, _SIMULATED_DIR, sequence_name)
            if folder_id is None:
                print_warning(f"{_SIMULATED_DIR}/{sequence_name} not found in Synapse project {self.synapse_project_id}")
                return
            print_info(f"Downloading {_SIMULATED_DIR}/{sequence_name} from Synapse ({folder_id}) -> {raw_dir}")
            synapse_download_folder(syn, folder_id, raw_dir)
            (raw_dir / _DOWNLOAD_MARKER).touch()

        raw_link = self.sequence_path(sequence_name) / _RAW_LINK_NAME
        if not (raw_link.is_symlink() or raw_link.exists()):
            self.sequence_path(sequence_name).mkdir(parents=True, exist_ok=True)
            # Absolute target on purpose: the raw folder lives outside the benchmark folder.
            os.symlink(raw_dir.resolve(), raw_link)

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

        rgb_files, depth_files = self._frame_files(sequence_name)
        target_size = None
        for index in tqdm(sorted(set(rgb_files) & set(depth_files)), desc=f"    resizing frames -> {rgb_path.name}/{depth_path.name}"):
            with Image.open(rgb_files[index]) as img:
                if target_size is None:
                    target_size = compute_scaled_size(img.size, self.target_resolution)
                rgb = img.convert("RGB")  # drop the constant alpha channel
                if self.target_resolution is not None:
                    rgb = rgb.resize(target_size, Image.Resampling.LANCZOS)
                rgb.save(tmp_rgb / _frame_name(index))

            depth_dm = self._read_depth(depth_files[index])
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
        # qw). Malformed rows (a truncated file) are skipped and reported, as are frames left
        # without a pose.
        poses: dict[int, list[float]] = {}
        skipped = 0
        with open(self._raw_link(sequence_name) / "trajectory.csv", encoding="utf-8") as f:
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
        # Deliberate no-op at every retention tier: raw/ is a symlink onto the raw folder (the
        # Synapse mirror, the only copy of the source frames), and nothing intermediate is written.
        return

    def get_download_issues(self, sequence_names: list[str]) -> list[dict]:
        # Only a problem when something must actually be fetched: a raw folder that already holds
        # the requested sequences needs no Synapse login at all.
        missing = [s for s in sequence_names if not self._raw_complete(self._raw_dir(s))]
        if not missing or synapse_client() is not None:
            return []
        return [_get_dataset_issue(issue_id="synapse_token", dataset_name=self.dataset_name, website=self.dataset_homepage)]

    # --- helpers, all recomputed from sequence_name (no per-sequence state on self) -------------
    def _raw_dir(self, sequence_name: str) -> Path:
        return self.raw_data_path / _SIMULATED_DIR / sequence_name

    def _raw_link(self, sequence_name: str) -> Path:
        raw_link = self.sequence_path(sequence_name) / _RAW_LINK_NAME
        if not raw_link.is_dir():
            raise FileNotFoundError(
                f"Raw folder for '{sequence_name}' not found at {raw_link}: run download_sequence_data first, and keep "
                f"raw_data_path in place while processing."
            )
        return raw_link

    @staticmethod
    def _raw_complete(raw_dir: Path) -> bool:
        """A finished Synapse download (marker), or a hand-placed copy with every piece present."""
        if (raw_dir / _DOWNLOAD_MARKER).is_file():
            return True
        return all((raw_dir / name).is_file() for name in _TEXT_FILES) and all(
            (raw_dir / folder).is_dir() and any((raw_dir / folder).iterdir()) for folder in (_RGB_DIR, _DEPTH_DIR)
        )

    def _frame_files(self, sequence_name: str) -> tuple[dict[int, Path], dict[int, Path]]:
        """(frame index -> rgb png, frame index -> depth exr) of the raw folder, reporting frames
        that have only one of the two (dropped: rgbd needs the pair)."""
        raw = self._raw_link(sequence_name)
        rgb = _indexed_files(raw / _RGB_DIR, _RGB_FRAME_RE)
        depth = _indexed_files(raw / _DEPTH_DIR, _DEPTH_FRAME_RE)
        for label, only in (("depth map", set(rgb) - set(depth)), ("RGB image", set(depth) - set(rgb))):
            if only:
                print_warning(f"{sequence_name}: dropping {len(only)} frames with no {label}")
        return rgb, depth

    @staticmethod
    def _read_depth(exr_path: Path) -> np.ndarray:
        """The EXR's depth channel as a float32 (H, W) array, in the source units (dm)."""
        image = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cv2 could not decode {exr_path} (is OPENCV_IO_ENABLE_OPENEXR set?)")
        return image[..., _DEPTH_CHANNEL] if image.ndim == 3 else image

    def _rgb_frames(self, sequence_name: str) -> list[Path]:
        rgb_path = self.rgb_path(sequence_name)
        return sorted(p for p in rgb_path.iterdir() if p.is_file() and p.suffix.lower() == ".png")

    def _ts_ns(self, frame_index: int) -> int:
        """Frame index -> timestamp in ns at the render rate (no timestamps ship with the frames)."""
        return int(round(frame_index * 1e9 / self.rgb_hz))

    def _read_calibration_txt(self, sequence_name: str) -> dict[str, float]:
        values: dict[str, float] = {}
        with open(self._raw_link(sequence_name) / "calibration.txt", encoding="utf-8") as f:
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
