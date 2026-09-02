"""
Module: VSLAM-LAB - Datasets - dataset_lizard_island.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5.1)
- Version: 1.1
- Created: 2026-09-02
- Updated: 2026-09-03
- License: GPLv3 License
"""

from __future__ import annotations

import calendar
import csv
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import compute_scaled_size, make_printers, scale_intrinsics, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# Name of the symlink download_sequence_data drops inside each sequence folder, pointing at that
# sequence's GoPro camera folder on the campaign drive (raw_data_path in the yaml).
_RAW_LINK_NAME: Final = "raw"
_IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg"})

# EXIF tags (Exif IFD) read for each frame's capture timestamp.
_EXIF_IFD: Final = 0x8769
_EXIF_DATETIME_ORIGINAL: Final = 0x9003  # "YYYY:MM:DD HH:MM:SS", camera local time
_EXIF_SUBSEC_TIME_ORIGINAL: Final = 0x9291  # fractional-second digits, e.g. "1370" or " 280"

# Camera calibration shared by all five captures (same GoPro HERO11 Black model and photo mode):
# pinhole + radtan4 [k1, k2, p1, p2], self-calibrated with COLMAP on feb24_gp1 (2026-09-03). The
# intrinsics are given at the *resized* frame size the calibration was run at (593x518 - what
# compute_scaled_size gives for the 5568x4872 stills at target_resolution [640, 480]) and are
# rescaled in create_calibration_yaml to whatever size the current target_resolution produces.
_CALIBRATION_RESOLUTION: Final[tuple[int, int]] = (593, 518)  # (width, height)
_CALIBRATION_FOCAL_LENGTH: Final[tuple[float, float]] = (373.55200001047257, 372.46950225699976)
_CALIBRATION_PRINCIPAL_POINT: Final[tuple[float, float]] = (296.5, 259.0)
_CALIBRATION_DISTORTION: Final[tuple[float, float, float, float]] = (
    0.29363414812198091, 0.41721504621138134, -0.00020843765252784958, 0.0018989445323345231
)

# WGS84 ellipsoid, for the GPS -> local ENU groundtruth conversion.
_WGS84_A: Final = 6378137.0
_WGS84_E2: Final = 6.69437999014e-3


@dataclass(frozen=True)
class _Capture:
    """One GoPro capture on the campaign drive."""

    camera_dir: str  # relative to raw_data_path: holds the 1xxGOPRO image folders + the GPS csv
    gps_csv: str  # relative to camera_dir: image_name, latitude, longitude, height per frame
    date: str  # EXIF capture date (YYYY:MM:DD) of the survey - frames from any other day are
    #           test shots left on the card and are dropped


# Sequence -> capture. The Sep campaign folder is named LIRS_Sep_25 on the drive, but the frames'
# EXIF dates and the ASV telemetry both say September 2024 (see the yaml). Sep GP1's first image
# folder also holds a 918-frame test session from 2024-09-25 and GP2 a stray GOPR0043.JPG from
# that day - the date filter drops both. Sep ships two GPS csvs per GoPro: output.csv (with a
# NAME,LAT,LON,HEIGHT header) and sept_south_palf_1_v3_<N>.csv; output.csv is the trusted one -
# GP1's output.csv matches GP2's simultaneous track to 0.5 m while its v3 file is ~29 m off
# (mis-synced), and for GP2 both files agree.
_CAPTURES: Final[dict[str, _Capture]] = {
    "feb24_gp1": _Capture("LIRS_Feb_24/South_Palfrey_1/GoPro1", "south_palf_1.csv", "2024:02:15"),
    "mar24_gp1": _Capture("LIRS_Mar_24/South_Palfrey_1/GoPro1", "march_south_palf_1-GP1.csv", "2024:03:15"),
    "mar24_gp2": _Capture("LIRS_Mar_24/South_Palfrey_1/GoPro2", "march_south_palf_1-GP2.csv", "2024:03:15"),
    "sep24_gp1": _Capture("LIRS_Sep_25/GP1", "output.csv", "2024:09:26"),
    "sep24_gp2": _Capture("LIRS_Sep_25/GP2", "output.csv", "2024:09:26"),
}


def _exif_capture_time(image_path: Path) -> tuple[str, str] | None:
    """The frame's raw EXIF (DateTimeOriginal, SubsecTimeOriginal) strings, or None if the frame
    carries no DateTimeOriginal or isn't a readable image at all (the drive holds e.g. a zero-byte
    LIRS_Mar_24/.../103GOPRO/G0013195.JPG)."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif().get_ifd(_EXIF_IFD)
    except (UnidentifiedImageError, OSError):
        return None
    date_time = exif.get(_EXIF_DATETIME_ORIGINAL)
    if not date_time:
        return None
    return str(date_time), str(exif.get(_EXIF_SUBSEC_TIME_ORIGINAL) or "0")


def _capture_time_exif_bytes(capture_time: tuple[str, str]) -> bytes:
    """A minimal EXIF block holding just (DateTimeOriginal, SubsecTimeOriginal), stamped onto the
    resized rgb_0 frames so they stay self-describing (~90 bytes, vs. the ~60 KB GoPro EXIF blob
    with maker notes and thumbnail that a verbatim copy would carry)."""
    exif = Image.Exif()
    ifd = exif.get_ifd(_EXIF_IFD)
    ifd[_EXIF_DATETIME_ORIGINAL], ifd[_EXIF_SUBSEC_TIME_ORIGINAL] = capture_time
    return exif.tobytes()


def _timestamp_ns(capture_time: tuple[str, str]) -> tuple[str, int]:
    """(capture date 'YYYY:MM:DD', capture time in ns) from the EXIF strings. The camera's local
    wall-clock time is taken as UTC (no timezone is recorded) - only differences between frames
    matter downstream."""
    date_time, subsec = capture_time
    seconds = calendar.timegm(datetime.strptime(date_time, "%Y:%m:%d %H:%M:%S").timetuple())
    # SubsecTimeOriginal is a fixed-width digit string, space-padded on the left (" 280" is
    # 0.0280 s, "5280" is 0.5280 s) - the spaces are zeros, not something to strip.
    subsec = subsec.replace(" ", "0") or "0"
    frac_ns = int(subsec) * 10 ** (9 - len(subsec)) if len(subsec) <= 9 else int(subsec[:9])
    return date_time[:10], seconds * 1_000_000_000 + frac_ns


def _frame_stem(name: str) -> str:
    """'g0018313.jpg' / 'G0018313' -> 'G0018313' - the comparable GoPro frame id."""
    return Path(name.strip()).stem.upper()


def _parse_frame_range(entry: str) -> tuple[str, str]:
    """One exclude_frames entry ("G0018313.JPG" or "G0018313.JPG-G0018376.JPG") -> inclusive
    (first, last) stems. GoPro stems are fixed-width and chronological, so plain string comparison
    orders them (G0019999 < G0020001)."""
    parts = [part for part in str(entry).split("-") if part.strip()]
    if len(parts) == 1:
        return _frame_stem(parts[0]), _frame_stem(parts[0])
    if len(parts) == 2:
        first, last = _frame_stem(parts[0]), _frame_stem(parts[1])
        if first > last:
            raise ValueError(f"exclude_frames range '{entry}' runs backwards")
        return first, last
    raise ValueError(f"exclude_frames entry '{entry}' must be '<frame>' or '<first>-<last>'")


def _geodetic_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, h: np.ndarray,
                     lat0_deg: float, lon0_deg: float, h0: float) -> np.ndarray:
    """WGS84 geodetic (deg, deg, m) -> local East/North/Up (m) about the (lat0, lon0, h0) origin."""

    def ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
        x = (n + alt) * np.cos(lat) * np.cos(lon)
        y = (n + alt) * np.cos(lat) * np.sin(lon)
        z = (n * (1.0 - _WGS84_E2) + alt) * np.sin(lat)
        return np.stack([x, y, z], axis=-1)

    lat, lon = np.deg2rad(np.asarray(lat_deg, float)), np.deg2rad(np.asarray(lon_deg, float))
    lat0, lon0 = np.deg2rad(lat0_deg), np.deg2rad(lon0_deg)
    delta = ecef(lat, lon, np.asarray(h, float)) - ecef(np.array(lat0), np.array(lon0), np.array(h0, float))
    R = np.array([
        [-np.sin(lon0), np.cos(lon0), 0.0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
    ])
    return delta @ R.T


class LizardIslandDataset(DatasetVSLAMLAB):
    """Lizard Island coral-reef survey dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "lizard-island") -> None:
        super().__init__(dataset_name)

        # All sequences are local (scalar in the yaml): the campaign drive is the only source,
        # entered through raw_data_path.
        self.sequence_location = self.cfg["sequence_location"]
        self.raw_data_path = Path(self.cfg["raw_data_path"])

        # Per-sequence frames to leave out (see the yaml's exclude_frames): parsed once into
        # (first, last) inclusive ranges of upper-cased file stems; a single name is a one-frame range.
        self.exclude_frames: dict[str, list[tuple[str, str]]] = {
            sequence_name: [_parse_frame_range(entry) for entry in (entries or [])]
            for sequence_name, entries in (self.cfg.get("exclude_frames") or {}).items()
        }

    def download_sequence_data(self, sequence_name: str) -> None:
        raw_link = self._raw_link(sequence_name)
        if raw_link.is_symlink() or raw_link.exists():
            return

        camera_dir = self.raw_data_path / self._capture(sequence_name).camera_dir
        if not camera_dir.is_dir():
            print_info(
                f"Sequence '{sequence_name}' is marked as 'local'. Its raw GoPro camera folder was not found at "
                f"{camera_dir} - mount the campaign drive, or point raw_data_path in dataset_{self.dataset_name}.yaml "
                f"at your copy of Serena_Mou_Data."
            )
            return

        self.sequence_path(sequence_name).mkdir(parents=True, exist_ok=True)
        # Absolute target on purpose: the raw data lives on an external drive, outside the
        # benchmark folder, so a relative link would break if either were moved.
        os.symlink(camera_dir.resolve(), raw_link)

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            # Already built: only apply an exclude_frames list that grew since (cheap, idempotent).
            self._prune_excluded(sequence_name)
            return

        frames = self._raw_frames(sequence_name)
        # Build in a sibling temp folder and rename once complete, so a crash midway can't leave a
        # partial rgb_0/ that later looks finished.
        tmp_path = rgb_path.with_name(rgb_path.name + ".tmp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True)

        target_size = None
        init_size = None
        for _, frame, capture_time in tqdm(frames, desc=f"    resizing frames -> {rgb_path.name}"):
            if self.target_resolution is None:
                shutil.copy2(frame, tmp_path / frame.name)  # keeps the full original EXIF
                continue

            try:
                with Image.open(frame) as img:
                    if target_size is None:
                        init_size = img.size
                        target_size = compute_scaled_size(img.size, self.target_resolution)
                    if img.size != init_size:
                        print_warning(f"{frame.name} {img.size} != {init_size}")
                    # 27 MP stills: let libjpeg decode at the largest DCT-scaled size that is still
                    # >= target_size (~8x cheaper than a full decode), then LANCZOS the rest of the way.
                    img.draft(img.mode, target_size)
                    img.load()
                    resized = img.resize(target_size, Image.Resampling.LANCZOS)
            except (UnidentifiedImageError, OSError) as err:
                # A truncated/corrupt raw frame: drop it (rgb.csv is derived from what lands in
                # rgb_0, so it stays consistent) rather than abort the whole sequence.
                print_warning(f"{sequence_name}: skipping unreadable frame {frame.name} ({err})")
                continue
            resized.save(tmp_path / frame.name, exif=_capture_time_exif_bytes(capture_time))

        tmp_path.rename(rgb_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists() and not self._csv_lists_excluded(sequence_name, rgb_csv):
            return

        rgb_path = self.rgb_path(sequence_name)
        rows = [[ts_ns, f"{rgb_path.name}/{name}"] for ts_ns, name in self._frame_timestamps(sequence_name)]
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # One shared pinhole + radtan4 calibration (see _CALIBRATION_* above). It describes the
        # 593x518 resized frames; rescale it to the size the current target_resolution actually
        # yields (a no-op at the default [640, 480]), and warn if rgb_0 disagrees with that (#99).
        self._check_calibration_resolution(sequence_name)
        focal_length, principal_point = scale_intrinsics(
            _CALIBRATION_FOCAL_LENGTH, _CALIBRATION_PRINCIPAL_POINT, _CALIBRATION_RESOLUTION, self.target_resolution
        )
        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "distortion_type": "radtan4",
            "distortion_coefficients": [float(v) for v in _CALIBRATION_DISTORTION],
            "focal_length": focal_length,
            "principal_point": principal_point,
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # Groundtruth is the Floatyboat's GPS, given per image (image_name, latitude, longitude,
        # height): local ENU positions (m, +Z up) about the first fix of the sequence. Position
        # only - orientation is written as identity. Height is 0 for every fix in every capture
        # (surface vessel), so tz is (up to Earth curvature) 0. The GPS updates slower than the
        # 2 Hz camera, so consecutive frames often repeat one fix - kept as-is, one row per frame
        # that has a fix; frames without a fix get no row.
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        fixes = self._gps_fixes(sequence_name)
        frames = self._frame_timestamps(sequence_name)
        matched = [(ts_ns, fixes[name.upper()]) for ts_ns, name in frames if name.upper() in fixes]
        if not matched:
            print_warning(f"{sequence_name}: no frame has a GPS fix - writing an empty groundtruth.csv")
            write_csv_rows(groundtruth_csv, header, [])
            return
        if len(matched) < len(frames):
            print_warning(f"{sequence_name}: {len(frames) - len(matched)} of {len(frames)} frames have no GPS fix")

        lat, lon, h = (np.array([fix[i] for _, fix in matched]) for i in range(3))
        enu = _geodetic_to_enu(lat, lon, h, float(lat[0]), float(lon[0]), float(h[0]))
        rows = [
            [ts_ns, float(e), float(n), float(u), 0.0, 0.0, 0.0, 1.0]
            for (ts_ns, _), (e, n, u) in zip(matched, enu)
        ]
        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberate no-op at every retention tier, including MINIMAL: raw/ is a symlink onto the
        # campaign drive (the only full-resolution copy, with no remote source to re-download
        # from), and nothing else intermediate is written.
        return

    def _check_calibration_resolution(self, sequence_name: str) -> None:
        """Warn if the first rgb_0 frame's size differs from what the calibration is being scaled
        to - the written intrinsics would then describe the wrong image size."""
        rgb_path = self.rgb_path(sequence_name)
        frames = sorted(p for p in rgb_path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES) if rgb_path.is_dir() else []
        if not frames:
            return
        expected_size = compute_scaled_size(_CALIBRATION_RESOLUTION, self.target_resolution)
        with Image.open(frames[0]) as img:
            if img.size != expected_size:
                print_warning(
                    f"{sequence_name}: {rgb_path.name}/{frames[0].name} is {img.size}, but the calibration is scaled "
                    f"for {expected_size} - intrinsics may describe the wrong image size."
                )

    def _is_excluded(self, sequence_name: str, name: str) -> bool:
        stem = _frame_stem(name)
        return any(first <= stem <= last for first, last in self.exclude_frames.get(sequence_name, []))

    def _prune_excluded(self, sequence_name: str) -> None:
        """Delete frames of an already-built rgb_0/ that exclude_frames now lists."""
        rgb_path = self.rgb_path(sequence_name)
        pruned = [p for p in rgb_path.iterdir() if p.is_file() and self._is_excluded(sequence_name, p.name)]
        for p in pruned:
            p.unlink()
        if pruned:
            print_info(f"{sequence_name}: removed {len(pruned)} excluded frames from {rgb_path.name}/")

    def _csv_lists_excluded(self, sequence_name: str, rgb_csv: Path) -> bool:
        with open(rgb_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            return any(row and self._is_excluded(sequence_name, Path(row[1]).name) for row in reader)

    def _capture(self, sequence_name: str) -> _Capture:
        try:
            return _CAPTURES[sequence_name]
        except KeyError:
            raise ValueError(f"Unknown {self.dataset_name} sequence '{sequence_name}' - expected one of {sorted(_CAPTURES)}") from None

    def _raw_link(self, sequence_name: str) -> Path:
        return self.sequence_path(sequence_name) / _RAW_LINK_NAME

    def _frame_timestamps(self, sequence_name: str) -> list[tuple[int, str]]:
        """The sequence's frames as (capture time ns, file name), in capture order, from the
        cheapest source that already has them - rgb.csv if written, else the EXIF stamped on the
        resized rgb_0 frames (local disk), else the raw frames on the campaign drive (the slow,
        authoritative path, ~minutes of EXIF reads over USB). Each hook stays independently
        callable; they just get faster once their predecessors have run."""
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.is_file():
            with open(rgb_csv, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                rows = [
                    (int(row[0]), Path(row[1]).name)
                    for row in reader
                    if row and not self._is_excluded(sequence_name, Path(row[1]).name)
                ]
            if rows:
                return rows

        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.is_dir():
            stamped = []
            for path in sorted(p for p in rgb_path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES):
                if self._is_excluded(sequence_name, path.name):
                    continue
                capture_time = _exif_capture_time(path)
                if capture_time is None:
                    break  # not stamped by this class - fall through to the raw frames
                stamped.append((_timestamp_ns(capture_time)[1], path.name))
            else:
                if stamped:
                    stamped.sort()
                    return stamped

        return [(ts_ns, frame.name) for ts_ns, frame, _ in self._raw_frames(sequence_name)]

    def _raw_frames(self, sequence_name: str) -> list[tuple[int, Path, tuple[str, str]]]:
        """The sequence's frames as (capture time ns, raw path, raw EXIF capture-time strings), in
        capture order - every JPEG under the linked camera folder (the 1xxGOPRO subfolders)
        captured on the survey date. Recomputed from sequence_name, never cached on self."""
        capture = self._capture(sequence_name)
        raw_link = self._raw_link(sequence_name)
        if not raw_link.is_dir():
            raise FileNotFoundError(
                f"Raw frames for '{sequence_name}' not found at {raw_link} (sequence marked as 'local'): run "
                f"download_sequence_data with the campaign drive mounted, and keep it mounted while processing."
            )

        frames: list[tuple[int, Path, tuple[str, str]]] = []
        skipped_days: dict[str, int] = {}
        no_exif: list[str] = []
        candidates = sorted(p for p in raw_link.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        excluded = [p for p in candidates if self._is_excluded(sequence_name, p.name)]
        if excluded:
            print_info(f"{sequence_name}: leaving out {len(excluded)} frames listed in exclude_frames")
            candidates = [p for p in candidates if not self._is_excluded(sequence_name, p.name)]
        for path in tqdm(candidates, desc=f"    reading EXIF capture times ({raw_link.name}/)"):
            capture_time = _exif_capture_time(path)
            if capture_time is None:
                no_exif.append(path.name)
                continue
            date, ts_ns = _timestamp_ns(capture_time)
            if date != capture.date:
                skipped_days[date] = skipped_days.get(date, 0) + 1
                continue
            frames.append((ts_ns, path, capture_time))

        if no_exif:
            print_warning(
                f"{sequence_name}: skipped {len(no_exif)} unreadable JPEGs / JPEGs without an EXIF capture time: "
                + ", ".join(no_exif[:5]) + (" ..." if len(no_exif) > 5 else "")
            )
        for date, count in sorted(skipped_days.items()):
            print_info(f"{sequence_name}: skipping {count} frames captured on {date} (survey day is {capture.date})")
        frames.sort(key=lambda item: (item[0], item[1].name))
        duplicates = sum(1 for i in range(1, len(frames)) if frames[i][0] == frames[i - 1][0])
        if duplicates:
            print_warning(f"{sequence_name}: {duplicates} frames share a capture timestamp with their predecessor")
        return frames

    def _gps_fixes(self, sequence_name: str) -> dict[str, tuple[float, float, float]]:
        """image name (upper-cased) -> (latitude deg, longitude deg, height m) from the
        capture's GPS csv. Tolerates an optional NAME,LAT,LON,HEIGHT header row."""
        capture = self._capture(sequence_name)
        gps_csv = self._raw_link(sequence_name) / capture.gps_csv
        fixes: dict[str, tuple[float, float, float]] = {}
        with open(gps_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 4:
                    continue
                try:
                    fixes[row[0].strip().upper()] = (float(row[1]), float(row[2]), float(row[3]))
                except ValueError:
                    continue  # header row
        return fixes
