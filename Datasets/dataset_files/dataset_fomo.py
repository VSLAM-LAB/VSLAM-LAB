"""
Module: VSLAM-LAB - Datasets - dataset_fomo.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5.1)
- Version: 1.0
- Created: 2026-10-02
- License: GPLv3 License

FoMo (Norlab, Université Laval): a Clearpath Warthog UGV repeating six fixed trajectories through
Forêt Montmorency over 12 deployments spanning one year. The public release is a plain anonymous
S3 bucket (s3://fomo-dataset) laid out as data/<deployment>/<recording>/<sensor>/, one PNG per
frame named by its microsecond UNIX timestamp, plus per-recording calib/ files, IMU CSVs and a
TUM-format gt.txt. This class takes the front-facing Stereolabs ZED X pair (factory-rectified
1920x1200 at 10 Hz; rgb_0 = left lens, rgb_1 = right lens) and the VectorNav VN100 IMU (200 Hz,
the body frame). The rear Basler ace2 is left out: its published calib/basler.json is a
byte-for-byte copy of the ZED X file, so no trustworthy intrinsics exist for it. Groundtruth is
the PPK-GNSS fused position of the three Emlid M2 receivers (MTM-7 / EPSG:2949 metres, 5 Hz) with
identity orientation, exactly as the release ships it.

Caveat on the IMU stream: the paper measures a ~56 ms USB latency of the VectorNav relative to
the ZED X exposure stamps (Kalibr time offset); whether the published vectornav.csv stamps are
already corrected for it is not documented, so they are written to imu_0.csv as-is.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, downloadFile, scale_intrinsics, write_csv_rows

# Bucket sensor folders, in rgb_<i> order. Left/right frames carry identical stamps (hardware
# pair, verified 2537/2537 on red_2024-11-21-10-34), so pairing is by filename.
CAMERAS: Final = ("zedx_left", "zedx_right")
IMU_NAME: Final = "vectornav"
# Everything fetched per recording: the two camera folders and calib/ recursively, plus these
# recording-root files (xsens.csv is 7 MB and kept so the other IMU can be swapped in without a
# re-download; CHANGELOG.md records the release's own data revisions).
RAW_PREFIXES: Final = CAMERAS + ("calib",)
RAW_FILES: Final = ("gt.txt", "vectornav.csv", "xsens.csv", "CHANGELOG.md")
# Week-long deployments are filed under their first day (fomo_sdk/common/naming.py's
# construct_deployment): recordings dated 2025-01-30 live in data/2025-01-29/, 2025-03-14 ones
# in data/2025-03-10/.
DEPLOYMENT_OF_DAY: Final[dict[str, str]] = {"2025-01-30": "2025-01-29", "2025-03-14": "2025-03-10"}
DOWNLOAD_WORKERS: Final = 8
DOWNLOAD_COMPLETE_MARKER: Final = ".download_complete"
S3_NS: Final = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
# VectorNav VN-100 datasheet ranges: accelerometer +-16 g, gyroscope +-2000 deg/s.
VN100_A_MAX: Final = 16.0 * 9.81
VN100_G_MAX: Final = float(np.deg2rad(2000.0))
GT_HEADER: Final = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def _timestamp_ns(stamp: str) -> int:
    """'1732203252.400000000' (seconds) -> integer ns without float round-off."""
    seconds, _, fraction = stamp.partition(".")
    return int(seconds) * 1_000_000_000 + int(fraction.ljust(9, "0")[:9])


def _pose_to_T(position: dict[str, float], orientation: dict[str, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat([orientation[k] for k in "xyzw"]).as_matrix()
    T[:3, 3] = [position[k] for k in "xyz"]
    return T


class FomoDataset(DatasetVSLAMLAB):
    """FoMo (Forêt Montmorency) multi-season UGV dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "fomo") -> None:
        super().__init__(dataset_name)

        # "https://<bucket>.s3.amazonaws.com/data/" -> the bucket endpoint (for ListObjectsV2 and
        # object GETs) and the key prefix the recordings live under.
        self.url_download_root: str = self.cfg["url_download_root"]
        parts = urlsplit(self.url_download_root)
        self._bucket_url: str = f"{parts.scheme}://{parts.netloc}"
        self._data_prefix: str = parts.path.strip("/")

    # ---- helpers ----
    def _raw_path(self, sequence_name: str) -> Path:
        """Mirror of the bucket's recording folder, inside the sequence folder."""
        return self.sequence_path(sequence_name) / "raw"

    def _recording_prefix(self, sequence_name: str) -> str:
        day = sequence_name.split("_", 1)[1][:10]
        return f"{self._data_prefix}/{DEPLOYMENT_OF_DAY.get(day, day)}/{sequence_name}/"

    def _rgb_paths(self, sequence_name: str) -> list[tuple[str, Path]]:
        sequence_path = self.sequence_path(sequence_name)
        return [(cam, self.rgb_path(sequence_name) if i == 0 else sequence_path / f"rgb_{i}")
                for i, cam in enumerate(CAMERAS)]

    def _s3_list(self, prefix: str, recursive: bool = True) -> list[tuple[str, int]]:
        """(key, size) of every object under prefix via anonymous ListObjectsV2, following
        continuation tokens (1000 keys per page). Non-recursive lists only the prefix's own files."""
        objects: list[tuple[str, int]] = []
        token = None
        while True:
            url = f"{self._bucket_url}/?list-type=2&max-keys=1000&prefix={quote(prefix)}"
            if not recursive:
                url += "&delimiter=/"
            if token:
                url += f"&continuation-token={quote(token)}"
            with urlopen(url) as response:
                root = ET.fromstring(response.read())
            for item in root.findall("s3:Contents", S3_NS):
                objects.append((item.find("s3:Key", S3_NS).text, int(item.find("s3:Size", S3_NS).text)))
            token_node = root.find("s3:NextContinuationToken", S3_NS)
            if token_node is None:
                return objects
            token = token_node.text

    def _fetch_object(self, raw_path: Path, recording_prefix: str, key: str, size: int) -> None:
        dest = raw_path / key[len(recording_prefix):]
        if dest.exists() and dest.stat().st_size == size:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloadFile(f"{self._bucket_url}/{quote(key)}", str(dest.parent))
        if dest.stat().st_size != size:
            raise ConnectionError(f"{key}: got {dest.stat().st_size} bytes, bucket lists {size}")

    def _transforms(self, sequence_name: str) -> dict[tuple[str, str], np.ndarray]:
        """calib/transforms.json as {(from, to): T_from_to}: each entry is a ROS-style TF, the
        pose of the `to` frame expressed in the `from` frame (fomo_sdk/tf/utils.py inverts it
        before handing it to pytransform3d, which wants the opposite direction)."""
        with (self._raw_path(sequence_name) / "calib" / "transforms.json").open("r", encoding="utf-8") as f:
            entries = json.load(f)
        return {(e["from"], e["to"]): _pose_to_T(e["position"], e["orientation"]) for e in entries}

    # ---- hooks ----
    def download_sequence_data(self, sequence_name: str) -> None:
        raw_path = self._raw_path(sequence_name)
        marker = raw_path / DOWNLOAD_COMPLETE_MARKER
        if marker.exists():
            return
        raw_path.mkdir(parents=True, exist_ok=True)

        recording_prefix = self._recording_prefix(sequence_name)
        objects = [obj for obj in self._s3_list(recording_prefix, recursive=False)
                   if obj[0][len(recording_prefix):] in RAW_FILES]
        for sub in RAW_PREFIXES:
            objects += self._s3_list(f"{recording_prefix}{sub}/")
        if not any(key.startswith(f"{recording_prefix}{CAMERAS[0]}/") for key, _ in objects):
            raise FileNotFoundError(f"{sequence_name}: no {CAMERAS[0]}/ frames under s3 prefix {recording_prefix}")

        # Objects already on disk with the listed size are skipped, so an interrupted run resumes.
        # downloadFile prints one progress line per object; with ~5000 PNGs per recording that is
        # swallowed in favour of a single tqdm bar (on stderr).
        total_gb = sum(size for _, size in objects) / 1e9
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool, contextlib.redirect_stdout(io.StringIO()):
            jobs = [pool.submit(self._fetch_object, raw_path, recording_prefix, key, size) for key, size in objects]
            for job in tqdm(jobs, desc=f"    fetching {sequence_name} ({total_gb:.1f} GB)", unit="file"):
                job.result()
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        raw_path = self._raw_path(sequence_name)
        for cam, rgb_path in self._rgb_paths(sequence_name):
            if rgb_path.exists():
                continue
            rgb_path.mkdir(parents=True, exist_ok=True)
            target_size = None
            for raw_image in tqdm(sorted((raw_path / cam).glob("*.png")), desc=f"    resizing {cam}"):
                if self.target_resolution is None:
                    shutil.copy2(raw_image, rgb_path / raw_image.name)
                    continue
                with Image.open(raw_image) as img:
                    # ZED frames are stored RGBA with an all-opaque alpha (bar a few pixels on the
                    # last row) - drop it rather than resample a useless channel.
                    rgb = img.convert("RGB")
                    if target_size is None:
                        target_size = compute_scaled_size(rgb.size, self.target_resolution)
                    rgb.resize(target_size, Image.Resampling.LANCZOS).save(rgb_path / raw_image.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # Filenames are the exposure-start UNIX time in microseconds; the pair shares one stamp.
        (cam_0, rgb_path_0), (cam_1, rgb_path_1) = self._rgb_paths(sequence_name)
        stamps_0 = {p.stem for p in rgb_path_0.glob("*.png")}
        stamps_1 = {p.stem for p in rgb_path_1.glob("*.png")}
        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]
        rows = []
        for stamp in sorted(stamps_0 & stamps_1, key=int):
            ts_ns = int(stamp) * 1000
            rows.append([ts_ns, f"{rgb_path_0.name}/{stamp}.png", ts_ns, f"{rgb_path_1.name}/{stamp}.png"])
        write_csv_rows(self.rgb_csv_path(sequence_name), header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        raw_path = self._raw_path(sequence_name)
        calib_path = raw_path / "calib"
        tf = self._transforms(sequence_name)

        # Body frame B = the VectorNav IMU. transforms.json gives the IMU's pose in the left lens
        # (Kalibr camera-IMU calibration) and the right lens' pose in the left lens (factory
        # stereo calibration): T_B_left = inv(T_left_imu), T_B_right = T_B_left @ T_left_right.
        T_BS = [np.linalg.inv(tf[(CAMERAS[0], IMU_NAME)])]
        T_BS.append(T_BS[0] @ tf[(CAMERAS[0], CAMERAS[1])])

        # The camera_info JSONs carry no width/height - take the native size off a raw frame, then
        # rescale the intrinsics to whatever create_rgb_folder wrote into rgb_<i> (issue #99).
        with Image.open(next((raw_path / CAMERAS[0]).glob("*.png"))) as raw_img:
            native_size = raw_img.size

        cams: list[dict[str, Any]] = []
        for i, cam in enumerate(CAMERAS):
            with (calib_path / f"{cam}.json").open("r", encoding="utf-8") as f:
                info = json.load(f)
            if any(float(d) != 0.0 for d in info["d"]):
                raise ValueError(f"{sequence_name}/{cam}: non-zero distortion {info['d']} - rectified pinhole expected")
            K = [float(v) for v in info["k"]]
            focal_length, principal_point = scale_intrinsics((K[0], K[4]), (K[2], K[5]), native_size, self.target_resolution)
            cams.append({
                "cam_name": f"rgb_{i}",
                "cam_type": "rgb",
                "cam_model": "pinhole",
                "focal_length": focal_length,
                "principal_point": principal_point,
                "fps": self.rgb_hz,
                "T_BS": T_BS[i],
            })

        # Noise densities / random walks: the release's Allan-variance result for this unit
        # (calib/kalibr-vectornav.yaml). g0: the per-recording static gyro bias the release
        # measured before each drive (calib/imu.json). Saturation limits from the VN-100 datasheet.
        with (calib_path / f"kalibr-{IMU_NAME}.yaml").open("r", encoding="utf-8") as f:
            noise = yaml.safe_load(f)
        with (calib_path / "imu.json").open("r", encoding="utf-8") as f:
            bias = json.load(f)[IMU_NAME]["angular_velocity"]
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": VN100_A_MAX,
            "g_max": VN100_G_MAX,
            "sigma_g_c": float(noise["gyroscope_noise_density"]),
            "sigma_a_c": float(noise["accelerometer_noise_density"]),
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": float(noise["gyroscope_random_walk"]),
            "sigma_aw_c": float(noise["accelerometer_random_walk"]),
            "g": 9.81007,
            "g0": [float(bias[k]) for k in "xyz"],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(noise["update_rate"]),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=cams, imu=[imu])

    def create_imu_csv(self, sequence_name: str) -> None:
        # Release header "t,wx,wy,wz,ax,ay,az", t in integer microseconds.
        header = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        rows = []
        with (self._raw_path(sequence_name) / f"{IMU_NAME}.csv").open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    rows.append([int(row[0]) * 1000] + row[1:7])
        write_csv_rows(self.imu_csv_path(sequence_name), header, rows)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # gt.txt is TUM-format "t x y z qx qy qz qw", t in seconds, x/y/z the PPK-fused antenna
        # position in MTM zone 7 (EPSG:2949) metres, and the quaternion an identity placeholder
        # (the GNSS pipeline estimates no orientation) - kept verbatim, like dataset_malaga.py.
        rows = []
        with (self._raw_path(sequence_name) / "gt.txt").open("r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                if len(values) != 8:
                    continue
                rows.append([_timestamp_ns(values[0])] + [float(v) for v in values[1:]])
        write_csv_rows(self.groundtruth_csv_path(sequence_name), GT_HEADER, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        # Nothing to drop at STANDARD: raw/ holds only original downloads (full-resolution PNGs,
        # the release's CSV/JSON files), no intermediate reformat. rgb_0/rgb_1 are real resized
        # copies, never symlinks into raw/, so raw/ is disposable at MINIMAL.
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(self._raw_path(sequence_name), ignore_errors=True)
