"""
Module: VSLAM-LAB - Datasets - dataset_grandtour.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5.1)
- Version: 1.0
- Created: 2026-10-01
- License: GPLv3 License

GrandTour (ETH Zurich RSL, ANYmal-D + Boxi payload), from the Hugging Face release
(leggedrobotics/grand_tour_dataset): per mission, one tarball per sensor stream holding a zarr-v2
group of timestamps/poses/IMU samples (data/<topic>.tar), one tarball of decoded frames per camera
(images/<camera>.tar) and per-sensor calibration yamls (metadata/*.yaml). Two sensor-rig splits of
the same missions share this module, like dataset_rover.py: grandtour-alphasense (monochrome
global-shutter stereo pair + BMI085 IMU) and grandtour-zed2i (color stereo pair, no IMU).
"""

from __future__ import annotations

import json
import shutil
import struct
import tarfile
from pathlib import Path
from typing import Any, Final

import lz4.block
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, hf_token, scale_intrinsics, write_csv_rows

# Paper mission tag (lower-cased, ASCII) -> Hugging Face mission folder. RIV-1 (2024-11-04-16-52-38)
# ships only a LiDAR map on Hugging Face and is left out; the two unlisted HF folders
# (2024-10-29-09-53-44, 2024-11-15-15-07-36) are empty.
MISSIONS: Final[dict[str, str]] = {
    "eth_1": "2024-10-01-11-29-55", "eth_2": "2024-10-01-11-47-44", "eth_3": "2024-10-01-12-00-49",
    "spx_1": "2024-11-02-17-10-25", "spx_2": "2024-11-02-17-18-32", "spx_3": "2024-11-02-17-43-10",
    "ice_1": "2024-11-02-21-12-51",
    "snow_1": "2024-11-03-07-52-45", "snow_2": "2024-11-03-07-57-34", "snow_3": "2024-11-03-08-17-23",
    "eig_1": "2024-11-03-13-51-43", "eig_2": "2024-11-03-13-59-54",
    "gri_1": "2024-11-04-10-57-34",
    "cyn_1": "2024-11-04-12-55-59", "cyn_2": "2024-11-04-13-07-13",
    "hil_1": "2024-11-04-16-05-00",
    "pil_1": "2024-11-11-12-07-40", "pil_2": "2024-11-11-12-42-47",
    "root_1": "2024-11-11-14-29-44",
    "haus_1": "2024-11-11-16-14-23",
    "hoeb_1": "2024-11-14-11-17-02", "hoeb_2": "2024-11-14-12-01-26",
    "heap_1": "2024-11-14-13-45-37",
    "kaeb_1": "2024-11-14-14-36-02", "kaeb_2": "2024-11-14-15-22-43", "kaeb_3": "2024-11-14-16-04-09",
    "trim_1": "2024-11-15-10-16-35",
    "alb_1": "2024-11-15-11-18-14", "alb_2": "2024-11-15-11-37-15", "alb_3": "2024-11-15-12-06-03",
    "lmb_1": "2024-11-15-14-14-12", "lmb_2": "2024-11-15-14-43-52",
    "lee_1": "2024-11-15-16-41-14",
    "arc_1": "2024-11-18-12-05-01", "arc_2": "2024-11-18-13-22-14", "arc_3": "2024-11-18-13-48-19",
    "arc_4": "2024-11-18-15-46-05", "arc_5": "2024-11-18-16-59-23", "arc_6": "2024-11-18-17-13-09",
    "arc_7": "2024-11-18-17-31-36",
    "leica_1": "2024-11-25-14-57-08", "leica_2": "2024-11-25-16-36-19",
    "sbb_1": "2024-12-03-13-15-38", "sbb_2": "2024-12-03-13-26-40",
    "con_1": "2024-12-09-09-34-43", "con_2": "2024-12-09-09-41-46", "con_3": "2024-12-09-11-28-28",
    "con_4": "2024-12-09-11-53-11",
}

# Groundtruth topics, in preference order. cpt7_ie_tc_odometry is the NovAtel Inertial Explorer
# tightly-coupled post-processed RTK-GNSS/INS pose of cpt7_imu (= box_base) in an ENU frame, 200 Hz,
# 6-DoF - missing on the indoor/no-GNSS missions. prism_position is the Leica MS60 total station's
# sub-cm position of the Boxi prism (position only, ~23 Hz, gaps whenever line of sight is lost) -
# withheld on the six COMFORT-benchmark test missions. ARC-7 and CON-4 have neither.
GT_TOPIC_IE: Final = "cpt7_ie_tc_odometry"
GT_TOPIC_PRISM: Final = "prism_position"
OPTIONAL_TOPICS: Final = frozenset({GT_TOPIC_IE, GT_TOPIC_PRISM})
TF_METADATA: Final = "tf"

# Both rigs' left/right frames carry identical hardware-trigger stamps; this only absorbs float
# round-off, not a real offset (one Alphasense stream drops a frame mid-mission, so pairing is by
# timestamp rather than by index).
STEREO_PAIR_TOLERANCE_S: Final = 0.002
IMAGE_SUFFIXES: Final = (".png", ".jpg", ".jpeg")
GT_HEADER: Final = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


##################################################################################################################################################
# Minimal zarr-v2 reader
#
# The data/<topic>.tar tarballs hold zarr-v2 groups whose chunks are blosc-compressed (lz4 codec,
# byte shuffle). The vslamlab pixi environment ships no zarr/numcodecs, but it does ship lz4 and
# numpy, which is all the blosc container needs: a 16-byte header, per-block offsets, and per block
# `typesize` lz4 streams (blosc's "split" mode) that are byte-unshuffled after decoding. Validated
# against the published stream descriptions (rates, field shapes, unit quaternions).
##################################################################################################################################################
_BLOSC_BYTESHUFFLE: Final = 0x1
_BLOSC_MEMCPYED: Final = 0x2
_BLOSC_BITSHUFFLE: Final = 0x4
_BLOSC_LZ4: Final = 1
_BLOSC_MAX_SPLITS: Final = 16
_BLOSC_MIN_BUFFERSIZE: Final = 128


def _blosc_decompress(buf: bytes) -> bytes:
    """Decodes one blosc-1 buffer (lz4 codec, optional byte shuffle)."""
    _version, _versionlz, flags, typesize = struct.unpack_from("<BBBB", buf, 0)
    nbytes, blocksize, _cbytes = struct.unpack_from("<III", buf, 4)
    if flags & _BLOSC_MEMCPYED:
        return bytes(buf[16:16 + nbytes])
    if flags & _BLOSC_BITSHUFFLE or (flags >> 5) != _BLOSC_LZ4:
        raise NotImplementedError("only lz4 + byte-shuffle blosc buffers are supported")
    nblocks = (nbytes + blocksize - 1) // blocksize
    bstarts = struct.unpack_from(f"<{nblocks}i", buf, 16)
    nsplits = typesize if (typesize <= _BLOSC_MAX_SPLITS and blocksize // typesize >= _BLOSC_MIN_BUFFERSIZE) else 1
    out = bytearray(nbytes)
    for j in range(nblocks):
        bsize = min(blocksize, nbytes - j * blocksize)
        neblock = bsize // nsplits
        pos = bstarts[j]
        block = bytearray()
        for _ in range(nsplits):
            (csize,) = struct.unpack_from("<i", buf, pos)
            pos += 4
            chunk = bytes(buf[pos:pos + csize])
            pos += csize
            block += chunk if csize == neblock else lz4.block.decompress(chunk, uncompressed_size=neblock)
        if flags & _BLOSC_BYTESHUFFLE and typesize > 1:
            block = np.frombuffer(bytes(block), dtype=np.uint8).reshape(typesize, bsize // typesize).T.tobytes()
        out[j * blocksize:j * blocksize + bsize] = block
    return bytes(out)


def _read_zarr_array(array_dir: Path) -> np.ndarray:
    """Loads a zarr-v2 array directory (.zarray + chunk files) into memory."""
    meta = json.loads((array_dir / ".zarray").read_text(encoding="utf-8"))
    if meta.get("zarr_format") != 2 or meta.get("filters"):
        raise NotImplementedError(f"unsupported zarr array layout in {array_dir}")
    compressor = meta.get("compressor")
    dtype = np.dtype(meta["dtype"])
    shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
    separator = meta.get("dimension_separator", ".")
    out = np.full(shape, meta.get("fill_value") or 0, dtype=dtype)
    grid = [(s + c - 1) // c for s, c in zip(shape, chunks)]
    for idx in np.ndindex(*grid):
        chunk_file = array_dir / separator.join(str(i) for i in idx)
        if not chunk_file.exists():
            continue
        raw = chunk_file.read_bytes()
        if compressor is None:
            data = raw
        elif compressor.get("id") == "blosc":
            data = _blosc_decompress(raw)
        else:
            raise NotImplementedError(f"unsupported zarr compressor {compressor} in {array_dir}")
        block = np.frombuffer(data, dtype=dtype).reshape(chunks, order=meta.get("order", "C"))
        region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[region] = block[tuple(slice(0, r.stop - r.start) for r in region)]
    return out


def _read_zarr_attrs(group_dir: Path) -> dict[str, Any]:
    return json.loads((group_dir / ".zattrs").read_text(encoding="utf-8"))
##################################################################################################################################################


def _pose_to_T(position: Any, quat_xyzw: Any) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


def _seconds_to_ns(ts_s: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(ts_s, dtype=np.float64) * 1e9).astype(np.int64)


class GrandtourDataset(DatasetVSLAMLAB):
    """GrandTour legged-robotics dataset helper for VSLAM-LAB benchmark (shared by both sensor-rig splits)."""

    # (image tag, metadata caminfo stem) per rgb_<i>; the image tag is also the transform yaml stem.
    CAMERAS: tuple[tuple[str, str], ...] = ()
    CAM_TYPE: str = "gray"
    # VSLAM-LAB cam_model written by create_calibration_yaml, with the number of D coefficients kept.
    DISTORTION_TYPE: str = "equid4"
    # zarr IMU topic feeding imu_0.csv, or None for a rig without an IMU in the release.
    IMU_TOPIC: str | None = None
    # Metadata stem of the sensor whose frame is VSLAM-LAB's body frame (every T_BS and the
    # groundtruth are expressed in it): the IMU for a -vi rig, else rgb_0's camera.
    BODY_SENSOR: str = ""

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id: str = self.cfg["hf_repo_id"]

    # ---- per-sequence layout helpers (each recomputed from sequence_name, never cached on self) ----
    def _mission(self, sequence_name: str) -> str:
        return MISSIONS[sequence_name]

    def _mission_path(self, sequence_name: str) -> Path:
        return self.sequence_path(sequence_name) / self._mission(sequence_name)

    def _data_topics(self) -> list[str]:
        topics = [tag for tag, _ in self.CAMERAS]
        if self.IMU_TOPIC:
            topics.append(self.IMU_TOPIC)
        return topics

    def _metadata_files(self) -> list[str]:
        files = [TF_METADATA]
        for tag, caminfo in self.CAMERAS:
            files += [tag, caminfo]
        if self.IMU_TOPIC:
            files.append(self.IMU_TOPIC)
        return files

    @staticmethod
    def _ensure_extracted(tar_path: Path) -> Path | None:
        """Extracts <dir>/<name>.tar next to itself (members are <name>/...) unless <dir>/<name>/
        already exists; returns the member directory, or None when neither exists."""
        member_dir = tar_path.with_suffix("")
        if member_dir.is_dir():
            return member_dir
        if not tar_path.is_file():
            return None
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=tar_path.parent, filter="data")
        return member_dir

    def _topic_dir(self, sequence_name: str, topic: str) -> Path | None:
        return self._ensure_extracted(self._mission_path(sequence_name) / "data" / f"{topic}.tar")

    def _image_dir(self, sequence_name: str, image_tag: str) -> Path:
        image_dir = self._ensure_extracted(self._mission_path(sequence_name) / "images" / f"{image_tag}.tar")
        if image_dir is None:
            raise FileNotFoundError(f"{sequence_name}: images/{image_tag}.tar not downloaded")
        return image_dir

    def _timestamps_s(self, sequence_name: str, topic: str) -> np.ndarray:
        topic_dir = self._topic_dir(sequence_name, topic)
        if topic_dir is None:
            raise FileNotFoundError(f"{sequence_name}: data/{topic}.tar not downloaded")
        return _read_zarr_array(topic_dir / "timestamp")

    def _T_box_sensor(self, sequence_name: str, metadata_stem: str) -> np.ndarray:
        """Pose of a sensor frame in box_base. The release's per-sensor `transform:` is the inverse
        of a conventional parent->child TF - it maps box_base points INTO the sensor frame
        (verified numerically: inverting it puts the ZED2i right camera 0.120 m along the left
        camera's +x, the spec'd 12 cm baseline, and points the front cameras' optical axis along
        box_base +x). A few sensors are published w.r.t. ANYmal's `base` instead; those get chained
        through tf.yaml's box_base entry."""
        metadata_path = self._mission_path(sequence_name) / "metadata"
        with (metadata_path / f"{metadata_stem}.yaml").open("r", encoding="utf-8") as f:
            transform = yaml.safe_load(f)["transform"]
        T_sensor_parent = _pose_to_T(
            [transform["translation"][k] for k in "xyz"], [transform["rotation"][k] for k in "xyzw"])
        T_parent_sensor = np.linalg.inv(T_sensor_parent)
        if transform["base_frame_id"] == "box_base":
            return T_parent_sensor
        if transform["base_frame_id"] != "base":
            raise ValueError(f"{sequence_name}/{metadata_stem}: unexpected base_frame_id {transform['base_frame_id']}")
        with (metadata_path / f"{TF_METADATA}.yaml").open("r", encoding="utf-8") as f:
            box = yaml.safe_load(f)["box_base"]
        T_box_base = _pose_to_T([box["translation"][k] for k in "xyz"], [box["rotation"][k] for k in "xyzw"])
        return T_box_base @ T_parent_sensor

    def _T_body_sensor(self, sequence_name: str, metadata_stem: str) -> np.ndarray:
        T_box_body = self._T_box_sensor(sequence_name, self.BODY_SENSOR)
        return np.linalg.inv(T_box_body) @ self._T_box_sensor(sequence_name, metadata_stem)

    # ---- hooks ----
    def download_sequence_data(self, sequence_name: str) -> None:
        # Only the streams this rig needs are fetched (~2 GB for the Alphasense pair, ~4 GB for the
        # ZED2i pair, per mission); a completion marker, touched once every file is on disk and
        # every tarball is extracted, is what makes a retry after a dropped connection resume
        # rather than be skipped. hf_hub_download itself resumes partial files.
        sequence_path = self.sequence_path(sequence_name)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return
        sequence_path.mkdir(parents=True, exist_ok=True)
        mission = self._mission(sequence_name)

        remote_files = [f"{mission}/data/.zgroup"]
        remote_files += [f"{mission}/metadata/{stem}.yaml" for stem in self._metadata_files()]
        remote_files += [f"{mission}/images/{tag}.tar" for tag, _ in self.CAMERAS]
        remote_files += [f"{mission}/data/{topic}.tar" for topic in self._data_topics() + sorted(OPTIONAL_TOPICS)]
        for remote_file in remote_files:
            try:
                hf_hub_download(repo_id=self.hf_repo_id, filename=remote_file, repo_type="dataset",
                                token=hf_token(), local_dir=sequence_path)
            except EntryNotFoundError:
                if Path(remote_file).stem not in OPTIONAL_TOPICS:
                    raise

        for tar_path in sorted((sequence_path / mission).glob("*/*.tar")):
            self._ensure_extracted(tar_path)
        shutil.rmtree(sequence_path / ".cache", ignore_errors=True)
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for i, (image_tag, _) in enumerate(self.CAMERAS):
            rgb_path = self.rgb_path(sequence_name) if i == 0 else sequence_path / f"rgb_{i}"
            if rgb_path.exists():
                continue
            source_dir = self._image_dir(sequence_name, image_tag)
            rgb_path.mkdir(parents=True, exist_ok=True)
            for raw_image in sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                if self.target_resolution is None:
                    shutil.copy2(raw_image, rgb_path / raw_image.name)
                    continue
                with Image.open(raw_image) as img:
                    target_size = compute_scaled_size(img.size, self.target_resolution)
                    resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    save_kwargs = {"quality": 95} if raw_image.suffix.lower() in (".jpg", ".jpeg") else {}
                    resized.save(rgb_path / raw_image.name, **save_kwargs)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # Each camera's zarr timestamp array is index-aligned with its decoded frames (frame k is
        # <k:06d>.<ext>, in the release's own convention). Left/right are then paired by nearest
        # timestamp, which drops the unpaired frame around a single dropped trigger instead of
        # shifting the rest of the stream by one frame as an index zip would.
        sequence_path = self.sequence_path(sequence_name)
        streams = []
        for i, (image_tag, _) in enumerate(self.CAMERAS):
            rgb_dir = self.rgb_path(sequence_name) if i == 0 else sequence_path / f"rgb_{i}"
            names = sorted(p.name for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            ts_ns = _seconds_to_ns(self._timestamps_s(sequence_name, image_tag))
            if len(names) != len(ts_ns):
                raise ValueError(f"{sequence_name}/{image_tag}: {len(names)} frames but {len(ts_ns)} timestamps")
            streams.append(pd.DataFrame({f"ts_rgb_{i} (ns)": ts_ns, f"path_rgb_{i}": [f"{rgb_dir.name}/{n}" for n in names]}))

        merged = streams[0]
        for i, stream in enumerate(streams[1:], start=1):
            merged = pd.merge_asof(merged, stream, left_on="ts_rgb_0 (ns)", right_on=f"ts_rgb_{i} (ns)",
                                   direction="nearest", tolerance=int(STEREO_PAIR_TOLERANCE_S * 1e9)).dropna()
            merged[f"ts_rgb_{i} (ns)"] = merged[f"ts_rgb_{i} (ns)"].astype(np.int64)
        header = list(merged.columns)
        write_csv_rows(self.rgb_csv_path(sequence_name), header, merged[header].astype(object).values.tolist())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        metadata_path = self._mission_path(sequence_name) / "metadata"
        cams: list[dict[str, Any]] = []
        for i, (image_tag, caminfo_stem) in enumerate(self.CAMERAS):
            with (metadata_path / f"{caminfo_stem}.yaml").open("r", encoding="utf-8") as f:
                info = yaml.safe_load(f)["camera_info"]
            K = [float(v) for v in info["K"]]
            native_size = (int(info["width"]), int(info["height"]))
            # Calibration is at the camera's native resolution - rescale to what create_rgb_folder
            # wrote into rgb_<i> (VSLAM-LAB issue #99).
            focal_length, principal_point = scale_intrinsics(
                (K[0], K[4]), (K[2], K[5]), native_size, self.target_resolution)
            n_coeffs = int(self.DISTORTION_TYPE[-1])
            cams.append({
                "cam_name": f"rgb_{i}",
                "cam_type": self.CAM_TYPE,
                "cam_model": "pinhole",
                "distortion_type": self.DISTORTION_TYPE,
                "focal_length": focal_length,
                "principal_point": principal_point,
                "distortion_coefficients": [float(v) for v in info["D"][:n_coeffs]],
                "fps": float(self.rgb_hz),
                "T_BS": self._T_body_sensor(sequence_name, image_tag),
            })
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=cams, imu=self._imu_calibration(sequence_name))

    def _imu_calibration(self, sequence_name: str) -> list[dict[str, Any]] | None:
        return None

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        rows: list[list[Any]] = []
        ie_dir = self._topic_dir(sequence_name, GT_TOPIC_IE)
        prism_dir = self._topic_dir(sequence_name, GT_TOPIC_PRISM)
        if ie_dir is not None:
            # T_enu_box (box_base == cpt7_imu, identity in the TF tree) re-expressed for the body
            # sensor: T_enu_body = T_enu_box @ T_box_body.
            T_box_body = self._T_box_sensor(sequence_name, self.BODY_SENSOR)
            ts_ns = _seconds_to_ns(_read_zarr_array(ie_dir / "timestamp"))
            R_enu_box = R.from_quat(_read_zarr_array(ie_dir / "pose_orien"))
            p_enu_body = _read_zarr_array(ie_dir / "pose_pos") + R_enu_box.apply(T_box_body[:3, 3])
            q_enu_body = (R_enu_box * R.from_matrix(T_box_body[:3, :3])).as_quat()
            for t, p, q in zip(ts_ns, p_enu_body, q_enu_body):
                rows.append([int(t), *map(float, p), *map(float, q)])
        elif prism_dir is not None:
            # Position-only: the prism's offset from the body frame isn't part of the Hugging Face
            # metadata, so orientation is written as identity (same convention as dataset_malaga.py).
            ts_ns = _seconds_to_ns(_read_zarr_array(prism_dir / "timestamp"))
            for t, p in zip(ts_ns, _read_zarr_array(prism_dir / "point")):
                rows.append([int(t), float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0, 1.0])
        write_csv_rows(self.groundtruth_csv_path(sequence_name), GT_HEADER, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        mission_path = self._mission_path(sequence_name)

        # Extracted frames and zarr groups are pure re-formats of the kept tarballs (rgb_<i> holds
        # real resized copies, never symlinks), re-extracted on demand by _ensure_extracted.
        if BENCHMARK_RETENTION != Retention.FULL:
            for tar_path in mission_path.glob("*/*.tar"):
                shutil.rmtree(tar_path.with_suffix(""), ignore_errors=True)

        # The mission folder is this sequence's own download (nothing is shared across sequences
        # or across the two rig splits, which keep separate benchmark folders).
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(mission_path, ignore_errors=True)


class GrandtourAlphasenseDataset(GrandtourDataset):
    """GrandTour (Alphasense monochrome stereo + BMI085 IMU rig) dataset helper for VSLAM-LAB benchmark."""

    CAMERAS = (("alphasense_front_left", "alphasense_front_left_caminfo"),
               ("alphasense_front_right", "alphasense_front_right_caminfo"))
    CAM_TYPE = "gray"
    DISTORTION_TYPE = "equid4"
    IMU_TOPIC = "alphasense_imu"
    BODY_SENSOR = "alphasense_imu"
    IMU_HZ: Final = 200.0  # "Bosch BMI085 200Hz" per the stream's own description

    def __init__(self, dataset_name: str = "grandtour-alphasense") -> None:
        super().__init__(dataset_name)

    def create_imu_csv(self, sequence_name: str) -> None:
        imu_dir = self._topic_dir(sequence_name, self.IMU_TOPIC)
        if imu_dir is None:
            raise FileNotFoundError(f"{sequence_name}: data/{self.IMU_TOPIC}.tar not downloaded")
        ts_ns = _seconds_to_ns(_read_zarr_array(imu_dir / "timestamp"))
        gyro = _read_zarr_array(imu_dir / "ang_vel")
        accel = _read_zarr_array(imu_dir / "lin_acc")
        header = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        rows = [[int(t), *map(float, w), *map(float, a)] for t, w, a in zip(ts_ns, gyro, accel)]
        write_csv_rows(self.imu_csv_path(sequence_name), header, rows)

    def _imu_calibration(self, sequence_name: str) -> list[dict[str, Any]]:
        # Noise densities come from the per-sample covariances the release ships with the stream
        # (constant diagonals, sigma_d^2 = sigma_c^2 * rate), which land within ~20% of the BMI085
        # datasheet. Random walks, saturation and priors aren't in the release - taken from the
        # same Alphasense Core hardware's manufacturer example (sevensense-robotics
        # alphasense_core_manual, files/example_7s_sensors_dont_use.yaml), as dataset_hilti2022.py
        # and dataset_ariel.py do for this unit. The IMU is the body frame, so T_BS is identity.
        imu_dir = self._topic_dir(sequence_name, self.IMU_TOPIC)
        if imu_dir is None:
            raise FileNotFoundError(f"{sequence_name}: data/{self.IMU_TOPIC}.tar not downloaded")
        gyro_var = float(_read_zarr_array(imu_dir / "ang_vel_cov")[0, 0, 0])
        accel_var = float(_read_zarr_array(imu_dir / "lin_acc_cov")[0, 0, 0])
        return [{
            "imu_name": "imu_0",
            "a_max": 150.0,
            "g_max": 7.5,
            "sigma_g_c": float(np.sqrt(gyro_var / self.IMU_HZ)),
            "sigma_a_c": float(np.sqrt(accel_var / self.IMU_HZ)),
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": 0.000266,
            "sigma_aw_c": 0.0043,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(self.IMU_HZ),
            "T_BS": np.eye(4),
        }]


class GrandtourZed2iDataset(GrandtourDataset):
    """GrandTour (Stereolabs ZED2i color stereo rig) dataset helper for VSLAM-LAB benchmark."""

    CAMERAS = (("zed2i_left_images", "zed2i_left_caminfo"),
               ("zed2i_right_images", "zed2i_right_caminfo"))
    CAM_TYPE = "rgb"
    DISTORTION_TYPE = "radtan4"
    IMU_TOPIC = None
    BODY_SENSOR = "zed2i_left_images"

    def __init__(self, dataset_name: str = "grandtour-zed2i") -> None:
        super().__init__(dataset_name)
