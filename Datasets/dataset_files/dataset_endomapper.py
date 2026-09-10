"""
Module: VSLAM-LAB - Datasets - dataset_endomapper.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2026-09-10
- Updated: 2026-09-10
- License: GPLv3 License
"""

from __future__ import annotations

import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import (
    compute_scaled_size,
    make_printers,
    read_colmap_images,
    scale_intrinsics,
    synapse_client,
    synapse_download_file,
    synapse_resolve_path,
    world_to_camera_to_pose,
    write_csv_rows,
)

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# Seq_<NNN> is a whole procedure, Seq_<NNN>_<MM> its COLMAP sub-model MM (see the yaml).
_SEQUENCE_NAME_RE: Final = re.compile(r"^(?P<procedure>Seq_\d{3})(?:_(?P<submodel>\d{2}))?$")
# Frame names inside the COLMAP export: 1-based video frame numbers (out<N>.png == frame N-1).
_COLMAP_FRAME_RE: Final = re.compile(r"^out(?P<number>\d+)\.png$")

# Layout of the local mirror of the Synapse project, kept inside the dataset folder (see the yaml).
_SEQUENCES_DIR: Final = "Sequences"
_CALIBRATIONS_DIR: Final = "Calibrations"
_COLMAP_META_DIRS: Final = ("meta-data", "colmap")
_COLMAP_INFO_TAG: Final = "Colmap Reconstructions"  # info json meta-data entry

# Per-sequence files download_sequence_data leaves in the sequence folder.
_RAW_VIDEO_LINK: Final = "raw.mov"  # symlink onto ../Sequences/Seq_<NNN>/Seq_<NNN>.mov
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


def _colmap_frame_index(colmap_name: str) -> int:
    """'out7199.png' (COLMAP's 1-based frame name) -> 7198 (0-based video frame index)."""
    match = _COLMAP_FRAME_RE.match(Path(colmap_name).name)
    if match is None:
        raise ValueError(f"Unexpected COLMAP frame name '{colmap_name}' - expected out<N>.png")
    return int(match["number"]) - 1


def _frame_name(frame_index: int) -> str:
    """rgb_0 file name of a video frame: zero-padded so lexicographic order is video order."""
    return f"{frame_index:06d}.png"


def _colmap_name(frame_index: int) -> str:
    """Inverse of _colmap_frame_index: 7198 -> 'out7199.png'."""
    return f"out{frame_index + 1}.png"


def _zip_root(zf: zipfile.ZipFile) -> str:
    """The COLMAP export's single top-level folder (Seq_001.zip: '33', an internal id)."""
    roots = {name.split("/", 1)[0] for name in zf.namelist() if "/" in name}
    if len(roots) != 1:
        raise ValueError(f"{zf.filename}: expected one top-level folder, found {sorted(roots)}")
    return roots.pop()


class EndomapperDataset(DatasetVSLAMLAB):
    """EndoMapper endoscopy dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "endomapper") -> None:
        super().__init__(dataset_name)

        self.dataset_homepage: str = self.cfg["api_url"]
        self.synapse_project_id: str = self.cfg["synapse_project_id"]
        # Local mirror of the Synapse project's Sequences/ and Calibrations/ folders, fetched on
        # demand into the dataset folder itself (next to the sequence folders) and reused as-is.
        self.mirror_path: Path = self.dataset_path

    def download_sequence_data(self, sequence_name: str) -> None:
        procedure, submodel = _split_sequence_name(sequence_name)
        video = self._raw_video(procedure)
        info_json = self._raw_info_json(procedure)
        if not (
            self._ensure_raw_file(video, _SEQUENCES_DIR, procedure, video.name)
            and self._ensure_raw_file(info_json, _SEQUENCES_DIR, procedure, info_json.name)
        ):
            return

        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)
        raw_link = sequence_path / _RAW_VIDEO_LINK
        if not (raw_link.is_symlink() or raw_link.exists()):
            # Relative target so the benchmark folder stays relocatable as a whole.
            os.symlink(os.path.relpath(video, sequence_path), raw_link)
        if not (sequence_path / info_json.name).exists():
            shutil.copy2(info_json, sequence_path / info_json.name)

        # The endoscope's calibration (none recorded for 17 procedures - see the yaml).
        endoscope = self._endoscope(sequence_name)
        if endoscope is not None:
            xml = self._raw_geometrical_xml(endoscope)
            if self._ensure_raw_file(xml, _CALIBRATIONS_DIR, xml.parent.name, xml.name) and not (sequence_path / xml.name).exists():
                shutil.copy2(xml, sequence_path / xml.name)

        # COLMAP poses (Seq_001/Seq_002 only): sparse/<M>/images.bin of this sequence's sub-model
        # (Seq_<NNN>: the largest one), extracted into a temp folder and renamed once complete.
        if not self._has_colmap(sequence_name):
            if submodel is not None:
                raise ValueError(f"{sequence_name}: {procedure} ships no COLMAP reconstructions - no sub-model sequences exist for it")
            return
        colmap_dir = self._colmap_dir(sequence_name)
        if (colmap_dir / "images.bin").is_file():
            return
        colmap_zip = self._raw_colmap_zip(procedure)
        if not self._ensure_raw_file(colmap_zip, _SEQUENCES_DIR, procedure, *_COLMAP_META_DIRS, colmap_zip.name):
            return
        tmp_dir = colmap_dir.with_name(colmap_dir.name + ".tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        with zipfile.ZipFile(colmap_zip) as zf:
            member = f"{_zip_root(zf)}/sparse/{self._submodel_index(sequence_name)}/images.bin"
            (tmp_dir / "images.bin").write_bytes(zf.read(member))
        tmp_dir.rename(colmap_dir)

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return

        first, last = self._frame_span(sequence_name)
        cap = self._open_video(sequence_name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if last is None or last >= total:
            if last is not None:
                print_warning(f"{sequence_name}: COLMAP registered frames up to {last} but the video has {total} - clipping")
            last = total - 1

        # Built in a sibling temp folder and renamed once complete, so a crash midway can't leave
        # a partial rgb_0/ that later looks finished.
        tmp_path = rgb_path.with_name(rgb_path.name + ".tmp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, first)  # frame-exact on these H.264 files (verified)
        target_size = None
        for index in tqdm(range(first, last + 1), desc=f"    extracting frames {first}-{last} -> {rgb_path.name}"):
            ok, frame = cap.read()
            if not ok:
                print_warning(f"{sequence_name}: video ended at frame {index - 1}, expected {last}")
                break
            if self.target_resolution is not None:
                if target_size is None:
                    target_size = compute_scaled_size((frame.shape[1], frame.shape[0]), self.target_resolution)
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(tmp_path / _frame_name(index)), frame)
        cap.release()

        tmp_path.rename(rgb_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return
        rgb_path = self.rgb_path(sequence_name)
        fps = self._video_fps(sequence_name)
        rows = [[_ts_ns(int(p.stem), fps), f"{rgb_path.name}/{p.name}"] for p in self._rgb_frames(sequence_name)]
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "fps": float(self._video_fps(sequence_name)),
            "T_BS": np.eye(4),
        }
        endoscope = self._endoscope(sequence_name)
        if endoscope is None:
            # No endoscope recorded for this procedure: no calibration exists to write.
            print_info(f"{sequence_name}: no endoscope recorded in its info json - writing an 'unknown' camera")
            rgb.update({"cam_model": "unknown", "focal_length": [0.0, 0.0], "principal_point": [0.0, 0.0]})
        else:
            # The endoscope's official EndoMapper calibration (geometrical xml), at the native
            # 1440x1080, rescaled to the size create_rgb_folder produced. Note the COLMAP
            # sub-models were reconstructed with a slightly different KB4 set (their cameras.bin:
            # fx 717.21 vs 717.69 here for Endoscope_01) - the published calibration is the one written.
            width, height, params = self._read_geometrical_xml(sequence_name, endoscope)
            fx, fy, cx, cy, k1, k2, k3, k4 = params
            self._check_calibration_resolution(sequence_name, (width, height))
            focal_length, principal_point = scale_intrinsics((fx, fy), (cx, cy), (width, height), self.target_resolution)
            rgb.update({
                "cam_model": "pinhole",
                "distortion_type": "equid4",
                "distortion_coefficients": [k1, k2, k3, k4],
                "focal_length": focal_length,
                "principal_point": principal_point,
            })
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # COLMAP poses of this sequence's sub-model (camera-in-world, the sub-model's own frame and
        # scale), one row per rgb_0 frame registered in it: the registered frames of a
        # Seq_<NNN>_<MM> clip, the largest sub-model's frames of a whole procedure Seq_<NNN>, and
        # nothing (header only) for the 91 procedures without COLMAP meta-data.
        images_bin = self._colmap_dir(sequence_name) / "images.bin"
        frames = self._rgb_frames(sequence_name)
        rows = []
        if images_bin.is_file():
            images = read_colmap_images(images_bin)
            fps = self._video_fps(sequence_name)
            for frame in frames:
                registered = images.get(_colmap_name(int(frame.stem)))
                if registered is None:
                    continue
                _, qvec, tvec = registered
                rows.append([_ts_ns(int(frame.stem), fps), *world_to_camera_to_pose(qvec, tvec)])
            print_info(f"{sequence_name}: COLMAP poses (sub-model {self._submodel_index(sequence_name)}) cover {len(rows)} of {len(frames)} frames")
        write_csv_rows(self.groundtruth_csv_path(sequence_name), _GROUNDTRUTH_HEADER, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        # colmap/images.bin is a copy of a zip member, fully turned into groundtruth.csv - gone at
        # STANDARD, re-extracted from the mirrored zip on demand. raw.mov is a symlink into the
        # Synapse mirror (the only copy of the video) and is never deleted, at any tier.
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(self._colmap_dir(sequence_name), ignore_errors=True)

    def get_download_issues(self, sequence_names: list[str]) -> list[dict]:
        # Only a problem when something must actually be fetched: a mirror that already holds the
        # requested videos needs no Synapse login at all.
        missing = [s for s in sequence_names if not self._raw_video(_split_sequence_name(s)[0]).is_file()]
        if not missing or synapse_client() is not None:
            return []
        return [_get_dataset_issue(issue_id="synapse_token", dataset_name=self.dataset_name, website=self.dataset_homepage)]

    # --- Synapse mirror paths and fetching -------------------------------------------------------
    def _raw_video(self, procedure: str) -> Path:
        return self.mirror_path / _SEQUENCES_DIR / procedure / f"{procedure}.mov"

    def _raw_info_json(self, procedure: str) -> Path:
        return self.mirror_path / _SEQUENCES_DIR / procedure / f"{procedure}_info.json"

    def _raw_colmap_zip(self, procedure: str) -> Path:
        return self.mirror_path / _SEQUENCES_DIR / procedure / Path(*_COLMAP_META_DIRS) / f"{procedure}.zip"

    def _raw_geometrical_xml(self, endoscope: int) -> Path:
        folder = f"Endoscope_{endoscope:02d}"
        return self.mirror_path / _CALIBRATIONS_DIR / folder / f"{folder}_geometrical.xml"

    def _ensure_raw_file(self, local: Path, *remote_names: str) -> bool:
        """True once `local` exists - fetching it from the Synapse project (remote_names: the
        entity names down from the project root) if it doesn't and credentials allow."""
        if local.is_file():
            return True
        syn = synapse_client()
        if syn is None:
            print_info(
                f"{local.name} is not in {local.parent} and no Synapse credentials are configured "
                f"(~/.synapseConfig or SYNAPSE_AUTH_TOKEN) - place the file there yourself (the folder mirrors "
                f"the Synapse project's layout), or set up the credentials (see `pixi run get-resources`)."
            )
            return False
        remote = "/".join(remote_names)
        file_id = synapse_resolve_path(syn, self.synapse_project_id, *remote_names)
        if file_id is None:
            print_warning(f"{remote} not found in Synapse project {self.synapse_project_id}")
            return False
        print_info(f"Downloading {remote} from Synapse ({file_id}) -> {local.parent}")
        synapse_download_file(syn, file_id, local)
        return True

    # --- helpers, all recomputed from sequence_name (no per-sequence state on self) -------------
    def _colmap_dir(self, sequence_name: str) -> Path:
        return self.sequence_path(sequence_name) / _COLMAP_DIR

    def _info(self, sequence_name: str) -> dict[str, Any]:
        """The procedure's info json - from the sequence folder if copied in, else the raw folder."""
        procedure, _ = _split_sequence_name(sequence_name)
        for info_json in (self.sequence_path(sequence_name) / f"{procedure}_info.json", self._raw_info_json(procedure)):
            if info_json.is_file():
                with open(info_json, encoding="utf-8") as f:
                    return json.load(f)
        raise FileNotFoundError(f"{sequence_name}: {procedure}_info.json not found - run download_sequence_data first")

    def _endoscope(self, sequence_name: str) -> int | None:
        """Endoscope number of the procedure, or None when none is recorded ("N/A" - the field is
        an int for some procedures, a digit string for others)."""
        value = str(self._info(sequence_name).get("endoscope_number", "")).strip()
        return int(value) if value.isdigit() else None

    def _has_colmap(self, sequence_name: str) -> bool:
        return _COLMAP_INFO_TAG in (self._info(sequence_name).get("meta-data") or [])

    def _submodel_index(self, sequence_name: str) -> int:
        """The COLMAP sub-model this sequence's poses come from: its own for Seq_<NNN>_<MM>, the
        largest cluster (most frames in cluster_list/<M>.txt, lowest M on a tie) for Seq_<NNN>."""
        procedure, submodel = _split_sequence_name(sequence_name)
        if submodel is not None:
            return submodel
        with zipfile.ZipFile(self._raw_colmap_zip(procedure)) as zf:
            prefix = f"{_zip_root(zf)}/cluster_list/"
            sizes = [
                (len(zf.read(name).split()), int(Path(name).stem))
                for name in zf.namelist()
                if name.startswith(prefix) and name.endswith(".txt")
            ]
        if not sizes:
            raise ValueError(f"{sequence_name}: no cluster_list/*.txt in {self._raw_colmap_zip(procedure)}")
        return max(sizes, key=lambda item: (item[0], -item[1]))[1]

    def _frame_span(self, sequence_name: str) -> tuple[int, int | None]:
        """(first, last) 0-based video frame indices rgb_0 covers: the whole video for Seq_<NNN>
        (last None = to the end), the sub-model's first..last registered frame for Seq_<NNN>_<MM>."""
        _, submodel = _split_sequence_name(sequence_name)
        if submodel is None:
            return 0, None
        indices = [_colmap_frame_index(name) for name in read_colmap_images(self._colmap_dir(sequence_name) / "images.bin")]
        return min(indices), max(indices)

    def _open_video(self, sequence_name: str) -> cv2.VideoCapture:
        raw_link = self.sequence_path(sequence_name) / _RAW_VIDEO_LINK
        cap = cv2.VideoCapture(str(raw_link))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open the video of '{sequence_name}' at {raw_link}: run download_sequence_data first")
        return cap

    def _video_fps(self, sequence_name: str) -> float:
        cap = self._open_video(sequence_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            raise ValueError(f"{sequence_name}: could not read the video frame rate")
        return float(fps)

    def _rgb_frames(self, sequence_name: str) -> list[Path]:
        rgb_path = self.rgb_path(sequence_name)
        return sorted(p for p in rgb_path.iterdir() if p.is_file() and p.suffix.lower() == ".png")

    def _read_geometrical_xml(self, sequence_name: str, endoscope: int) -> tuple[int, int, list[float]]:
        """(width, height, [fx, fy, cx, cy, k1, k2, k3, k4]) from the endoscope's calibu xml (the
        copy in the sequence folder, else the raw folder's)."""
        raw_xml = self._raw_geometrical_xml(endoscope)
        xml_path = next((p for p in (self.sequence_path(sequence_name) / raw_xml.name, raw_xml) if p.is_file()), None)
        if xml_path is None:
            raise FileNotFoundError(f"{sequence_name}: {raw_xml.name} not found - run download_sequence_data first")
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
        height, width = cv2.imread(str(frames[0])).shape[:2]
        if (width, height) != expected_size:
            print_warning(
                f"{sequence_name}: rgb_0/{frames[0].name} is {(width, height)}, but the calibration is scaled for "
                f"{expected_size} - intrinsics may describe the wrong image size."
            )


def _ts_ns(frame_index: int, fps: float) -> int:
    """Video frame index -> timestamp in ns at the video's frame rate (no timestamps ship)."""
    return int(round(frame_index * 1e9 / fps))
