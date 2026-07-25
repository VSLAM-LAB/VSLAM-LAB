"""
Module: VSLAM-LAB - Datasets - dataset_<name>.py
- Author: <your name>
- Assisted by: <agent name, or None if no AI coding agent was involved>
- Version: 1.0
- Created: YYYY-MM-DD
- License: GPLv3 License

Keep this in sync with the new dataset's own YAML vslamlab_maintainer: block (SKILL.md step 3) -
same name, same assisted_by, same date.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final
from urllib.parse import urljoin

import numpy as np

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention


class TemplateDataset(DatasetVSLAMLAB):
    """<Display Name> dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "dataset_name_template") -> None:
        super().__init__(dataset_name)

        # Get download url
        # self.cfg (set by DatasetVSLAMLAB.__init__ above) is this dataset's already-parsed yaml —
        # don't reopen self.yaml_file here, just pull whichever source-specific field(s) it carries:
        #   website      -> self.url_download_root = self.cfg["url_download_root"]  (a root + filename
        #                   pattern serving a .zip/.tar/.7z); if each sequence has its own unrelated
        #                   URL instead of a shared root, use self.url_download_sequences =
        #                   self.cfg["url_download_sequences"]  (a dict keyed by sequence_name), Model:
        #                   dataset_s3li.py — keyed lookup, not a positionally-indexed list
        #   hugging-face -> self.hf_repo_id = self.cfg["hf_repo_id"]; if the repo is gated it needs a
        #                   token — see HUGGINGFACE_TOKEN in path_constants.py (falls back to the
        #                   HF_TOKEN env var)
        #   google-drive -> self.url_download_root = self.cfg["url_download_root"]  (a drive.google.com
        #                   share link, or a drive.usercontent.google.com pre-resolved direct-download URL)
        #   local        -> nothing to pull here; affected sequences carry sequence_location: local instead
        # A dataset can mix patterns per sequence (see dataset_strayscanner.py: HF-backed, with
        # local overrides for sequences the user must place manually).
        # Also pull any mode-specific fields a sibling YAML of the same modes carries (e.g.
        # depth_factor for rgbd, url_download_root_gt for a separate groundtruth archive, or
        # further url_download_<what-it-is> fields for extra assets the source splits into
        # separate downloads, e.g. url_download_timestamps in dataset_caves.py).
        self.url_download_root: str = self.cfg["url_download_root"]

        # self.target_resolution is already set by DatasetVSLAMLAB.__init__ (super().__init__()
        # above) directly from this same yaml's target_resolution field — nothing to do here.
        # See dataset_soneva.py/dataset_sweetcorals.py, which likewise don't re-read it.

        # Sequence nicknames
        # Short, human-friendly labels shown in CLI output, one per entry in self.sequence_names.
        # e.g. self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        # Fetch the raw sequence data and leave it under self.dataset_path / sequence_name,
        # in whatever shape the source ships it — the create_* hooks below normalize it into
        # VSLAM-LAB's standard layout. Skip re-downloading/re-decompressing if the target
        # already exists (see check_sequence_availability in DatasetVSLAMLAB.py).
        # Pick the implementation matching this dataset's download pattern:
        #   website      -> utilities.downloadFile(url, self.dataset_path) + decompressFile(...)
        #                   Model: dataset_7scenes.py
        #   hugging-face -> use utilities.py's hf_token() / ensure_hf_sequence_download() (which
        #                   wraps download_hf_snapshot()) against self.hf_repo_id — resumable,
        #                   idempotent per-sequence fetch+flatten, don't hand-roll this with
        #                   HfApi/HfFileSystem/snapshot_download directly. Model:
        #                   dataset_soneva.py, dataset_sweetcorals.py
        #   google-drive -> a share link (drive.google.com/...) needs gdown.download /
        #                   gdown.download_folder with a file/folder id, Model: dataset_hilti2026.py,
        #                   dataset_drunkards.py; a pre-resolved direct-download link
        #                   (drive.usercontent.google.com/download?...&confirm=t&...) already bypasses
        #                   Drive's interstitial page, so a plain downloadFile(url, self.dataset_path)
        #                   works instead, Model: dataset_tartanair.py
        #   local        -> no fetch; print "Sequence '{sequence_name}' is marked as 'local'. Please
        #                   ensure the data is available at {path}." and return (never exit()/crash —
        #                   this must only skip the one sequence, base-class integrity checks report
        #                   what's still missing). If every sequence is local, `sequence_location:
        #                   local` is a single YAML scalar and the message is unconditional, Model:
        #                   dataset_iphone.py, dataset_scannetplusplus.py. If only some sequences are
        #                   local, `sequence_location` is a YAML list (one entry per sequence_name)
        #                   read into self.sequence_location and indexed by sequence_name to decide
        #                   per sequence, Model: dataset_strayscanner.py.
        # Always pin down one of these four real patterns — don't leave the source undetermined.
        # "other" is not one to implement against; it's just what generate_dataset_table.py reports
        # when a dataset's source isn't declared through any of the YAML fields above (e.g. a URL
        # hardcoded in the .py file, as in dataset_euroc.py).
        return

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Normalize the raw downloaded images into rgb_0/ (plus rgb_1/ for stereo modes)
        # under self.dataset_path / sequence_name — renaming/moving files as needed so every
        # dataset exposes the same folder layout regardless of the source's original format.
        # Branch on self.target_resolution (not a separate resize flag) for every rgb_0/rgb_1
        # image:
        #   None     -> source images are already <= 640x480 (or the yaml's target_resolution was
        #               removed) - copy/link the file into rgb_0/ unresized (e.g. shutil.copy2),
        #               never round-trip it through PIL just to leave it the same size.
        #   not None -> scale the image down to match self.target_resolution's pixel area while
        #               preserving aspect ratio via utilities.compute_scaled_size(img.size,
        #               self.target_resolution), then save into rgb_0/.
        # rgbd modes also need depth_0/, following the same self.target_resolution branch as
        # rgb_0/rgb_1 above when the source needs resizing — but never resize a depth map with
        # PIL's LANCZOS (or any interpolating resample); that blends depth values across object
        # boundaries and corrupts the metric data. Use nearest-neighbor (e.g. PIL's Image.NEAREST,
        # or cv2.resize(..., interpolation=cv2.INTER_NEAREST)) instead, which just samples the
        # nearest source pixel per output pixel and keeps every depth value exact.
        # dataset_eth.py's depth_0/ is a plain rename with no resizing at all — not because rgbd
        # depth shouldn't be resized in general, but because ETH3D's source images are already
        # close enough to 640x480 that eth.yaml sets no target_resolution, so nothing (rgb or
        # depth) gets resized for this particular dataset.
        # Model: dataset_soneva.py/dataset_sweetcorals.py (HFColmapDatasetMixin.create_rgb_folder)
        # for the rgb_0 resize pattern; dataset_eth.py for depth_0/'s folder layout (unresized in
        # eth's case specifically, see above).
        return

    def create_rgb_csv(self, sequence_name: str) -> None:
        # Write rgb.csv: one row per frame, with the standardized header for this dataset's mode(s), e.g.
        #   mono   -> ts_rgb_0 (ns), path_rgb_0
        #   stereo -> ts_rgb_0 (ns), path_rgb_0, ts_rgb_1 (ns), path_rgb_1
        #   rgbd   -> ts_rgb_0 (ns), path_rgb_0, ts_depth_0 (ns), path_depth_0
        # Timestamps in nanoseconds; derive them from self.rgb_hz if the source ships none.
        # Write to a <name>.csv.tmp file first, then .replace() it onto the final path — the
        # atomic-write pattern used throughout Datasets/dataset_files/*.py.
        return

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # Write calibration.yaml via self.write_calibration_yaml(rgb=[...], rgbd=[...], imu=[...]),
        # one dict per camera/IMU (cam_model, focal_length, principal_point, T_BS, ...) — see
        # Datasets/DatasetVSLAMLAB_calibration.py for the exact dict shape expected per cam_model.
        # calibration_type from SKILL.md step 1 decides where the values come from:
        #   global       -> the same fixed values are written for every sequence
        #                   Model: dataset_7scenes.py (constant CAMERA_PARAMS)
        #   per-sequence -> parse this sequence's own calibration file
        #                   Model: dataset_eth.py, dataset_kitti.py, dataset_euroc.py
        return

    def create_imu_csv(self, sequence_name: str) -> None:
        # Only needed for a "-vi" mode (mono-vi/rgbd-vi/stereo-vi) — otherwise delete this
        # method and let the base class's no-op default apply.
        # Write imu_0.csv: one row per IMU sample — ts (ns), wx, wy, wz (rad/s), ax, ay, az (m/s^2).
        return

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # Write groundtruth.csv: ts (ns), tx (m), ty (m), tz (m), qx, qy, qz, qw — one row per pose.
        # Always create this file, even when groundtruth_available (SKILL.md step 1) is false for
        # this dataset — write just the header row with no data rows (an empty groundtruth.csv)
        # rather than deleting the method / leaving no file at all. Model: dataset_videos.py.
        sequence_path = self.dataset_path / sequence_name
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            # If groundtruth_available is true, write one row per pose here — parsed per
            # calibration_type (global/per-sequence), same as create_calibration_yaml.

        tmp.replace(groundtruth_csv)

    def remove_unused_files(self, sequence_name: str) -> None:
        # Delete raw/intermediate files left over after create_rgb_folder / create_rgb_csv /
        # create_calibration_yaml / create_groundtruth_csv have consumed them (e.g. the original
        # compressed archive, per-frame pose .txt files), so the benchmark directory only keeps
        # the standardized layout. Check BENCHMARK_RETENTION / Retention if this dataset should
        # keep raw files around at higher retention levels.
        return

    def get_download_issues(self, _):
        # Only implement this if the dataset has one of the known constraints that block
        # *automatic* download of a sequence (see Datasets/DatasetVSLAMLAB_issues.py):
        #   "complete_dataset"  -> can't be split into per-sequence downloads (pass size_gb)
        #   "api_token"         -> requires an API token (pass website, yaml_file)
        #   "huggingface_token" -> requires a Hugging Face token (pass website, yaml_file)
        #   "license_required"  -> requires accepting license terms on the dataset's page first
        # Otherwise leave unimplemented — it inherits the base class's no-op default (no issues).
        # For "api_token"/"huggingface_token": read it in __init__ as self.api_token =
        # cfg.get("api_token", "not_set") — never cfg["api_token"], a missing token must produce
        # this reported issue, not a KeyError crash at load time. Don't exit()/crash in __init__
        # either if it's missing; report it here (return the issue below) and let the pipeline
        # continue and warn, Model: dataset_madmax.py, dataset_squidle.py.
        # Return a list of dicts built via _get_dataset_issue(issue_id=..., dataset_name=self.dataset_name, ...).
        return
