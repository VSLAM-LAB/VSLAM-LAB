from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final
from urllib.parse import urljoin

import numpy as np
import yaml

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, Retention

class DATASET_NAME_TEMPLATE_dataset(DatasetVSLAMLab):
    """DATASET_NAME_TEMPLATE dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path):
        super().__init__('dataset_name_template', benchmark_path)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        # Pull out whichever source-specific field(s) this dataset's YAML carries:
        #   website      -> self.url_download_root = cfg["url_download_root"]  (a root + filename
        #                   pattern serving a .zip/.tar/.7z); if each sequence has its own unrelated
        #                   URL instead of a shared root, use self.url_download_sequences =
        #                   cfg["url_download_sequences"]  (a dict keyed by sequence_name), Model:
        #                   dataset_s3li.py — keyed lookup, not a positionally-indexed list
        #   hugging-face -> self.repo_id = cfg["repo_id"]; if the repo is gated it needs a token —
        #                   see HUGGINGFACE_TOKEN in path_constants.py (falls back to the HF_TOKEN env var)
        #   google-drive -> self.url_download_root = cfg["url_download_root"]  (a drive.google.com
        #                   share link, or a drive.usercontent.google.com pre-resolved direct-download URL)
        #   local        -> nothing to pull here; affected sequences carry sequence_location: local instead
        # A dataset can mix patterns per sequence (see dataset_strayscanner.py: HF-backed, with
        # local overrides for sequences the user must place manually).
        # Also pull any mode-specific fields a sibling YAML of the same modes carries (e.g.
        # depth_factor for rgbd, url_download_root_gt for a separate groundtruth archive, or
        # further url_download_<what-it-is> fields for extra assets the source splits into
        # separate downloads, e.g. url_download_timestamps in dataset_caves.py).
        self.url_download_root: str = cfg["url_download_root"]

        # Get resolution size
        # Only needed if resize (step 1) is true — the (width, height) images are downscaled
        # to before use, matching this target's pixel area while preserving aspect ratio.
        # Delete this line entirely if resize is false (source images are already <= 640x480).
        # Model: dataset_sweetcorals.py/.yaml (target_resolution: [640, 480]).
        self.target_resolution: tuple[int, int] = tuple(cfg["target_resolution"])

        # Sequence nicknames
        # Short, human-friendly labels shown in CLI output, one per entry in self.sequence_names.
        # e.g. self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        # Fetch the raw sequence data and leave it under self.dataset_path / sequence_name,
        # in whatever shape the source ships it — the create_* hooks below normalize it into
        # VSLAM-LAB's standard layout. Skip re-downloading/re-decompressing if the target
        # already exists (see check_sequence_availability in DatasetVSLAMLab.py).
        # Pick the implementation matching this dataset's download pattern:
        #   website      -> utilities.downloadFile(url, self.dataset_path) + decompressFile(...)
        #                   Model: dataset_7scenes.py
        #   hugging-face -> HfApi / HfFileSystem from huggingface_hub, using self.repo_id
        #                   Model: dataset_ariel.py
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
        # resize from step 1 decides what happens to each image on the way in:
        #   true  -> source images are bigger than 640x480; scale each one down to match
        #            self.target_resolution's pixel area while preserving aspect ratio before
        #            writing it into rgb_0/. Model: dataset_sweetcorals.py (create_rgb_folder +
        #            the _compute_scaled_size helper).
        #   false -> source images are already <= 640x480; copy/link them into rgb_0/ unresized.
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
        # Datasets/DatasetVSLAMLab_calibration.py for the exact dict shape expected per cam_model.
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
        # Only implement this if groundtruth_available (from SKILL.md step 1) is true for this
        # dataset — otherwise delete this method and let the base class's no-op default apply
        # (create_groundtruth_csv will simply have nothing to write).
        # Write groundtruth.csv: ts (ns), tx (m), ty (m), tz (m), qx, qy, qz, qw — one row per pose.
        return

    def remove_unused_files(self, sequence_name: str) -> None:
        # Delete raw/intermediate files left over after create_rgb_folder / create_rgb_csv /
        # create_calibration_yaml / create_groundtruth_csv have consumed them (e.g. the original
        # compressed archive, per-frame pose .txt files), so the benchmark directory only keeps
        # the standardized layout. Check BENCHMARK_RETENTION / Retention if this dataset should
        # keep raw files around at higher retention levels.
        return

    def get_download_issues(self, _):
        # Only implement this if the dataset has one of the known constraints that block
        # *automatic* download of a sequence (see Datasets/DatasetVSLAMLab_issues.py):
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
