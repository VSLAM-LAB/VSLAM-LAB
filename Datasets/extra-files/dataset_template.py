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
        # don't reopen self.yaml_file here.

        # Pull this dataset's source-specific field(s) into an attribute of the same name — the
        # exact field per download pattern is canonical in
        # Datasets/extra-files/closed_lists.md#download-type-download's YAML Field column; if the
        # pattern needs auth (api_token), that handling is canonical in
        # Datasets/extra-files/closed_lists.md#download-issues-download_issues. Not repeated here.
        # The fetch mechanism, gotchas, and Model: citations are canonical in
        # download_sequence_data below.
        #
        # Also pull any mode-specific fields that a same-mode sibling YAML carries (e.g.
        # depth_factor for rgbd, or url_download_root_gt for a separate groundtruth archive).
        # Model: dataset_euroc.py, dataset_kitti.py, dataset_eth.py.
        self.url_download_root: str = self.cfg["url_download_root"]

        # Sequence nicknames - short, human-friendly labels for CLI output, one per sequence_names
        # entry. self.sequence_nicknames already has a sensible default from DatasetVSLAMLAB.__init__
        # above (underscore -> space) - only assign it here if something fancier is genuinely needed
        # (truncation, dropping a shared prefix - see dataset_eth.py), never as a no-op restating the
        # default (dataset_tartanair.py used to, removed once spotted). If the transform matches a
        # substring that itself contains an underscore (e.g. TUM's "rgbd_dataset_freiburg" -> "fr"),
        # build it from self.sequence_names (raw), not self.sequence_nicknames - the underscore is
        # already gone by the time the default has run. See dataset_rgbdtum.py/dataset_7scenes.py.

    def download_sequence_data(self, sequence_name: str) -> None:
        # Fetch the raw sequence data into self.sequence_path(sequence_name), in whatever shape
        # the source ships it — the create_* hooks below normalize it into VSLAM-LAB's standard
        # layout. Skip re-downloading/re-decompressing if the target already exists (see
        # check_sequence_availability in DatasetVSLAMLAB.py).
        #
        # download (SKILL.md step 1) closed-list definition — the five real patterns, the "other"
        # caveat, mixed-pattern-per-sequence handling — is canonical in
        # Datasets/extra-files/closed_lists.md#download-type-download. Not repeated here.
        #
        # PROCEDURAL INSTRUCTIONS
        # - Completion marker: use a marker file (e.g. rgb_path / ".download_complete", touched
        #   only after every output file is written) to record a fully downloaded sequence — a
        #   plain rgb_path.exists() check can't tell "fully downloaded" apart from "crashed
        #   partway through". Model: dataset_squidle.py (api), dataset_rover.py's
        #   _ensure_data_exists (website), dataset_msd.py (hugging-face, single-named-file).
        # - Never stash per-sequence derived state on self here for a *different* hook to read
        #   later — each hook must be independently callable (SKILL.md step 8 tests them one at a
        #   time), with no guarantee which hook ran first. Give the later hook its own small
        #   helper that recomputes the value from sequence_name instead. Model: dataset_rover.py's
        #   _sequence_group_path(sequence_name) helper, called independently by both hooks
        #   (replaced an earlier class-level cache dict that broke this).
        #
        # WARNINGS
        # - Marker/substring collision, hit by dataset_videos.py/dataset_youtube.py: if the marker
        #   lives in the same flat directory a *different* hook later scans by substring match
        #   (e.g. `if sequence_name in p.name`), the marker's own filename (e.g.
        #   "<name>.mp4.download_complete") also matches — directory iteration order isn't
        #   guaranteed, so the scan can pick the marker instead of the real file. Exclude
        #   marker-suffixed names from any such search.
        return

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Normalize the raw downloaded images into rgb_0/ (plus rgb_1/ for stereo modes, depth_0/
        # for rgbd) under self.sequence_path(sequence_name), so every dataset exposes the same
        # folder layout regardless of the source's original format. Use
        # self.sequence_path(sequence_name)/self.rgb_path(sequence_name)/self.depth_path(sequence_name)
        # (DatasetVSLAMLAB base-class helpers) to build these paths — never hardcode the
        # 'rgb_0'/'depth_0' string literals yourself.
        #
        # raw_formats (SKILL.md step 1) closed-list definition — what each value means for turning
        # the raw download into rgb_0/rgb_1/depth_0, including the multi-value case — is canonical
        # in Datasets/extra-files/closed_lists.md#raw-format-raw_formats. Not repeated here.
        # ros1/ros2's exact utilities.run_rosbag_frame_extraction(...) call signature and its
        # idempotency-fix rationale are in that function's own docstring.
        #
        # PROCEDURAL INSTRUCTIONS — branch on self.target_resolution (not a separate resize flag)
        # for every rgb_0/rgb_1 image:
        # - None     -> source already <= 640x480 (or target_resolution was removed from the
        #               yaml) — copy/link unresized (e.g. shutil.copy2); never round-trip through
        #               PIL just to leave it the same size.
        # - not None -> utilities.compute_scaled_size(img.size, self.target_resolution) to
        #               preserve aspect ratio, then img.resize(target_size,
        #               Image.Resampling.LANCZOS) (avoid the legacy Image.LANCZOS alias).
        # rgbd modes also need depth_0/, same branch — but nearest-neighbor only (PIL's
        # Image.NEAREST, or cv2.resize(..., interpolation=cv2.INTER_NEAREST)), never LANCZOS or
        # any other interpolating resample, which blends depth values across object boundaries and
        # corrupts the metric data.
        #
        # Model: dataset_soneva.py/dataset_sweetcorals.py (HFColmapDatasetMixin.create_rgb_folder,
        # rgb_0 resize pattern); dataset_eth.py (depth_0/'s folder layout — a plain unresized
        # rename in eth's case specifically, since eth.yaml sets no target_resolution at all, not
        # because rgbd depth shouldn't be resized in general).
        return

    def create_rgb_csv(self, sequence_name: str) -> None:
        # Write rgb.csv: one row per frame, with the standardized header for this dataset's
        # mode(s) — mono: ts_rgb_0 (ns), path_rgb_0; stereo: + ts_rgb_1 (ns), path_rgb_1; rgbd: +
        # ts_depth_0 (ns), path_depth_0. Timestamps in nanoseconds; derive them from self.rgb_hz if
        # the source ships none. Build a `rows` list and write it via
        # utilities.write_csv_rows(path, header, rows) — the atomic write-then-replace pattern used
        # throughout Datasets/dataset_files/*.py.
        #
        # PROCEDURAL INSTRUCTIONS — rgbd only: check whether the source's RGB/depth pair is a
        # single hardware-synchronized capture or two independently-timestamped streams:
        # - synchronized -> rgb_0/ and depth_0/ already correspond 1:1 by capture order — list
        #                    both, sort, zip the two sorted filename lists by index. The common
        #                    case. Model: dataset_eth.py, dataset_nuim.py, dataset_replica.py,
        #                    dataset_7scenes.py.
        # - async        -> read each stream's real timestamps, associate by nearest-timestamp
        #                    match within a tolerance (pandas.merge_asof(..., direction="nearest",
        #                    tolerance=...)), dropping any frame with no close-enough match — a
        #                    naive index-zip would silently pair frames from different moments
        #                    (e.g. two independently-timestamped sensors, as with a Kinect's RGB
        #                    and depth cameras). Model: dataset_rgbdtum.py — the first (and, as of
        #                    this writing, only) dataset in this repo needing this.
        return

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # Write calibration.yaml via self.write_calibration_yaml(rgb=[...], rgbd=[...], imu=[...]),
        # one dict per camera/IMU (cam_model, focal_length, principal_point, T_BS, ...) — see
        # Datasets/DatasetVSLAMLAB_calibration.py for the exact dict shape expected per cam_model.
        # calibration_type and cam_model closed-list definitions: see
        # Datasets/extra-files/closed_lists.md#calibration-type-calibration_type and
        # Datasets/extra-files/closed_lists.md#camera-models-cam_models — not repeated here, only
        # the implementation guidance below.
        #
        # PROCEDURAL INSTRUCTIONS
        # - Rescale on resize: if self.target_resolution is set, focal_length/principal_point
        #   describe the source's native resolution, not resized rgb_0/rgb_1 — pass them through
        #   utilities.scale_intrinsics(focal_length, principal_point, native_size,
        #   self.target_resolution) first. native_size = a documented resolution field
        #   (dataset_hilti2026.py's cam_cfg["resolution"]) or, failing that, whatever image
        #   create_rgb_folder resizes from (dataset_ut_coda.py). No-op when target_resolution is
        #   None, so always safe to call. Model: dataset_hilti2026.py. Trap: see WARNINGS -
        #   native_size isn't always the declared one.
        # - colmap parsing lives here: raw_formats' colmap value -> cameras.bin
        #   (read_colmap_cameras) here for focal_length/principal_point/dimensions; images.bin
        #   (read_colmap_images) goes in create_groundtruth_csv instead. The one raw_formats value
        #   not covered by create_rgb_folder's breakdown. Model: dataset_soneva.py.
        #
        # WARNINGS
        # - cam_model must match what's written: don't declare pinhole while writing real
        #   distortion coefficients (dataset_sesoko.yaml), and don't let a helper return
        #   cam_model/distortion_type as two values that can diverge (dataset_youtube.py).
        # - Cast to float: write_calibration_yaml f-string-embeds whatever type it's given — an
        #   uncast value from a raw text/CSV source silently becomes a quoted string in
        #   calibration.yaml, parsing back as str, not float, with no error anywhere.
        #   dataset_kitti.py hit this (fx/fy/cx/cy via .split(), never cast).
        # - native_size isn't always the declared one (see PROCEDURAL INSTRUCTIONS - Rescale on
        #   resize): a calibration source's declared width/height can disagree with what
        #   create_rgb_folder actually resizes from. dataset_soneva.py/dataset_sweetcorals.py's
        #   COLMAP reconstruction declares a size that doesn't match the real JPEGs, so they read
        #   the resized rgb_0 image's real dimensions off disk instead (see
        #   HFColmapDatasetMixin._pinhole_rgb_calibration). Verify against real data, don't assume
        #   (issue #99).
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
        # rather than deleting the method / leaving no file at all. Model: dataset_rgbdtum.py
        # (TUM's "validation" sequences ship no public groundtruth).
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            # If groundtruth_available is true, write one row per pose here — parsed per
            # calibration_type (Datasets/extra-files/closed_lists.md#calibration-type-calibration_type),
            # same as create_calibration_yaml.

        tmp.replace(groundtruth_csv)

    def remove_unused_files(self, sequence_name: str) -> None:
        # Delete files that create_rgb_folder / create_rgb_csv / create_calibration_yaml /
        # create_groundtruth_csv have already consumed and turned into the standardized layout
        # (rgb_0/, rgb.csv, calibration.yaml, groundtruth.csv, ...), so the benchmark directory
        # doesn't keep redundant copies of the same data in two formats.
        #
        # BENCHMARK_RETENTION (SKILL.md step 4h) closed-list definition — the three tiers' exact
        # meanings, the `if BENCHMARK_RETENTION != Retention.FULL` / `== Retention.MINIMAL` code
        # shape, and Model: citations — is canonical in
        # Datasets/extra-files/closed_lists.md#benchmark-retention-benchmark_retention. Not
        # repeated here, only the implementation gotchas below.
        #
        # WARNINGS
        # - Symlinked raw folder: MINIMAL's "delete un-resized raw images" clause only applies
        #   when create_rgb_folder copied those images into place (e.g. HFColmapDatasetMixin's
        #   rgb_0_raw/, disposable precisely because it's a copy). If create_rgb_folder instead
        #   symlinks rgb_0/depth_0/rgb_1 directly onto the raw source folder (Model:
        #   dataset_openloris.py, dataset_ut_coda.py's 2d_rect/), that raw folder must never be
        #   deleted at *any* retention tier, including MINIMAL — doing so leaves the standardized
        #   layout's own symlinks dangling. Check which one your create_rgb_folder does first.
        # - Unlink-path mismatch: whatever path you unlink() here must be *exactly* where
        #   download_sequence_data actually wrote the file — dataset_tartanair.py/dataset_nuim.py
        #   both once unlinked VSLAMLAB_BENCHMARK / <archive> here while the archive was actually
        #   downloaded to self.dataset_path / <archive> (one level deeper). unlink(missing_ok=True)
        #   silently swallows the mismatch, so nothing ever actually got deleted at MINIMAL and it
        #   went unnoticed until a later cleanup pass. Double-check this path against the one
        #   download_sequence_data/download_process actually built.
        # - rmtree vs. unlink, hit by dataset_euroc.py/dataset_kitti.py: use
        #   shutil.rmtree(path, ignore_errors=True) for a directory — unlink() on one raises
        #   IsADirectoryError. Compounds with the path-mismatch bug above: while the path is
        #   wrong, missing_ok=True masks it as a silent no-op; once the path gets fixed, it
        #   becomes a hard crash instead.
        #
        # SHARED-ARCHIVE SCOPING — an archive shared across multiple sequences needs different
        # handling depending on scope:
        # - whole-dataset: source can't be split into per-sequence downloads at all
        #   (get_download_issues' "complete_dataset" case) — do the cleanup in an overridden
        #   download_process instead of here, after the loop over every sequence finishes, never
        #   in remove_unused_files itself (runs per sequence). Model: dataset_tartanair.py,
        #   dataset_replica.py.
        # - scene/group: only a *subset* of sequences share one archive (e.g. 7-Scenes: chess.zip
        #   shared by chess_seq-01..06, each still independently downloadable) — delete only this
        #   sequence's exclusive sub-file safely; the shared file itself is also fine to delete
        #   early *if* download_sequence_data re-downloads it on demand — verify that fallback
        #   exists first (don't coordinate cleanup across sequences in a dataset-wide
        #   download_process override instead, that breaks requesting one sequence in isolation).
        #   Model: dataset_7scenes.py. Caveat, hit by dataset_rover.py's shared groundtruth.txt:
        #   the "re-downloads on demand" fallback usually only covers the whole group folder being
        #   gone entirely, not a single file deleted from an otherwise-intact group — a
        #   still-present completion marker short-circuits re-extraction. If you can't tell
        #   whether every sibling has already consumed the file, don't delete it.
        # - dataset-wide, indefinitely reused: some shared resources aren't scoped to one download
        #   batch at all (e.g. a small reference/calibration file re-read on every
        #   create_calibration_yaml call — Model: dataset_rover.py's master_calibration_path). The
        #   archive that produced it is safe to delete once extracted; never delete the
        #   *decompressed* resource itself, at any tier — that would force a wasteful re-download
        #   for the next sequence that needs it (self-healing, but not real "cleanup").
        # - exact-file share (multi-sequence): two or more sequences point at the literal *same*
        #   file (not just a shared archive — e.g. two time-windowed clips cut from one source
        #   video) — safe to delete only once every sequence sharing it has been processed; check
        #   each sibling's own completed state directly (e.g. self.rgb_path(sibling).exists())
        #   rather than assuming a re-download-on-demand fallback covers it. Model:
        #   dataset_youtube.py.
        return

    def get_download_issues(self, _):
        # Only implement this if the dataset has one of the known constraints that blocks
        # *automatic* download of a sequence (see Datasets/DatasetVSLAMLAB_issues.py). Otherwise
        # leave unimplemented — it inherits the base class's no-op default (no issues).
        #
        # download_issues (SKILL.md step 1) closed-list definition — the four known values'
        # meanings, the _get_dataset_issue kwargs each one needs, Model: citations, and the
        # "verify with a live check, don't copy from a same-pattern sibling" caveat — is canonical
        # in Datasets/extra-files/closed_lists.md#download-issues-download_issues. Not repeated
        # here.
        #
        # Return a list of dicts built via _get_dataset_issue(issue_id=..., dataset_name=self.dataset_name, ...).
        return
