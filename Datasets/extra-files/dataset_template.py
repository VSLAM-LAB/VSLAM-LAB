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

        # Pull this dataset's source-specific field(s) into an attribute of the same name,
        # one per download pattern:
        #   website/api  -> self.url_download_root = self.cfg["url_download_root"]
        #   hugging-face -> self.hf_repo_id = self.cfg["hf_repo_id"]
        #   google-drive -> self.google_drive_link = self.cfg["google_drive_link"]
        #   api (only if auth needed) -> self.api_token = self.cfg.get("api_token", "not_set") —
        #                                never cfg["api_token"]; a missing token must surface via
        #                                get_download_issues below, not a KeyError at load time
        #   local        -> nothing to pull here; sequences carry sequence_location instead
        #
        # Field name/shape per pattern is canonical in dataset_template.yaml; the fetch mechanism,
        # gotchas, and Model: citations are canonical in download_sequence_data below.
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
        # Completion marker: use a marker file (e.g. rgb_path / ".download_complete", touched only
        # after every output file is written) to record a fully downloaded sequence. A plain
        # rgb_path.exists() check can't tell "fully downloaded" apart from "crashed partway
        # through, folder exists but incomplete":
        #   - Model: dataset_squidle.py (api), dataset_rover.py's _ensure_data_exists (website),
        #     dataset_msd.py's download_sequence_data (hugging-face, single-named-file).
        #   - Caveat, hit by dataset_videos.py/dataset_youtube.py: if the marker lives in the same
        #     flat directory a *different* hook later scans by substring match (e.g.
        #     `if sequence_name in p.name`), the marker's own filename (e.g.
        #     "<name>.mp4.download_complete") also matches that substring — directory iteration
        #     order isn't guaranteed, so the scan can pick the marker instead of the real file.
        #     Exclude marker-suffixed names from any such search.
        #
        # Don't stash per-sequence derived state on self here for a *different* hook to read later
        # (e.g. a parsed sub-path some other create_* hook needs) — each hook must be independently
        # callable (SKILL.md step 8 tests them one at a time), with no guarantee which hook ran
        # first on a given instance. Give a later hook its own small helper that recomputes the
        # value from sequence_name instead. dataset_rover.py's RoverDataset used to cache a computed
        # sequence_group_path in a class-level dict just for create_groundtruth_csv to read —
        # removed in favor of a _sequence_group_path(sequence_name) helper both hooks call
        # independently.
        #
        # Pick the implementation matching this dataset's download pattern:
        #   website      -> utilities.downloadFile(url, self.dataset_path) + decompressFile(...).
        #                   Also covers a *pre-resolved* drive.usercontent.google.com
        #                   direct-download link (...?...&confirm=t&...) — it already bypasses
        #                   Drive's virus-scan interstitial, so a plain downloadFile works, no
        #                   gdown needed.
        #                   Model: dataset_7scenes.py; dataset_tartanair.py (pre-resolved link).
        #   hugging-face -> authenticate via utilities.py's hf_token() (HUGGINGFACE_TOKEN in
        #                   path_constants.py, falling back to the HF_TOKEN env var), then:
        #                     - directory of many ready-to-use files ->
        #                       ensure_hf_sequence_download() (wraps download_hf_snapshot())
        #                       against self.hf_repo_id — resumable, idempotent per-sequence
        #                       fetch+flatten.
        #                       Model: dataset_soneva.py, dataset_sweetcorals.py.
        #                     - one specific named file (e.g. an archive needing further
        #                       decompression, or a single metadata/calibration file) ->
        #                       huggingface_hub.hf_hub_download(repo_id=self.hf_repo_id,
        #                       filename=..., repo_type='dataset', token=hf_token()) directly —
        #                       ensure_hf_sequence_download()'s snapshot+flatten machinery doesn't
        #                       fit a single file.
        #                       Model: dataset_soneva.py's _fetch_colmap_file, dataset_openloris.py.
        #                   Don't hand-roll either case with HfApi/HfFileSystem/snapshot_download.
        #   google-drive -> a real share link (drive.google.com/...) needs gdown.download /
        #                   gdown.download_folder with a file/folder id — Drive shows a virus-scan
        #                   interstitial for most files this size that a plain download can't click
        #                   through. If a plain HTTP GET works on the URL, it's the website case
        #                   above instead, not this one.
        #                   Model: dataset_hilti2026.py, dataset_drunkards.py.
        #   local        -> no fetch; print "Sequence '{sequence_name}' is marked as 'local'. Please
        #                   ensure the data is available at {path}." and return (never exit()/crash
        #                   — this must only skip the one sequence; base-class integrity checks
        #                   report what's still missing).
        #                     - all sequences local -> sequence_location: local is a single YAML
        #                       scalar, message unconditional.
        #                       Model: dataset_iphone.py, dataset_scannetplusplus.py.
        #                     - only some sequences local -> sequence_location is a YAML list (one
        #                       entry per sequence_name), read into self.sequence_location and
        #                       indexed by sequence_name to decide per sequence.
        #                       Model: dataset_strayscanner.py.
        #   api          -> paginated JSON requests against self.url_download_root, not
        #                   downloadFile/decompressFile. Per-item metadata (pose/timestamp) often
        #                   lives only in the API response with no raw sidecar file — if so,
        #                   fetch/parse/write rgb.csv and groundtruth.csv directly here rather than
        #                   leaving that job for create_rgb_csv/create_groundtruth_csv (which can
        #                   then legitimately stay no-ops). Guard each per-item parse in its own
        #                   try/except — a single malformed item should be skipped with a warning,
        #                   not crash the whole sequence.
        #                   Model: dataset_squidle.py.
        #
        # A dataset can mix patterns per sequence (see dataset_strayscanner.py: HF-backed, with
        # local overrides for sequences the user must place manually).
        #
        # Always pin down one of these five real patterns — don't leave the source undetermined.
        # "other" is not one to implement against; it's just what generate_dataset_table.py reports
        # when a dataset's source isn't declared through any YAML field above (e.g. a URL hardcoded
        # in the .py file, as in dataset_euroc.py).
        return

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Normalize the raw downloaded images into rgb_0/ (plus rgb_1/ for stereo modes)
        # under self.sequence_path(sequence_name) — renaming/moving files as needed so every
        # dataset exposes the same folder layout regardless of the source's original format.
        # Use self.sequence_path(sequence_name)/self.rgb_path(sequence_name)/self.depth_path(sequence_name)
        # (DatasetVSLAMLAB base-class helpers) to build these folder paths — never hardcode the
        # 'rgb_0'/'depth_0' string literals yourself, even though that's what those helpers
        # resolve to under the hood.
        #
        # Branch on self.target_resolution (not a separate resize flag) for every rgb_0/rgb_1
        # image:
        #   None     -> source images are already <= 640x480 (or the yaml's target_resolution was
        #               removed) - copy/link the file into rgb_0/ unresized (e.g. shutil.copy2),
        #               never round-trip it through PIL just to leave it the same size.
        #   not None -> scale the image down to match self.target_resolution's pixel area while
        #               preserving aspect ratio via utilities.compute_scaled_size(img.size,
        #               self.target_resolution), then img.resize(target_size,
        #               Image.Resampling.LANCZOS) (the modern, non-deprecated Pillow API - avoid
        #               the legacy Image.LANCZOS alias) and save into rgb_0/.
        # rgbd modes also need depth_0/, following the same self.target_resolution branch as
        # rgb_0/rgb_1 above when the source needs resizing. Use nearest-neighbor only (e.g. PIL's
        # Image.NEAREST, or cv2.resize(..., interpolation=cv2.INTER_NEAREST)), which just samples
        # the nearest source pixel per output pixel and keeps every depth value exact — never
        # PIL's LANCZOS or any other interpolating resample, which blends depth values across
        # object boundaries and corrupts the metric data.
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
        # Build a `rows` list and write it via utilities.write_csv_rows(path, header, rows) — the
        # atomic write-then-replace pattern used throughout Datasets/dataset_files/*.py.
        #
        # rgbd only: check whether the source's RGB/depth pair is a single hardware-synchronized
        # capture or two independently-timestamped streams before picking how to associate them:
        #   synchronized -> rgb_0/ and depth_0/ already correspond 1:1 by capture order - list both,
        #                   sort, and zip the two sorted filename lists by index. This is the common
        #                   case. Model: dataset_eth.py, dataset_nuim.py, dataset_replica.py,
        #                   dataset_7scenes.py.
        #   async        -> read each stream's real timestamps and associate them by
        #                   nearest-timestamp match within a tolerance (e.g. pandas.merge_asof(...,
        #                   direction="nearest", tolerance=...)), dropping any frame with no match
        #                   close enough in time. Each stream carries its own independent
        #                   timestamps here (e.g. two separate sensors, as with a Kinect's RGB and
        #                   depth cameras) - a naive index-zip would silently pair frames from
        #                   different moments.
        #                   Model: dataset_rgbdtum.py - the first (and, as of this writing, only)
        #                   dataset in this repo needing this.
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
        # The cam_model you write must match the yaml's cam_models value, and both must describe
        # what's actually being written, not just "this is a perspective camera":
        #   pinhole             -> zero distortion - omit distortion_type/distortion_coefficients
        #                          entirely. Model: dataset_eth.py.
        #   radtan4/radtan5/    -> pinhole + that distortion model's real, trusted
        #   equid4                 distortion_coefficients. Model: dataset_eiffel_tower.py (radtan4).
        #   unknown             -> no verified calibration exists at all - zero focal_length/
        #                          principal_point, no distortion fields. Model: dataset_sweetcorals.py's
        #                          non-pinhole branch.
        # Get this wrong and dataset_table.md's Camera Models column silently misreports the
        # calibration's actual trustworthiness - dataset_sesoko.yaml once declared pinhole while
        # writing real radtan4 distortion coefficients, caught only in a later cleanup pass.
        # A related trap: collapse the camera model and distortion type into one indivisible value,
        # used both as cam_model and to decide whether distortion_type/distortion_coefficients get
        # written at all — don't have your own calibration-parameter helper return them as two
        # separately-returned values that can diverge (e.g. one saying pinhole, the other saying
        # radtan4 with real coefficients). dataset_youtube.py hit exactly this, caught in a later
        # cleanup pass.
        # If the source is a raw text/CSV file (not already-typed YAML), cast every numeric value to
        # float(...) before putting it in the calibration dict — write_calibration_yaml just
        # f-string-embeds whatever type it's given, so an uncast string silently gets written into
        # calibration.yaml as a quoted string (e.g. focal_length: ['718.856000', ...]), which then
        # parses back as str, not float, with no error anywhere in the pipeline. dataset_kitti.py hit
        # exactly this bug (fx/fy/cx/cy read via .split() and never cast) before it was caught.
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
            # calibration_type (global/per-sequence), same as create_calibration_yaml.

        tmp.replace(groundtruth_csv)

    def remove_unused_files(self, sequence_name: str) -> None:
        # Delete files that create_rgb_folder / create_rgb_csv / create_calibration_yaml /
        # create_groundtruth_csv have already consumed and turned into the standardized layout
        # (rgb_0/, rgb.csv, calibration.yaml, groundtruth.csv, ...), so the benchmark directory
        # doesn't keep redundant copies of the same data in two formats. How much gets deleted
        # depends on BENCHMARK_RETENTION (path_constants.py, default Retention.STANDARD) — a
        # three-level Enum every dataset's remove_unused_files should follow the same way:
        #
        #   Retention.FULL     -> delete nothing. Keep every raw/intermediate file exactly as
        #                         downloaded/generated, alongside the standardized layout.
        #   Retention.STANDARD -> delete intermediate files that are pure reformats of data
        #                         already fully captured in the standardized layout - no
        #                         information is lost by deleting them (e.g. eth.py's per-frame
        #                         calibration.txt/groundtruth.txt/rgb.txt/depth.txt/associated.txt,
        #                         once parsed into calibration.yaml/groundtruth.csv/rgb.csv). Keep
        #                         the *original source* downloads (zip archives, un-resized raw
        #                         images) - they're not redundant, since re-deriving the
        #                         standardized layout from them again (e.g. at a different
        #                         target_resolution) would otherwise require re-downloading.
        #   Retention.MINIMAL  -> delete everything STANDARD does, plus the original source
        #                         downloads too (e.g. eth.py's downloaded {sequence}_{mode}.zip,
        #                         HFColmapDatasetMixin's rgb_0_raw/ pre-resize images) - only the
        #                         standardized layout remains: smallest footprint, but not
        #                         reprocessable without a fresh download.
        #
        # Caveat on MINIMAL's "un-resized raw images" clause above: it only applies when
        # create_rgb_folder copied those images into place (e.g. HFColmapDatasetMixin's
        # rgb_0_raw/, disposable precisely because it's a copy). If create_rgb_folder instead
        # symlinks rgb_0/depth_0/rgb_1 directly onto the raw source folder (Model:
        # dataset_openloris.py, dataset_ut_coda.py's 2d_rect/), that raw folder must never be
        # deleted at *any* retention tier, including MINIMAL - doing so leaves the standardized
        # layout's own rgb_0/depth_0 symlinks dangling. Check which one your create_rgb_folder
        # actually does before writing this method.
        #
        # In code this is almost always exactly two checks:
        #   if BENCHMARK_RETENTION != Retention.FULL: <delete STANDARD-tier files>
        #   if BENCHMARK_RETENTION == Retention.MINIMAL: <delete MINIMAL-tier files too>
        # Model: dataset_eth.py (both tiers - raw .txt files vs. downloaded .zip archives),
        # HFColmapDatasetMixin.remove_unused_files in dataset_soneva.py (MINIMAL-only: rgb_0_raw/).
        #
        # Common bug: whatever path you unlink() here must be *exactly* where download_sequence_data
        # actually wrote the file - dataset_tartanair.py and dataset_nuim.py both once unlinked
        # VSLAMLAB_BENCHMARK / <archive> here while the archive was actually downloaded to
        # self.dataset_path / <archive> (one level deeper, inside the dataset's own folder).
        # unlink(missing_ok=True) silently swallows the mismatch, so nothing ever actually got
        # deleted at MINIMAL retention and it went unnoticed until a later cleanup pass. Double-check
        # this path against the one download_sequence_data/download_process actually built.
        #
        # Two more gotchas, both hit by dataset_euroc.py/dataset_kitti.py:
        #   - Always gate a STANDARD-tier delete with `if BENCHMARK_RETENTION != Retention.FULL:`
        #     — omitting it runs the delete unconditionally, deleting files even at Retention.FULL,
        #     which must delete nothing.
        #   - Use shutil.rmtree(path, ignore_errors=True) for a directory — calling unlink() on one
        #     raises IsADirectoryError instead. This compounds with the wrong-path bug above: while
        #     the path is wrong, missing_ok=True masks the mistake as a silent no-op; the moment the
        #     path gets fixed, it becomes a hard crash instead.
        #
        # An archive shared across multiple sequences needs different handling depending on scope:
        #   whole-dataset -> if the source can't be split into per-sequence downloads at all
        #                    (get_download_issues' "complete_dataset" case), do the shared-archive
        #                    cleanup in an overridden download_process instead of here, after the
        #                    loop over every sequence has finished - never in remove_unused_files
        #                    itself, since it runs per sequence. Model: dataset_tartanair.py,
        #                    dataset_replica.py.
        #   scene/group     -> if only a *subset* of sequences share one archive (e.g. 7-Scenes:
        #                      chess.zip is shared by chess_seq-01..06, but each sequence is still
        #                      independently downloadable), handle it in remove_unused_files: delete
        #                      only this sequence's own exclusive sub-file safely, and it's fine to
        #                      also delete the shared file itself (even before sibling sequences in
        #                      the same group are downloaded) *if* download_sequence_data
        #                      re-downloads it on demand when a later sequence needs it again - check
        #                      that fallback exists before relying on it. Don't try to coordinate
        #                      cleanup across sequences in a dataset-wide download_process override
        #                      instead - that would break requesting a single sequence in isolation,
        #                      which no longer loops over the whole dataset. Model: dataset_7scenes.py.
        #                      Caveat, hit by dataset_rover.py's shared groundtruth.txt (one per
        #                      location+date group, read by every sensor-class sub-sequence in
        #                      that group): the "re-downloads it on demand" fallback usually only
        #                      covers the whole group folder (plus its completion marker) being
        #                      gone entirely - it does *not* cover a single file being deleted
        #                      from an otherwise-intact group folder, since a still-present
        #                      completion marker short-circuits re-extraction. If you can't tell
        #                      whether every sibling sequence sharing that file has already
        #                      consumed it, don't delete it at all - leaving one small shared
        #                      file behind is safer than silently corrupting a not-yet-processed
        #                      sibling's output.
        #   dataset-wide,   -> some shared resources aren't scoped to one download batch at all -
        #   indefinitely       e.g. a small reference/calibration file every future sequence's
        #   reused             create_calibration_yaml re-reads on every call (Model:
        #                      dataset_rover.py's master_calibration_path). The archive that
        #                      produced it is still safe to delete once extracted, by the same
        #                      one-time-read reasoning as above. Never delete the *decompressed*
        #                      resource itself, at any retention tier - doing so would just force a
        #                      wasteful re-download for the very next sequence that needs it
        #                      (self-healing, but not a real "cleanup").
        #   exact-file      -> two or more sequences pointing at the literal *same* file (not just
        #   share              an archive covering several sequences, e.g. two time-windowed clips
        #   (multi-sequence)   cut from one shared source video) - deleting it is only safe once
        #                      every sequence sharing it has already been processed; check each
        #                      sibling's own already-completed state directly (e.g.
        #                      self.rgb_path(sibling).exists()) rather than assuming a
        #                      re-download-on-demand fallback covers it. Model: dataset_youtube.py.
        return

    def get_download_issues(self, _):
        # Only implement this if the dataset has one of the known constraints that block
        # *automatic* download of a sequence (see Datasets/DatasetVSLAMLAB_issues.py). Otherwise
        # leave unimplemented — it inherits the base class's no-op default (no issues).
        #
        #   "complete_dataset"  -> can't be split into per-sequence downloads (pass size_gb)
        #   "api_token"         -> requires an API token (pass website, yaml_file)
        #   "huggingface_token" -> requires a Hugging Face token (pass website, yaml_file)
        #   "license_required"  -> requires accepting license terms on the dataset's page first
        #
        # Per-issue implementation notes:
        #   api_token         -> read it in __init__ as self.api_token = cfg.get("api_token",
        #                        "not_set") — never cfg["api_token"]; a missing token must produce
        #                        this reported issue, not a KeyError crash at load time. If it's
        #                        missing, report it here (return the issue below) and let the
        #                        pipeline continue and warn — don't exit()/crash in __init__ either.
        #                        Model: dataset_madmax.py.
        #   huggingface_token -> there's no yaml field to read — check utilities.hf_token()
        #                        (HUGGINGFACE_TOKEN in path_constants.py, falling back to the
        #                        HF_TOKEN env var) directly here: `if hf_token() is not None:
        #                        return []`, else return the issue below.
        #                        Model: HFColmapDatasetMixin.get_download_issues
        #                        (dataset_soneva.py, shared by dataset_sweetcorals.py),
        #                        dataset_openloris.py.
        #
        # Before implementing any of these, verify the constraint actually exists — don't just
        # copy it from a sibling sharing the same download pattern:
        #   - Test the real access requirement directly (an anonymous API call, or a plain curl
        #     against a website URL), not by inference from the download pattern or a sibling's
        #     existing code.
        #   - dataset_msd.py's Hugging Face repo turned out fully public (confirmed via
        #     huggingface_hub.HfApi().dataset_info(repo_id, token=None).gated == False, plus an
        #     anonymous list_repo_files() call succeeding), so no huggingface_token issue was
        #     added for it.
        #   - This is unresolved even for the Model: examples above — soneva/sweetcorals/
        #     openloris's own repos also came back gated=False on the same live check, despite all
        #     three reporting this issue. See #91 before trusting any of them as a "definitely
        #     right" reference.
        #
        # Return a list of dicts built via _get_dataset_issue(issue_id=..., dataset_name=self.dataset_name, ...).
        return
