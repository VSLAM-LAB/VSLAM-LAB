# VSLAM-LAB Closed-List Definitions

Canonical value → meaning definitions for every closed list used by `add-dataset`. This is the
only place these values are defined — `SKILL.md`, `dataset_template.py`, and `dataset_template.yaml`
all cross-reference this file rather than duplicating a definition; if you're implementing a hook
(not just choosing a value), see those files' own comments for the gotchas/warnings that go with
each definition below.

| Field | Resolved in |
|---|---|
| `modes` | SKILL.md step 1 |
| `cam_models` | SKILL.md step 1 |
| `raw_formats` | SKILL.md step 1 |
| `calibration_type` | SKILL.md step 1 |
| `download` | SKILL.md step 1 |
| `download_issues` | SKILL.md step 1 |
| `BENCHMARK_RETENTION` | `path_constants.py` — global constant, not chosen per dataset |
| `about.features` | SKILL.md step 7 — drives README.md's Datasets/Tools-table rendering |

The *current* closed-list values in use (which can grow over time) live in
`Datasets/extra-files/dataset_table.md`'s columns, generated from `Datasets/dataset_files/*.yaml`
via `pixi run generate-dataset-table` — read that live rather than trusting this file's examples to
stay exhaustive.

## Modes (`modes`)

Include the native mode(s) *and* every mode derivable by dropping a channel — `stereo`/`rgbd` →
`mono` (drop the second image/depth; stereo and rgbd don't reduce to each other), `-vi` →
non-`-vi` (drop the IMU stream, one-way only, never invent IMU data). E.g. native `stereo-vi` →
`['mono', 'mono-vi', 'stereo', 'stereo-vi']`; plain `stereo` → `['mono', 'stereo']`; plain `mono`
→ `['mono']`.

| Value      | Meaning                                                        |
|------------|-----------------------------------------------------------------|
| mono       | one RGB image per frame (rgb_0)                                |
| stereo     | two RGB images per frame (rgb_0 + rgb_1)                       |
| rgbd       | one RGB image + one depth map per frame (rgb_0 + depth_0)      |
| mono-vi / stereo-vi / rgbd-vi | the above, plus an IMU stream (imu_0.csv)           |

Applies dataset-wide, not per-sequence — see `add-dataset`'s SKILL.md step 1 for the
split-by-capability rule this implies when a source's sequences aren't all equally capable.

## Calibration type (`calibration_type`)

`calibration_type` decides where `create_calibration_yaml`'s values come from:

| Value        | Values come from                             | Model                                              |
|--------------|-----------------------------------------------|-----------------------------------------------------|
| global       | the same fixed values, for every sequence    | dataset_7scenes.py (constant CAMERA_PARAMS)        |
| per-sequence | parsing this sequence's own calibration file | dataset_eth.py, dataset_kitti.py, dataset_euroc.py |

## Camera Models (`cam_models`)

`cam_model` must describe what `create_calibration_yaml` actually writes, not just "this is a
perspective camera" — see `dataset_template.py`'s WARNINGS for the consistency gotchas this
invites getting wrong (live-included in `add-dataset`'s SKILL.md step 4, so no separate lookup
needed there).

| Value                       | Distortion written                                                                               | Model                                         |
|------------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------|
| pinhole                     | zero distortion - omit distortion_type/distortion_coefficients entirely                          | dataset_eth.py                                |
| radtan4 / radtan5 / equid4  | pinhole + that distortion model's real, trusted distortion_coefficients                          | dataset_eiffel_tower.py (radtan4)             |
| unknown                     | no verified calibration exists at all - zero focal_length/principal_point, no distortion fields  | dataset_sweetcorals.py's non-pinhole branch   |

## Raw Format (`raw_formats`)

The closed list of shapes the source can ship its sequence data in — not the same axis as
`download` below (`download` is the *transport*, `raw_formats` is the payload's *packaging* once
fetched; a `hugging-face`-hosted dataset can still ship a `ros2` bag). A dataset commonly needs
more than one value (`dataset_hilti2022.py`'s `['ros1', 'zip']`: a rosbag for the sequence data, a
separate zip for calibration). Each value describes what `create_rgb_folder` actually does to turn
that raw shape into `rgb_0`/`rgb_1`/`depth_0` — except `colmap`, which never produces rgb frames
at all (it's a calibration/pose source, handled in `create_calibration_yaml` instead).

| Value          | Turns into rgb_0/rgb_1/depth_0 via                                                                                                                                      | Model                                                    |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| zip / tar / 7z | already decompressed by download_sequence_data into a plain image folder — locate + copy/resize                                                                       | dataset_eth.py, dataset_kitti.py                          |
| ros1 / ros2    | utilities.run_rosbag_frame_extraction(...) per camera — never hand-roll the pixi extract-rosbag-frames/extract-ros2bag-frames call. ros2 on post-Humble (e.g. Jazzy) also needs utilities.patch_ros2_qos_profiles_metadata(...) first | dataset_hilti2022.py (ros1), dataset_pamir.py (ros2)       |
| video          | frame-extract via ffmpeg or cv2.VideoCapture, one image per frame                                                                                                      | dataset_youtube.py, dataset_strayscanner.py, dataset_scannetplusplus.py |
| images         | already individual, already-decoded frame files (HF snapshot, per-item API fetch) — no extraction, only copy/resize                                                   | dataset_soneva.py                                          |
| hdf5           | parse the image arrays directly out of the .h5 file (e.g. h5py) — no extraction subprocess                                                                            | dataset_nsavp.py                                           |
| local          | no-op — rgb_0/(rgb_1/) is whatever the user already placed at self.sequence_path(...), nothing to normalize                                                           | —                                                           |
| colmap         | not a create_rgb_folder concern — describes calibration/pose (cameras.bin/images.bin), never rgb frames; see Calibration type above instead                          | dataset_soneva.py (combined with images)                   |

## Download type (`download`)

Pick the implementation matching this dataset's download pattern. The YAML Field column is what
`generate_dataset_table.py`'s `_download_labels` actually keys off of to infer this value — website
and api look identical in the Implementation column's "root URL" shape, but each owns its own
field precisely so the generator (and a human skimming the YAML) can tell them apart without
reading the `.py` file:

| Value        | YAML Field                                  | Implementation                                                                                                                                                        | Model                                                                       |
|--------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| website      | `url_download_root` / `url_download_sequences` | utilities.downloadFile(url, self.dataset_path) + decompressFile(...) — also covers pre-resolved drive.usercontent.google.com direct-download links (no gdown needed) | dataset_7scenes.py; dataset_tartanair.py (pre-resolved link)               |
| hugging-face | `hf_repo_id`                                | hf_token() auth (HUGGINGFACE_TOKEN/HF_TOKEN) -> ensure_hf_sequence_download() for many ready-to-use files, or hf_hub_download(repo_id, filename, repo_type='dataset', token=hf_token()) for one named file. Never hand-roll with HfApi/HfFileSystem/snapshot_download. | dataset_soneva.py, dataset_sweetcorals.py (many files); dataset_soneva.py's _fetch_colmap_file, dataset_openloris.py (one file) |
| google-drive | `google_drive_link`                         | gdown.download / gdown.download_folder by file/folder id (bypasses Drive's virus-scan interstitial). Plain HTTP GET works? That's website instead.                  | dataset_hilti2026.py, dataset_drunkards.py                                 |
| local        | `sequence_location: local`                  | No fetch — print "Sequence marked as 'local'" notice, return (never exit()/crash). sequence_location: local as a scalar (all sequences local) or list (only some, indexed by sequence_name). | dataset_iphone.py, dataset_scannetplusplus.py (all local); dataset_strayscanner.py (some local) |
| api          | `api_url`                                   | Paginated JSON requests against self.api_url, not downloadFile/decompressFile. If the API is the only pose/timestamp source, write rgb.csv/groundtruth.csv here directly. Guard each item in try/except — skip malformed items, don't crash the sequence. | dataset_sesoko.yaml (field), dataset_squidle.py (SquidleDataset base class implementation) |

A dataset can mix patterns per sequence (see `dataset_strayscanner.py`: HF-backed, with local
overrides for sequences the user must place manually).

Always pin down one of these five real patterns — don't leave the source undetermined. `other` is
not one to implement against; it's just what `generate_dataset_table.py` reports when a dataset's
source isn't declared through any of the recognized YAML fields above (`hf_repo_id`,
`google_drive_link`, `url_download_root`/`url_download_sequences`, `api_url`,
`sequence_location: local`) — e.g. a URL hardcoded directly in the `.py` file instead of pulled
from `self.cfg`.

## Download Issues (`download_issues`)

A known constraint blocking *automatic* download of a sequence, reported by overriding
`get_download_issues` to return `_get_dataset_issue(issue_id=..., dataset_name=self.dataset_name,
...)` (`Datasets/DatasetVSLAMLAB_issues.py`) — leave blank/unimplemented (inherits the base
class's no-op default) when no such constraint exists; most datasets are blank.

| Value                | Meaning                                                                                                          | `_get_dataset_issue` kwargs | Model                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|------------------------------|--------------------------------------------------------------------------------------------|
| complete_dataset     | source can't be split into per-sequence downloads — the whole dataset (or group) must be fetched together. Still `automatic download` (green) — just wider-scoped than requested, not blocked | `size_gb`                    | dataset_kitti.py, dataset_euroc.py, dataset_replica.py, dataset_tartanair.py               |
| api_token            | requires an API token. Read it in `__init__` as `self.api_token = self.cfg.get("api_token", "not_set")` — never `cfg["api_token"]`, so a missing token surfaces here, not a `KeyError` at load time. `user action required` (yellow) | `website`, `yaml_file`       | dataset_madmax.py, dataset_squidle.py (SesokoDataset)                                      |
| huggingface_token    | requires a Hugging Face token. No YAML field to read — check `utilities.hf_token()` (`HUGGINGFACE_TOKEN` in `path_constants.py`, falling back to the `HF_TOKEN` env var) directly: `if hf_token() is not None: return []`. `user action required` (yellow) | `website`, `yaml_file`       | `HFColmapDatasetMixin.get_download_issues` (dataset_soneva.py, shared by dataset_sweetcorals.py), dataset_openloris.py |
| license_required     | requires accepting license terms on the dataset's page first, before any automated download can proceed. `user action required` (yellow) | none                          | —                                                                                            |

Before implementing any of these, confirm the constraint actually exists with a live check (an
anonymous API call, or a plain curl against a website URL) — don't copy it from a same-pattern
sibling's existing code just because the pattern matches. `dataset_msd.py`'s Hugging Face repo
turned out fully public (confirmed via `huggingface_hub.HfApi().dataset_info(repo_id,
token=None).gated == False`, plus an anonymous `list_repo_files()` call succeeding), so no
`huggingface_token` issue was added for it. This is unresolved even for the Model examples above —
soneva/sweetcorals/openloris's own repos also came back `gated=False` on the same live check,
despite all three reporting this issue. See
[#91](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/91) before trusting any of them as a
"definitely right" reference.

## Benchmark Retention (`BENCHMARK_RETENTION`)

Gated by `BENCHMARK_RETENTION` (`path_constants.py`, default `Retention.STANDARD`) — every
dataset's `remove_unused_files` deletes files `create_rgb_folder`/`create_rgb_csv`/
`create_calibration_yaml`/`create_groundtruth_csv` have already consumed and turned into the
standardized layout, so the benchmark directory doesn't keep redundant copies of the same data in
two formats. How much gets deleted depends on the tier:

| Value    | Meaning                                                                                                                                                          | Model                                                                     |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| FULL     | delete nothing — keep every raw/intermediate file exactly as downloaded/generated, alongside the standardized layout                                              | —                                                                          |
| STANDARD | delete intermediate files that are pure reformats of data already fully captured in the standardized layout — no information is lost by deleting them. Keep the *original source* downloads (zip archives, un-resized raw images) — re-deriving the standardized layout from them again (e.g. at a different `target_resolution`) would otherwise require re-downloading | dataset_eth.py (per-frame `calibration.txt`/`groundtruth.txt`/`rgb.txt`/`depth.txt`/`associated.txt`, once parsed) |
| MINIMAL  | delete everything STANDARD does, plus the original source downloads too — only the standardized layout remains: smallest footprint, but not reprocessable without a fresh download                | dataset_eth.py (downloaded `{sequence}_{mode}.zip`), `HFColmapDatasetMixin` in dataset_soneva.py (`rgb_0_raw/`) |

In code this is almost always exactly two checks:
```python
if BENCHMARK_RETENTION != Retention.FULL: ...   # delete STANDARD-tier files
if BENCHMARK_RETENTION == Retention.MINIMAL: ...  # delete MINIMAL-tier files too
```
Always gate the STANDARD-tier delete with the first check — omitting it deletes files even at
`Retention.FULL`, which must delete nothing (a real bug hit by `dataset_euroc.py`/
`dataset_kitti.py`).

The gotchas for actually implementing this correctly — symlinked raw folders that must survive
even `MINIMAL`, unlink-path mismatches, `shutil.rmtree` vs. `unlink()`, and the four shared-archive
scoping patterns (whole-dataset-only, scene/group-scoped, dataset-wide indefinitely reused,
exact-file share) — are canonical in `dataset_template.py`'s `remove_unused_files` comment. Not
repeated here.

## Dataset Features (`about.features`)

Word-form tags in each dataset's YAML `about.features` list. README.md's Datasets-table generator
(`generate_readme_datasets_table.py`) renders them as emoji and uses them to assign a section
divider (`_SECTION_PRIORITY`, first match wins: Synthetic, Underwater, Vehicle, UAV,
Intracorporeal, Planetary-analog Terrain, Construction Site, Robot, else the default section).
Also the emoji legend for README's hand-edited Tools table (SKILL.md step 7).

| Category            | Values                                                                 | Cardinality  |
|----------------------|-------------------------------------------------------------------------|---------------|
| Origin               | Real 📸, Synthetic 💻                                                   | exactly one   |
| Environment          | Indoor 🏠, Outdoor 🏞️, Underwater 🌊, Intracorporeal 🫀                  | one or more   |
| Platform             | Handheld 🤳, Headmounted 🥽, Vehicle 🚗, UAV 🚁, Robot 🤖                 | one or more   |
| Section tags (real per-dataset values, not divider-only) | Construction Site 🏗️, Planetary-analog Terrain 🪐 | optional |

Model: `dataset_hilti2022.yaml` (Construction Site), `dataset_madmax.yaml` (Planetary-analog
Terrain) — both tag the individual dataset directly, they aren't divider-only labels applied
externally. Ask the user if a value is unclear — don't guess.

Never hand-edit README's Datasets-table emoji/row — edit `about.features` in the YAML, then re-run
`generate_readme_datasets_table.py` (SKILL.md step 7); the table is generated *from* this field,
not kept in sync with it by hand.
