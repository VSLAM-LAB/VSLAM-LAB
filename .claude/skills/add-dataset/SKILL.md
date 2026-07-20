---
name: add-dataset
description: Add a new dataset to VSLAM-LAB. Use when the user asks to add/integrate a new benchmark dataset, wire up a dataset for download/evaluation, or asks "how do I add a dataset". Requires a dataset name and a source location.
---

Usage: `/add-dataset <name> <source>` — e.g. `/add-dataset soneva https://example.com/soneva-dataset`, or `/add-dataset soneva /mnt/data/soneva` for a local path. `<source>` maps to one of five download labels: `website` → URL, `hugging-face` → repo URL or `org/name`, `google-drive` → shared link, `local` → filesystem path, `other` → anything else (ask the user).

Both `<name>` and `<source>` are required — parse them from `$ARGUMENTS` (or from however the user phrased the request, if invoked without the slash command). If either is missing or ambiguous, stop and ask the user rather than guessing a dataset name or searching for a source yourself.

Adding a dataset means creating a `DatasetVSLAMLab` subclass plus a settings YAML, then registering it in `Datasets/get_dataset.py`.

0. **Refresh the dataset table first.** Run `python3 Datasets/extra-files/generate_dataset_table.py` to regenerate `Datasets/extra-files/dataset_table.md` from the current `Datasets/dataset_files/*.yaml`/`.py` — steps 1 and 2 below both read from it, so it must reflect the repo's current state before anything else.

@../../../Datasets/extra-files/dataset_table.md

1. **Gather the required fields, then report them before writing anything.**

   - `dataset_name` — lowercase slug reused everywhere: `dataset_<name>.py`/`.yaml`, the class prefix, the `get_dataset.py` switcher key.
   - `sequence_names` — the list of sequence IDs this dataset ships.
   - `cam_models` — closed list; every value must already appear in the **Camera Models** column of `Datasets/extra-files/dataset_table.md` (currently `pinhole`, `radtan4`, `radtan5`, `equid4`, `unknown` — read live, the list can grow).
   - `modes` — closed list; every value must already appear in the **Modes** column of `Datasets/extra-files/dataset_table.md` (currently `mono`, `mono-vi`, `rgbd`, `rgbd-vi`, `stereo`, `stereo-vi`). Include the native mode(s) *and* every mode derivable by dropping a channel:
     - `stereo`/`rgbd` → `mono` (one image of the pair / drop depth) — `stereo` and `rgbd` don't reduce to each other.
     - `-vi` → non-`-vi` (drop the IMU stream) — one-way only, never invent IMU data that isn't in the source.
     - e.g. native `stereo-vi` → `{mono, mono-vi, stereo, stereo-vi}`; `rgbd-vi` → `{mono, mono-vi, rgbd, rgbd-vi}`; plain `stereo` → `{mono, stereo}`; plain `mono` → `{mono}`. Each derived mode still needs its own implementation (e.g. a `mono` path that reads only the LHS image).
   - `groundtruth_available` — true/false; if false, `create_groundtruth_csv` has nothing to write.
   - `calibration_type` — `global` (same values for every sequence, e.g. `dataset_7scenes.py`'s fixed `CAMERA_PARAMS`) or `per-sequence` (locate and parse each sequence's own calibration file, e.g. `dataset_eth.py`/`dataset_kitti.py`/`dataset_euroc.py`) — determines whether `create_calibration_yaml` reuses one value set or parses per sequence.
   - `download` — one of `website`/`hugging-face`/`google-drive`/`local`, resolved from `<source>` per step (b) below.
   - `download_issues` — any known constraint blocking *automatic* download apparent from `<source>` (`complete_dataset`/`api_token`/`huggingface_token`/`license_required`, see step 4); leave blank if none.

   If the prompt, `<source>`, or the user names a mode or camera model outside the current `dataset_table.md` set, don't add it as new — flag it in the Notes column of the table below and ask the user how to proceed (map to the closest existing value, or treat as a genuinely new category worth discussing separately).

   Resolve each field in this order, don't skip ahead:
   a. **From the prompt** — whatever the user already gave in `$ARGUMENTS`/the request beyond just `<name>` and `<source>` (e.g. they may state modes or sequence names directly).
   b. **From `<source>`** — inspect it to fill in whatever the prompt didn't. Visiting/browsing `<source>` also tells you which download pattern (`website`/`hugging-face`/`google-drive`/`local`) this dataset fits — see the `__init__`/`download_sequence_data` comments in `Datasets/extra-files/dataset_template.py` for the exact YAML field and model file to follow per pattern. A dataset can mix patterns per sequence. Always pin down one of these four real patterns; `other` isn't one to implement against.
   c. **Ask the user** — for any field still unresolved after (a) and (b), ask directly. Keep asking until every field has a value; don't fill gaps with a guess or a plausible-looking default.

   **Before moving to step 2**, print a table of all eight fields with how each was resolved, then a notes line for anything worth flagging — most importantly any inconsistency between what the user said and what `<source>` actually shows (e.g. user said mono-only but the data also has depth frames; user gave 12 sequence names but the source lists 15). `download` and `download_issues` feed step 2's cross-check, so resolve them here even though they're not written to the YAML as single fields:

   | Field | Value | Source | Notes |
   |---|---|---|---|
   | dataset_name | ... | prompt / url / asked | |
   | sequence_names | ... | prompt / url / asked | |
   | cam_models | ... | prompt / url / asked | |
   | modes | ... | prompt / url / asked | |
   | groundtruth_available | ... | prompt / url / asked | |
   | calibration_type | ... | prompt / url / asked | |
   | download | ... | prompt / url / asked | |
   | download_issues | ... | prompt / url / asked | |

   If a note flags an unresolved inconsistency, stop and confirm with the user before proceeding — don't silently pick one side of a contradiction.

2. **Cross-check against similar existing datasets, and confirm the match with the user.** Using `Datasets/extra-files/dataset_table.md` (regenerated in step 0) — a table of every existing dataset's `Camera Models`, `Modes`, `Download` source, and `Download Issues` — search for datasets sharing at least one label with the step-1 table (same camera model, same mode, same download source, or same download issue) and show the user a comparison table of just those overlaps, e.g.:

   | Dataset | Shared Camera Model | Shared Modes | Shared Download | Shared Download Issues |
   |---|---|---|---|---|
   | kitti | pinhole | mono | website | complete_dataset |
   | ariel | | mono-vi stereo-vi | hugging-face | |

   **Ask the user if this list looks right** (anything obviously missing or wrongly matched?) before going further — don't silently proceed on a bad match.

   Once confirmed, read the `.py` and `.yaml` files of those matched datasets to learn the concrete implementation and stay consistent with it, rather than inventing a new shape. Which function(s) to study depends on *which* label matched, since each label maps to a different part of the implementation:
   - Shared **Camera Model** → study `create_calibration_yaml` (how intrinsics/distortion for that model are structured and written).
   - Shared **Modes** → study `create_rgb_folder`/`create_rgb_csv`/`create_groundtruth_csv` and the folder layout (`rgb_0`/`rgb_1`, `imu_0.csv`, etc.) for that mode.
   - Shared **Download** source → study `__init__` and `download_sequence_data`, and the corresponding YAML field (`repo_id`, `url_download_root`, `sequence_location`).
   - Shared **Download Issues** → study `get_download_issues` and the exact `_get_dataset_issue(issue_id=...)` call.

   A dataset can match on more than one label (e.g. same camera model *and* same download source) — in that case it's worth studying more closely than a single-label match.

3. **Write the dataset YAML.** Copy `Datasets/extra-files/dataset_template.yaml` to `Datasets/dataset_files/dataset_<name>.yaml` and populate it directly from the step-1 table — this produces the real, final YAML, not a placeholder:
   - `dataset_name`, `sequence_names` (the confirmed list).
   - `rgb_hz` — the RGB capture rate in Hz; required by the base class (`cfg["rgb_hz"]`, no default) even though it isn't one of the step-1 fields — get it from `<source>` (spec sheet, README, or metadata file) or ask.
   - `cam_models`, `modes` (both as YAML lists, e.g. `['pinhole']`, `['mono', 'stereo']`).
   - The YAML field for the `download` pattern resolved in step 1: `url_download_root` for `website` or `google-drive` (a `drive.google.com` URL for the latter, see `dataset_hilti2026.yaml`), `repo_id` for `hugging-face`, `sequence_location: local` per affected sequence for `local` (see `dataset_strayscanner.yaml`).
   - Any mode-specific fields a sibling YAML of the same modes/source-pattern carries (e.g. `depth_factor` for `rgbd`, `url_download_root_gt` when groundtruth ships as a separate archive like `dataset_kitti.yaml`) — check the closest model file from step 1 for what it reads.
   - An `about:` block (license, summary, homepage, authors) and a `vslamlab_maintainer:` block, following the shape already used in every existing dataset YAML (see `dataset_hilti2026.yaml` for the full shape).
   `calibration_type` and `download_issues` aren't YAML fields themselves — they inform how `create_calibration_yaml` and `get_download_issues` get implemented in step 4.

4. **Copy the Python template and implement the class**: start from `Datasets/extra-files/dataset_template.py`, save as `Datasets/dataset_files/dataset_<name>.py`, subclass `DatasetVSLAMLab` (`Datasets/DatasetVSLAMLab.py`), name it `<NAME>_dataset` — study the source-pattern model from step 1 (and a sibling of the same mode: monocular/RGBD/stereo/stereo-VI, see the section comments in `get_dataset.py`) rather than writing from scratch.

   @../../../Datasets/extra-files/dataset_template.py

   Each hook in the template carries an inline comment on what it's responsible for and which model file(s) to study for that download/calibration pattern — read those before writing real logic, and delete a hook entirely (rather than leaving a hollow stub) if it doesn't apply to this dataset: `create_imu_csv` for a non-`-vi` dataset, `create_groundtruth_csv` if `groundtruth_available` is false, `get_download_issues` if `download_issues` from step 1 is blank.

   **Scaffold before implementing anything for real.** Add every hook from the template as a placeholder (e.g. `pass`, or a `print("TODO: ...")` + `pass`) so the class is concrete and importable/instantiable immediately — this catches import errors, signature mismatches, and ABC/abstract-method gaps before any real logic exists to hide behind.

   Then implement each hook for real, in this order:
   a. **`__init__`**: call `super().__init__(...)`, load the YAML (`self.yaml_file`), pull out the source-specific field from step 3 (`self.url_download_root` / `self.repo_id`) and any mode-specific fields the YAML carries, and build `self.sequence_nicknames`. Follow a sibling of the same source pattern (e.g. `dataset_ariel.py` for Hugging Face) rather than inventing the shape.
   b. **`download_sequence_data(sequence_name)`** — fetch and decompress raw sequence data per the `download` pattern from step 1 (or, for `local` sequences, point the user at where to place it).
   c. `create_rgb_folder`.
   d. `create_rgb_csv`.
   e. `create_calibration_yaml` (and `create_imu_csv` if applicable).
   f. `create_groundtruth_csv`, if `groundtruth_available`.
   g. `remove_unused_files`.
   h. `get_download_issues`, if `download_issues` is non-blank — built via `_get_dataset_issue(issue_id=..., dataset_name=self.dataset_name, ...)` in `Datasets/DatasetVSLAMLab_issues.py`.

5. **Register it** in `Datasets/get_dataset.py`:
   - Add `from Datasets.dataset_files.dataset_<name> import <NAME>_dataset` under the correct mode section comment (Monocular / RGBD / Stereo / Stereo-VI / Development).
   - Add an entry to the `switcher` dict in `get_dataset()`: `"<name>": lambda: <NAME>_dataset(benchmark_path),`.

Full reference docs live on the project's GitHub Wiki if more detail is needed.
