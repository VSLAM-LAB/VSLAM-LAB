---
name: add-dataset
description: Add a new dataset to VSLAM-LAB. Use when the user asks to add/integrate a new benchmark dataset, wire up a dataset for download/evaluation, or asks "how do I add a dataset". Requires a dataset name and a source location.
---

Usage: `/add-dataset <name> <source>` — e.g. `/add-dataset soneva https://example.com/soneva-dataset`, or `/add-dataset soneva /mnt/data/soneva` for a local path. `<source>` maps to one of five download labels: `website` → URL, `hugging-face` → repo URL or `org/name`, `google-drive` → shared link, `local` → filesystem path, `other` → anything else (ask the user).

Both `<name>` and `<source>` are required — parse them from `$ARGUMENTS` (or from however the user phrased the request, if invoked without the slash command). If either is missing or ambiguous, stop and ask the user rather than guessing a dataset name or searching for a source yourself.

Adding a dataset means creating a `DatasetVSLAMLAB` subclass plus a settings YAML, registering it in `Datasets/get_dataset.py`, adding a smoke-test config/experiment pair under `configs/`, actually running that smoke test end to end, and committing the result. **The skill is not complete until step 9 (the commit) has run.** Steps 1–7 produce files that look correct by inspection; step 8 is what actually proves the dataset works, and it has been skipped in past runs because step 7 (the README row) feels like a natural stopping point — it is not. Do not report the dataset as done, and do not stop, until you have executed step 8's simulation and step 9's commit, in order.

**Hard constraint — file scope.** The only files this skill may create or modify are:

- `Datasets/dataset_files/dataset_<name>.py` — new dataset class, created in step 4.
- `Datasets/dataset_files/dataset_<name>.yaml` — new dataset settings file, created in step 3.
- `Datasets/get_dataset.py` — only step 5's two edits (the import line and the `switcher` dict entry), nothing else in the file.
- `configs/test_config_<name>.yaml` — new smoke-test sequence list, created in step 6.
- `configs/test_exp_<name>.yaml` — new smoke-test experiment config, created in step 6.
- `README.md` — only the new row appended to the Datasets table in step 7, nothing else in the file.
- `Datasets/extra-files/dataset_table.md` — regenerated (never hand-edited) via the script in step 0, since it's generated output.

Everything else — `Datasets/DatasetVSLAMLAB.py`, `Datasets/DatasetVSLAMLAB_calibration.py`, `Datasets/DatasetVSLAMLAB_issues.py`, any other dataset's `.py`/`.yaml`, any other `configs/*.yaml` (including a non-`test_`-prefixed `config_<name>.yaml`/`exp_<name>.yaml`, which is a production config outside this skill's scope), the templates in `Datasets/extra-files/`, any part of `README.md` outside the single new table row, etc. — is read-only reference material, even when it would be convenient to tweak (e.g. to add a shared helper, fix something noticed in passing, or relax a base-class check). If something outside this scope genuinely needs to change, stop and flag it to the user instead of editing it directly. This scope covers the repo only. Outside the repo, the one place this skill may touch is `VSLAM-LAB-Benchmark/<DATASET_FOLDER>/` (the downloaded benchmark data for this dataset, `<DATASET_FOLDER>` matching what this dataset's YAML/class designate) — freely creating, removing, or recreating files and folders anywhere inside it is expected and fine, that's exactly what steps 6/8's test runs are for. Nothing outside `VSLAM-LAB-Benchmark/<DATASET_FOLDER>/` may be touched — not other datasets' folders under `VSLAM-LAB-Benchmark/`, not `VSLAM-LAB-Evaluation/`, nothing else.

The one exception to "create or modify" above is git itself: step 9 stages and commits the files in this list (nothing else) as a single local commit. That's the only git write this skill performs — never `git push`, never amend or rewrite an existing commit, never touch branches.

0. **Refresh the dataset table first.** Run `python3 Datasets/extra-files/generate_dataset_table.py` to regenerate `Datasets/extra-files/dataset_table.md` from the current `Datasets/dataset_files/*.yaml`/`.py` — steps 1 and 2 below both read from it, so it must reflect the repo's current state before anything else.

@../../../Datasets/extra-files/dataset_table.md

1. **Gather the required fields, then report them before writing anything.**

   - `dataset_name` — lowercase slug reused everywhere: `dataset_<name>.py`/`.yaml`, the class prefix, the `get_dataset.py` switcher key.
   - `sequence_names` — the list of sequence IDs this dataset ships. Keep each name descriptive but simple — drop redundant prefixes shared by every sequence (e.g. a repeated dataset/location prefix). For `soneva`, `hb_20250710` would have been simpler than `maldives_soneva_hb_20250710`.
   - `cam_models` — closed list; every value must already appear in the **Camera Models** column of `Datasets/extra-files/dataset_table.md` (currently `pinhole`, `radtan4`, `radtan5`, `equid4`, `unknown` — read live, the list can grow).
   - `modes` — closed list; every value must already appear in the **Modes** column of `Datasets/extra-files/dataset_table.md` (currently `mono`, `mono-vi`, `rgbd`, `rgbd-vi`, `stereo`, `stereo-vi`). Include the native mode(s) *and* every mode derivable by dropping a channel:
     - `stereo`/`rgbd` → `mono` (one image of the pair / drop depth) — `stereo` and `rgbd` don't reduce to each other.
     - `-vi` → non-`-vi` (drop the IMU stream) — one-way only, never invent IMU data that isn't in the source.
     - e.g. native `stereo-vi` → `{mono, mono-vi, stereo, stereo-vi}`; `rgbd-vi` → `{mono, mono-vi, rgbd, rgbd-vi}`; plain `stereo` → `{mono, stereo}`; plain `mono` → `{mono}`. Each derived mode still needs its own implementation (e.g. a `mono` path that reads only the LHS image).
   - `resize` — true/false; true if this dataset's source images are bigger than 640×480 (by pixel area) and need downscaling, false if they're already at or below that. This only decides whether to *populate* the YAML's `target_resolution: [640, 480]` field by default (see `dataset_sweetcorals.py`/`.yaml`) — the implementation itself is always the same, uniform code (see step 4c): `create_rgb_folder` checks `self.target_resolution` at runtime (`None` if the YAML field is absent) and downscales to match its pixel area while preserving aspect ratio when set, or copies the raw image unresized when not. This means a user (now or later) can delete `target_resolution` from the YAML to fall back to original-resolution images without touching any code — build the class to support that from the start, don't hardcode a resize-or-don't branch based on the step-1 answer.
   - `groundtruth_available` — true/false; if false, `create_groundtruth_csv` has nothing to write.
   - `calibration_type` — `global` (same values for every sequence, e.g. `dataset_7scenes.py`'s fixed `CAMERA_PARAMS`) or `per-sequence` (locate and parse each sequence's own calibration file, e.g. `dataset_eth.py`/`dataset_kitti.py`/`dataset_euroc.py`) — determines whether `create_calibration_yaml` reuses one value set or parses per sequence.
   - `download` — one of `website`/`hugging-face`/`google-drive`/`local`, resolved from `<source>` per step (b) below.
   - `download_issues` — any known constraint blocking *automatic* download apparent from `<source>` (`complete_dataset`/`api_token`/`huggingface_token`/`license_required`, see step 4); leave blank if none.

   If the prompt, `<source>`, or the user names a mode or camera model outside the current `dataset_table.md` set, don't add it as new — flag it in the Notes column of the table below and ask the user how to proceed (map to the closest existing value, or treat as a genuinely new category worth discussing separately).

   Resolve each field in this order, don't skip ahead:
   a. **From the prompt** — whatever the user already gave in `$ARGUMENTS`/the request beyond just `<name>` and `<source>` (e.g. they may state modes or sequence names directly).
   b. **From `<source>`** — inspect it to fill in whatever the prompt didn't. Visiting/browsing `<source>` also tells you which download pattern (`website`/`hugging-face`/`google-drive`/`local`) this dataset fits — see the `__init__`/`download_sequence_data` comments in `Datasets/extra-files/dataset_template.py` for the exact YAML field and model file to follow per pattern. A dataset can mix patterns per sequence. Always pin down one of these four real patterns; `other` isn't one to implement against.
   c. **Ask the user** — for any field still unresolved after (a) and (b), ask directly. Keep asking until every field has a value; don't fill gaps with a guess or a plausible-looking default.

   **Before moving to step 2**, print a table of all nine fields with how each was resolved, then a notes line for anything worth flagging — most importantly any inconsistency between what the user said and what `<source>` actually shows (e.g. user said mono-only but the data also has depth frames; user gave 12 sequence names but the source lists 15). `download` and `download_issues` feed step 2's cross-check, so resolve them here even though they're not written to the YAML as single fields:

   | Field | Value | Source | Notes |
   |---|---|---|---|
   | dataset_name | ... | prompt / url / asked | |
   | sequence_names | ... | prompt / url / asked | |
   | cam_models | ... | prompt / url / asked | |
   | modes | ... | prompt / url / asked | |
   | resize | ... | prompt / url / asked | |
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
   - Shared **Download** source → study `__init__` and `download_sequence_data`, and the corresponding YAML field (`hf_repo_id`, `url_download_root`, `sequence_location`).
   - Shared **Download Issues** → study `get_download_issues` and the exact `_get_dataset_issue(issue_id=...)` call.

   A dataset can match on more than one label (e.g. same camera model *and* same download source) — in that case it's worth studying more closely than a single-label match.

   **Important: when several matched datasets are candidates to study for the same label, trust them in this order** (check each candidate's `vslamlab_maintainer`/`about.authors` in its `.yaml`):
   1. Datasets whose `vslamlab_maintainer.name` is Alejandro Fontan.
   2. Datasets whose `vslamlab_maintainer.name` also appears in that same dataset's `about.authors` (the VSLAM-LAB integrator was one of the dataset's original creators, so the integration reflects first-hand knowledge of the source).
   3. Any other dataset.
   Prefer the highest-priority candidate available; only fall back to a lower tier if nothing higher matches the label in question.

3. **Write the dataset YAML.** Copy `Datasets/extra-files/dataset_template.yaml` to `Datasets/dataset_files/dataset_<name>.yaml` and populate it directly from the step-1 table — this produces the real, final YAML, not a placeholder:
   - `dataset_name`, `sequence_names` (the confirmed list).
   - `rgb_hz` — the RGB capture rate in Hz; required by the base class (`cfg["rgb_hz"]`, no default) even though it isn't one of the step-1 fields — get it from `<source>` (spec sheet, README, or metadata file) or ask.
   - `cam_models`, `modes` (both as YAML lists, e.g. `['pinhole']`, `['mono', 'stereo']`).
   - The YAML field for the `download` pattern resolved in step 1: `url_download_root` for `website` or `google-drive` (a `drive.google.com` URL for the latter, see `dataset_hilti2026.yaml`), `hf_repo_id` for `hugging-face`, `sequence_location: local` per affected sequence for `local` (see `dataset_strayscanner.yaml`).
   - If `resize` (step 1) is true, add `target_resolution: [640, 480]` — see `dataset_sweetcorals.yaml`. Omit it entirely if `resize` is false. Either way this is just the *initial* value: the field is meant to be safely removable later (falls back to original-resolution images, no code change needed) — see step 4c.
   - Any mode-specific fields a sibling YAML of the same modes/source-pattern carries (e.g. `depth_factor` for `rgbd`, `url_download_root_gt` when groundtruth ships as a separate archive like `dataset_kitti.yaml`) — check the closest model file from step 1 for what it reads.
   - An `about:` block (license, summary, homepage, authors) and a `vslamlab_maintainer:` block, following the shape already used in every existing dataset YAML (see `dataset_hilti2026.yaml` for the full shape).
   `calibration_type` and `download_issues` aren't YAML fields themselves — they inform how `create_calibration_yaml` and `get_download_issues` get implemented in step 4.

4. **Copy the Python template and implement the class**: start from `Datasets/extra-files/dataset_template.py`, save as `Datasets/dataset_files/dataset_<name>.py`, subclass `DatasetVSLAMLAB` (`Datasets/DatasetVSLAMLAB.py`), name it `<Name>Dataset` in PEP 8 CapWords (e.g. `soneva` -> `SonevaDataset`, `eiffel_tower` -> `EiffelTowerDataset`, `hilti2022` -> `Hilti2022Dataset` — capitalize each underscore-separated token of `dataset_name`, no acronym exceptions) — study the source-pattern model from step 1 (and a sibling of the same mode: monocular/RGBD/stereo/stereo-VI, see the section comments in `get_dataset.py`) rather than writing from scratch.

   @../../../Datasets/extra-files/dataset_template.py

   Fill in the module header at the top of the file (`Author`/`Assisted by`/`Version: 1.0`/`Created`/`License`) from the same values as the YAML's `vslamlab_maintainer:` block written in step 3 — `Author` = `vslamlab_maintainer.name`, `Assisted by` = `vslamlab_maintainer.assisted_by` (write `None` if absent), `Created` = `vslamlab_maintainer.date`. Every dataset file's header and its YAML's maintainer block should always agree.

   Give the class a one-line docstring, `"""<Display Name> dataset helper for VSLAM-LAB benchmark."""`, where `<Display Name>` is the dataset's proper/brand name in natural Title Case (from the YAML's `about.summary`/`about.homepage`) — never the internal `<Name>Dataset` class spelling or the lowercase `dataset_name` slug. A few extra descriptive words are fine if they add real disambiguating context (see `dataset_eiffel_tower.py`: `"""Eiffel Tower deep-sea underwater dataset helper for VSLAM-LAB benchmark."""`).

   Keep the template's import grouping as you fill the file in: `from __future__ import annotations` first, then up to three blank-line-separated groups (stdlib, third-party, project-local), each with plain `import x` lines before `from x import y` lines, alphabetical case-insensitive within each — see `dataset_soneva.py` for a fully-worked example. No linter enforces this yet (see CLAUDE.md), so this is manual discipline, not something a tool will catch.

   Each hook in the template carries an inline comment on what it's responsible for and which model file(s) to study for that download/calibration pattern — read those before writing real logic, and delete a hook entirely (rather than leaving a hollow stub) if it doesn't apply to this dataset: `create_imu_csv` for a non-`-vi` dataset, `create_groundtruth_csv` if `groundtruth_available` is false, `get_download_issues` if `download_issues` from step 1 is blank.

   **Scaffold before implementing anything for real.** Add every hook from the template as a placeholder (e.g. `pass`, or a `print("TODO: ...")` + `pass`) so the class is concrete and importable/instantiable immediately — this catches import errors, signature mismatches, and ABC/abstract-method gaps before any real logic exists to hide behind.

   **Important: prefer `utilities.py` functions over hand-rolled logic, but never add to `utilities.py` itself.** Before writing any hook's real logic, check whether `utilities.py` already has a function for it (path helpers, CSV read/write, `downloadFile`/`decompressFile`, the Hugging Face helpers, the COLMAP helpers, `make_printers`, etc. — read the block-summary comment near the top of the file) and use it rather than reimplementing the same thing inline. `utilities.py` is not in this skill's file-scope (see the hard constraint above) — if something genuinely reusable is missing from it, do not add it yourself; tell the user what's missing and suggest it as a follow-up, the same way this skill flags any other out-of-scope need.

   Then implement each hook for real, in this order:
   a. **`__init__`**: call `super().__init__(...)`, load the YAML (`self.yaml_file`), pull out the source-specific field from step 3 (`self.url_download_root` / `self.hf_repo_id`) and any mode-specific fields the YAML carries, and build `self.sequence_nicknames`. Follow a sibling of the same source pattern (e.g. `dataset_ariel.py` for Hugging Face) rather than inventing the shape. Read `target_resolution` with `cfg.get("target_resolution")` (never `cfg["target_resolution"]`) and fall back to `None` — `self.target_resolution: tuple[int, int] | None = tuple(cfg["target_resolution"]) if cfg.get("target_resolution") else None` — see `dataset_soneva.py`.
   b. **`download_sequence_data(sequence_name)`** — fetch and decompress raw sequence data per the `download` pattern from step 1 (or, for `local` sequences, point the user at where to place it).
   c. `create_rgb_folder` — branch per image on `self.target_resolution`, not a separate resize flag: if `None`, copy/link the raw image unresized (e.g. `shutil.copy2` — never round-trip an unresized image through PIL, which would needlessly re-encode/strip metadata); otherwise scale it down to match `self.target_resolution`'s pixel area while preserving aspect ratio via `utilities.compute_scaled_size(img.size, self.target_resolution)`. See `dataset_soneva.py`/`dataset_sweetcorals.py` (`HFColmapDatasetMixin.create_rgb_folder`) or `dataset_eiffel_tower.py`.
   d. `create_rgb_csv`.
   e. `create_calibration_yaml` (and `create_imu_csv` if applicable).
   f. `create_groundtruth_csv`, if `groundtruth_available`.
   g. `remove_unused_files`.
   h. `get_download_issues`, if `download_issues` is non-blank — built via `_get_dataset_issue(issue_id=..., dataset_name=self.dataset_name, ...)` in `Datasets/DatasetVSLAMLAB_issues.py`.

5. **Register it** in `Datasets/get_dataset.py`:
   - Add `from Datasets.dataset_files.dataset_<name> import <Name>Dataset` under the correct mode section comment (Monocular / RGBD / Stereo / Stereo-VI / Development).
   - Add an entry to the `switcher` dict in `get_dataset()`: `"<name>": lambda: <Name>Dataset(),`.

6. **Create a smoke-test config + experiment pair** under `configs/`, following the shape of the closest sibling from step 2 (e.g. `test_config_sweetcorals.yaml`/`test_exp_sweetcorals.yaml`, `test_config_eth.yaml`/`test_exp_eth.yaml`, `test_config_videos.yaml`/`test_exp_videos.yaml`):
   - `configs/test_config_<name>.yaml` — a single `<dataset_name>:` key with a YAML list of sequence names to smoke-test:
     ```yaml
     <dataset_name>:
     - sequence_01
     - sequence_02
     ```
     For a small dataset, list every sequence (see `test_config_sweetcorals.yaml`, 13/13). For a large one, a small representative handful is enough (see `test_config_videos.yaml`, 5 sequences; `test_config_strayscanner.yaml`, 1) — don't enumerate hundreds of sequences just to be thorough; pick ones that exercise different sizes/conditions if the dataset is heterogeneous.
   - `configs/test_exp_<name>.yaml` — one `exp_<name>_<baseline>:` block (or a few, one per baseline worth smoke-testing):
     ```yaml
     exp_<name>_<baseline>:
       Config: test_config_<name>.yaml
       NumRuns: 1
       Parameters: {verbose: 1, mode: <one of this dataset's modes>, rgb_idx: [0,2000]}
       Module: <baseline>
     ```
     Pick `<baseline>` (the `Module`) matching what the closest sibling's `test_exp_*.yaml` uses where possible (`droidslam` and `dpvo` are common lightweight choices for `mono`/`rgbd`) — it must be a pixi environment name from `pixi.toml`'s `[environments]` table. `rgb_idx: [0,2000]` caps the run to the first ~2000 frames for a quick test; omit it if the matched sibling's convention doesn't use it (see `test_exp_videos.yaml`/`test_exp_strayscanner.yaml`).

7. **Add the dataset to the README table.** In `README.md`, under the "VSLAM-LAB Supported Baselines and Datasets" section, add one new row to the **Datasets** table (the table headed `| Datasets | Features | Label | Modes | Camera Models |`) — append it as the last real row, immediately above the commented-out placeholder rows at the bottom of that table (the `<!-- | [**Sweet Corals**]... -->`-style rows for datasets not yet implemented). Match the exact shape of existing rows:
   ```
   | [**<Display Name>**](<homepage URL>) | <feature emoji(s)> | `<dataset_name>` | <modes> | <cam_models> |
   ```
   - `<Display Name>` and `<homepage URL>` — from the `about:` block written into the YAML in step 3.
   - `<feature emoji(s)>` — pick from the legend printed just below the table (Real 📸 / Synthetic 💻; Indoor 🏠 / Outdoor 🏞️ / Underwater 🌊 / Intracorporeal 🫀; Handheld 🤳 / Headmounted 🥽 / Vehicle 🚗 / UAV 🚁 / Robot 🤖), based on what's known about the dataset from step 1/2. If it's unclear which apply, ask the user rather than guessing.
   - `<modes>` — the confirmed step-1 `modes` list, formatted like existing rows: a mode and its `-vi` variant collapse to one entry with the suffix in parentheses (e.g. `mono` + `mono-vi` → `` `mono(-vi)` ``), each mode backticked and space-separated.
   - `<cam_models>` — the confirmed step-1 `cam_models` list, backticked and space-separated, in the same style as existing rows.

8. **(Required — do not skip.) Simulate `pixi run download-sequence <dataset_name> <sequence_name>` function-by-function**, using the **first sequence** in `sequence_names`. This step is mandatory even when steps 1–7 already look correct on inspection — code that imports cleanly and reads right can still fail the first time it actually runs (wrong URL, a path typo, a malformed calibration field), and that's only caught by running it. `download-sequence` (`vslamlab_gui.py download_sequence` → `dataset.download_sequence(sequence_name)`) ultimately just calls `download_process`, which runs these hooks in order:

   1. `download_sequence_data`
   2. `create_rgb_folder`
   3. `create_rgb_csv`
   4. `create_imu_csv`
   5. `create_calibration_yaml`
   6. `create_groundtruth_csv`
   7. `remove_unused_files`

   (Drop rows 4/6 if `create_imu_csv`/`create_groundtruth_csv` were deleted in step 4 because they don't apply to this dataset.) Instead of invoking the CLI as one opaque call, drive the same sequence yourself one hook at a time (e.g. `dataset = get_dataset(dataset_name)` then call each method directly in the order above) so every stage's inputs and outputs can be inspected before moving to the next.

   Keep the list above visible as a running checklist, updating each row's state as you go (`to be run` → `running` → `processed`), e.g.:

   | # | Function | State |
   |---|---|---|
   | 1 | `download_sequence_data` | processed |
   | 2 | `create_rgb_folder` | running |
   | 3 | `create_rgb_csv` | to be run |
   | 4 | `create_calibration_yaml` | to be run |
   | 5 | `create_groundtruth_csv` | to be run |
   | 6 | `remove_unused_files` | to be run |

   After each function returns, report in detail — not just that it ran:
   - **Inputs**: `sequence_name`, plus whatever state/files it consumed (`self.hf_repo_id`/`self.url_download_root`, the folder(s) the previous hook produced).
   - **Output generated**: exact paths created, file/image counts, folder sizes; for `create_rgb_csv`/`create_groundtruth_csv` the row count and first couple of rows; for `create_calibration_yaml` the actual `focal_length`/`principal_point`/`image_dimension` values written; for `remove_unused_files` which paths it actually deleted.

   This is slower than letting the whole pipeline run silently, but it pinpoints exactly which stage produced bad output (wrong image count, zeroed calibration, empty groundtruth) instead of only learning after the fact that the sequence failed (or silently passed with wrong data).

   Only move on to step 9 once every row in the checklist reads `processed`. If any hook fails, fix the underlying code in `dataset_<name>.py`/`.yaml` and re-run from that hook (or from the top, if the fix touches `download_sequence_data`/`__init__`) — don't proceed to commit a dataset that hasn't been proven to run.

9. **Commit the new dataset.** Stage exactly the files this skill created or modified, by name — never `git add -A`/`git add .`, which could sweep in unrelated in-progress work elsewhere in the repo:
   - `Datasets/dataset_files/dataset_<name>.py`
   - `Datasets/dataset_files/dataset_<name>.yaml`
   - `Datasets/get_dataset.py`
   - `configs/test_config_<name>.yaml`
   - `configs/test_exp_<name>.yaml`
   - `README.md`
   - `Datasets/extra-files/dataset_table.md` (regenerated in step 0 — include it so the repo isn't left with a stray uncommitted diff)

   Run `git status` first and confirm the staged set matches this list exactly (nothing more, nothing less) before committing. Commit with a concise message such as `Add <name> dataset` (following the repo's existing commit-message style — check `git log --oneline -10` if unsure). This step only creates a local commit: never push, force-push, or amend an existing commit as part of this skill.

Full reference docs live on the project's GitHub Wiki if more detail is needed.
