---
name: add-dataset
description: Add a new dataset to VSLAM-LAB — a benchmark dataset (local or remote), a data-capture tool, a set of videos, a set of images, or an API-backed collection. Use when the user asks to add/integrate a new dataset or tool, integrate their own data (e.g. "integrate my data", "I want to benchmark on my own images/video"), wire up a dataset for download/evaluation, or asks "how do I add a dataset". Requires a dataset name and a source location to start; the rest of the dataset's fields are gathered interactively (step 1).
---

## Usage

`/add-dataset <name> <source>` — e.g. `/add-dataset soneva https://example.com/soneva-dataset`, or `/add-dataset soneva /mnt/data/soneva` for a local path.

`<source>` maps to one of these download labels: `website` · `hugging-face` · `google-drive` · `local` · `api` · `other` (anything else — ask the user). Step 1's "From `<source>`" resolution rule (item 2 below) classifies `<source>` against these for real — not restated here.

Both `<name>` and `<source>` are required. Parse them from `$ARGUMENTS` (or from the request, if invoked without the slash command). If either is missing or ambiguous, stop and ask — don't guess a name or search for a source yourself.

## What "done" means

Adding a dataset = a `DatasetVSLAMLAB` subclass + settings YAML, registered in `Datasets/get_dataset.py`, plus a smoke-test config/experiment pair that has actually been **run end to end**, then committed.

**Not done until step 9 (commit) has run.** Steps 1–7 produce files that look correct by inspection; step 8 is what proves the dataset works. In past runs step 8/9 got skipped because step 7 (the README row) feels like a natural stopping point — it isn't.

## File scope (hard constraint)

The only files this skill may create or modify:

| File | Written in | Notes |
|---|---|---|
| `Datasets/dataset_files/dataset_<name>.py` | Step 4 | new dataset class |
| `Datasets/dataset_files/dataset_<name>.yaml` | Step 3 | new dataset settings |
| `Datasets/get_dataset.py` | Step 5 | only the import line + `switcher` dict entry — nothing else in the file |
| `configs/test_config_<name>.yaml` | Step 6 | smoke-test sequence list |
| `configs/test_exp_<name>.yaml` | Step 6 | smoke-test experiment config |
| `README.md` | Step 7 | only the new Datasets- or Tools-table row — nothing else in the file |
| `Datasets/extra-files/dataset_table.md` | Step 0 | regenerated only, never hand-edited |

- Everything else in the repo — base classes, other datasets' files, other `configs/*.yaml`, the templates themselves, the rest of `README.md` — is **read-only reference material**, even when editing it would be convenient. If working through this skill turns up a bug, inconsistency, or improvement outside this scope, don't edit it — find a way to finish the dataset without that change, and file it instead (see the issue exception below).
- Outside the repo, the only path this skill may touch is `VSLAM-LAB-Benchmark/<DATASET_FOLDER>/` (this dataset's downloaded benchmark data) — freely create/remove/recreate anything inside it, that's what steps 6/8's test runs are for. Nothing else outside the repo.
- **Git exception**: step 9 stages and commits exactly the files listed above, as one local commit. Never `push`, amend, rewrite, or touch branches.
- **Issue exception**: an out-of-scope finding gets filed as a GitHub issue, not silently worked around and not left for the user to notice on their own — `gh issue create --label <label> ...` (see CLAUDE.md's Issue Labels for which one fits). Include what was found, where, and how this run worked around it. This is the only other out-of-scope write this skill may perform, alongside step 9's commit — the file itself still never gets edited.

## Workflow

### Step 0 — Refresh the dataset table

Run `python3 Datasets/extra-files/generate_dataset_table.py` to regenerate `Datasets/extra-files/dataset_table.md` from the current `Datasets/dataset_files/*.yaml`/`.py`. Steps 1 and 2 both read from it, so it must reflect current repo state first.

@../../../Datasets/extra-files/dataset_table.md

### Step 1 — Gather the required fields, then report them

Resolve each field in this order — don't skip ahead:
1. **From the prompt** — whatever the user already gave beyond `<name>`/`<source>` (modes, sequence names, etc.).
2. **From `<source>`** — inspect it to fill whatever the prompt didn't; this also reveals which of the five download patterns fits (see `download_sequence_data`'s comment in `Datasets/extra-files/dataset_template.py` for the per-pattern breakdown). A dataset can mix patterns per sequence. Always pin down a real pattern — `other` isn't one to implement against.
3. **Ask the user** — for anything still unresolved. Don't fill gaps with a guess or plausible default.

| Field | Meaning |
|---|---|
| `dataset_name` | lowercase slug, reused everywhere: file names, class prefix, switcher key |
| `sequence_names` | sequence IDs shipped; drop redundant shared prefixes (e.g. `hb_20250710`, not `maldives_soneva_hb_20250710`) |
| `cam_models` | closed list — see below |
| `modes` | closed list — see below |
| `resize` | true if source images are bigger than 640×480 by pixel area, else false — see below |
| `groundtruth_available` | true/false; false → `create_groundtruth_csv` writes header only |
| `calibration_type` | `global` (same values every sequence) or `per-sequence` (parsed per sequence) |
| `download` | one of `website`/`hugging-face`/`google-drive`/`local`/`api` |
| `download_issues` | known constraint blocking *automatic* download — `complete_dataset`/`api_token`/`huggingface_token`/`license_required`, or blank |

**`cam_models`** — closed list, must already appear in `dataset_table.md`'s Camera Models column (read live: currently `pinhole`, `radtan4`, `radtan5`, `equid4`, `unknown`). Each value must describe what `create_calibration_yaml` actually writes, not just "this is a perspective camera" — the exact mapping, per-value `Model:` citations, and detailed gotchas are defined once in `dataset_template.py`'s `create_calibration_yaml` comment (live-included in step 4 — read it now rather than waiting, the same way step 1's "From `<source>`" resolution rule above sends you into the template early for the download pattern). Not restated here.

**`modes`** — closed list, must already appear in `dataset_table.md`'s Modes column (read live: currently `mono`, `mono-vi`, `rgbd`, `rgbd-vi`, `stereo`, `stereo-vi`). Include the native mode(s) *and* every mode derivable by dropping a channel (`stereo`/`rgbd` → `mono`, `-vi` → non-`-vi`) — the exact derivation rule and examples are defined once in `dataset_template.yaml`'s `modes` comment (live-included in step 3). Not restated here.

**`modes` applies dataset-wide, not per-sequence** — `DatasetVSLAMLAB.check_sequence_integrity()` requires *every* sequence to satisfy *every* listed mode's requirements (an `rgb_1` folder for every sequence if `stereo`/`stereo-vi` is listed, an `imu_0.csv` for every sequence if `mono-vi`/`stereo-vi`/`rgbd-vi` is listed), regardless of whether that particular sequence's own source data actually supports it. If the source's sequences have genuinely different capabilities (e.g. some are stereo-capable, others mono-only), don't force them into one `dataset_name` with the richest `modes` list — split into separate `dataset_name`s by capability instead, each with its own `modes` matching only what its own sequences support, and repeat steps 3–9 for each. Model: `dataset_rover.py`'s `rover-t265`/`rover-d435i`/`rover-picam` split (by rig/sensor type); `dataset_pamir.py`/`dataset_pamir_rig.py`'s `pamir`/`pamir-rig` split (2024 two-camera rig dive vs. 2025 single-camera dives) is a second real-world example — the split there was only caught in step 8, after a stereo-mode sequence had already been merged into a mono-only dataset and its sibling sequences started failing `check_sequence_availability`. Catch this here, in step 1, instead.

**`resize`** — true if source images are bigger than 640×480 by pixel area, else false. Only decides the YAML's initial `target_resolution` value; the field is safely removable later with no code change (`create_rgb_folder`'s uniform runtime branch on `self.target_resolution` — step 4c below). Full detail: `dataset_template.yaml`'s `target_resolution` comment.

If the prompt/source/user names a mode or camera model outside the current closed lists, don't add it as new — flag it in step 2's Notes and ask the user how to proceed.

**Before moving to step 2**, print this table plus a Notes line for anything worth flagging (most importantly any inconsistency between what the user said and what `<source>` shows — e.g. user said mono-only but the source also has depth frames):

| Field | Value | Source | Notes |
|---|---|---|---|
| dataset_name | | prompt / url / asked | |
| sequence_names | | prompt / url / asked | |
| cam_models | | prompt / url / asked | |
| modes | | prompt / url / asked | |
| resize | | prompt / url / asked | |
| groundtruth_available | | prompt / url / asked | |
| calibration_type | | prompt / url / asked | |
| download | | prompt / url / asked | |
| download_issues | | prompt / url / asked | |

If a note flags an unresolved inconsistency, stop and confirm with the user before proceeding.

### Step 2 — Cross-check against similar datasets, confirm with the user

Using `Datasets/extra-files/dataset_table.md`, find datasets sharing at least one label with the step-1 table (camera model, mode, download source, or download issue) and show the user a comparison table of just the overlaps:

| Dataset | Shared Camera Model | Shared Modes | Shared Download | Shared Download Issues |
|---|---|---|---|---|
| kitti | pinhole | mono | website | complete_dataset |
| ariel | | mono-vi stereo-vi | hugging-face | |

**Ask the user if this list looks right** before going further.

Once confirmed, read the matched datasets' `.py`/`.yaml` — which function(s) to study depends on which label matched:

| Shared label | Study |
|---|---|
| Camera Model | `create_calibration_yaml` (intrinsics/distortion structure) |
| Modes | `create_rgb_folder`/`create_rgb_csv`/`create_groundtruth_csv` + folder layout |
| Download | `__init__` + `download_sequence_data` + the YAML field (`hf_repo_id`, `url_download_root`, ...) |
| Download Issues | `get_download_issues` + the `_get_dataset_issue(issue_id=...)` call |

A dataset matching on more than one label is worth studying more closely.

**When several matches are candidates for the same label, trust them in this order** (check `vslamlab_maintainer`/`about.authors` in each `.yaml`):
1. `vslamlab_maintainer.name` is Alejandro Fontan.
2. `vslamlab_maintainer.name` also appears in that dataset's own `about.authors` (integrator was an original creator).
3. Any other dataset.

Prefer the highest-priority candidate available.

### Step 3 — Write the dataset YAML

Copy `Datasets/extra-files/dataset_template.yaml` → `Datasets/dataset_files/dataset_<name>.yaml` and populate it directly from the step-1 table — this is the real, final YAML, not a placeholder. The template's own inline comments are the canonical reference for each field's exact shape and gotchas (the index below just points into them — don't re-derive or re-explain a pattern that's already documented there):

@../../../Datasets/extra-files/dataset_template.yaml

- `dataset_name`, `sequence_names` (confirmed list).
- `rgb_hz` — RGB capture rate in Hz; required by the base class (`cfg["rgb_hz"]`, no default), not a step-1 field — get it from `<source>` or ask.
- `cam_models`, `modes` — as YAML lists, e.g. `['pinhole']`, `['mono', 'stereo']`. The template above lists the closed-list values; the full mapping/gotchas for `cam_models` are canonical in step 4's template (`create_calibration_yaml`), for `modes` canonical right here in this template.
- The download field for the step-1 `download` pattern — the template above lists the exact field name/shape; the fetch mechanism and gotchas are canonical in step 4's template (`download_sequence_data`).

- If `resize` is true, add `target_resolution: [640, 480]`; omit entirely if false — see step 1's `resize` field above for why this is just an initial value, safely changed later with no code change.
- Any mode-specific fields a sibling YAML of the same modes/source carries (e.g. `depth_factor` for `rgbd`, `url_download_root_gt` for a separate groundtruth archive like `dataset_kitti.yaml`).
- An `about:` block (license, summary, homepage, authors) and `vslamlab_maintainer:` block, matching the shape in `dataset_eth.yaml`.

`calibration_type` and `download_issues` aren't YAML fields — they inform `create_calibration_yaml`/`get_download_issues` in step 4.

### Step 4 — Implement the Python class

Copy `Datasets/extra-files/dataset_template.py` → `Datasets/dataset_files/dataset_<name>.py`, subclass `DatasetVSLAMLAB`, name it `<Name>Dataset` in PEP 8 CapWords — capitalize each underscore-separated token of `dataset_name`, no acronym exceptions (`soneva` → `SonevaDataset`, `eiffel_tower` → `EiffelTowerDataset`, `hilti2022` → `Hilti2022Dataset`). Study the source-pattern model from step 1 and a same-mode sibling rather than writing from scratch.

@../../../Datasets/extra-files/dataset_template.py

**Before any real logic:**
- Module header (`Author`/`Assisted by`/`Version: 1.0`/`Created`/`License`) must match the YAML's `vslamlab_maintainer:` block exactly (`Author` = `.name`, `Assisted by` = `.assisted_by` or `None`, `Created` = `.date`).
- Class docstring: `"""<Display Name> dataset helper for VSLAM-LAB benchmark."""` — `<Display Name>` is the proper/brand name from `about.summary`/`about.homepage` (natural Title Case), never the class name or the lowercase slug. A few disambiguating words are fine (`"""MADMAX Mars rover navigation dataset helper..."""`).
- Import grouping: `from __future__ import annotations` first, then up to three blank-line-separated groups (stdlib, third-party, project-local), plain `import x` before `from x import y` within each, alphabetical case-insensitive. No linter enforces this — see `dataset_soneva.py`. Manual discipline only.
- **Scaffold first**: add every template hook as a placeholder (`pass` or `print("TODO")` + `pass`) so the class is importable/instantiable immediately, before any real logic exists to hide import/signature/ABC errors behind.
- **Prefer `utilities.py` over hand-rolled logic** (path helpers, CSV read/write, `downloadFile`/`decompressFile`, HF/COLMAP helpers, `make_printers`, ...) — but `utilities.py` is out of scope for this skill. If something reusable is genuinely missing, tell the user and suggest it as a follow-up; don't add it yourself.
- Delete a hook entirely (not a hollow stub) if it doesn't apply: `create_imu_csv` for non-`-vi`, `get_download_issues` if `download_issues` is blank.

**Then implement each hook for real, in order:**

| # | Hook | Responsibility | Model |
|---|---|---|---|
| a | `__init__` | call `super().__init__(...)`, pull the source field (`self.url_download_root`/`self.hf_repo_id`) and mode-specific fields from `self.cfg` | — |
| b | `download_sequence_data` | fetch + decompress per pattern; skip if already done | `dataset_squidle.py` (api), `dataset_rover.py` (website), `dataset_msd.py` (hf) |
| c | `create_rgb_folder` | normalize into `rgb_0`/`rgb_1`/`depth_0` via `self.rgb_path()`/`self.depth_path()` | `dataset_soneva.py`, `dataset_sweetcorals.py`, `dataset_eth.py` |
| d | `create_rgb_csv` | write `rgb.csv`, ns timestamps | `dataset_rgbdtum.py` (async rgbd) |
| e | `create_calibration_yaml` | write via `self.write_calibration_yaml(...)` | `dataset_7scenes.py` (global), `dataset_eth.py`/`kitti`/`euroc` (per-sequence) |
| f | `create_imu_csv` | only for `-vi` modes | — |
| g | `create_groundtruth_csv` | write `groundtruth.csv` | `dataset_rgbdtum.py` |
| h | `remove_unused_files` | delete per `BENCHMARK_RETENTION` tier | `dataset_eth.py`, `HFColmapDatasetMixin` |
| i | `get_download_issues` | only if `download_issues` non-blank | `Datasets/DatasetVSLAMLAB_issues.py` |

**Gotchas, per hook.** The template included above already carries the full explanation for each one — this table is only an index of what to double-check and where it's demonstrated, not a restatement:

| Hook | Watch for | Model |
|---|---|---|
| a. `__init__` | Override `sequence_nicknames` only when genuinely needed; if the transform matches an underscore-containing substring, build it from `self.sequence_names` (raw), not the already-transformed nicknames | `dataset_rgbdtum.py`, `dataset_7scenes.py` |
| b. `download_sequence_data` | A completion marker beats a plain `.exists()` check; exclude marker-suffixed names from any later substring scan; never stash per-sequence state on `self` for another hook to read | `dataset_videos.py`/`dataset_youtube.py`, `dataset_rover.py` |
| c. `create_rgb_folder` | Branch on `self.target_resolution` (`None` → unresized copy, set → `compute_scaled_size` + LANCZOS); depth maps get nearest-neighbor only, never LANCZOS | `dataset_soneva.py`, `dataset_eth.py` |
| d. `create_rgb_csv` | Hardware-synchronized RGB/depth → sort + index-zip; independently-timestamped streams → `pandas.merge_asof(..., direction="nearest")` | `dataset_eth.py` vs. `dataset_rgbdtum.py` |
| e. `create_calibration_yaml` | The written `cam_model` must equal step 1's resolved value, as one indivisible choice (never two separately-returned values that can diverge); cast parsed numerics to `float(...)` before writing | `dataset_youtube.py`, `dataset_kitti.py` |
| g. `create_groundtruth_csv` | Always write the file, even when `groundtruth_available` is false — header row only, never a missing file | `dataset_rgbdtum.py` |
| h. `remove_unused_files` | Retention-tier gating + shared-archive scoping — the hairiest hook; see the dedicated pointer below | `dataset_eth.py` |
| i. `get_download_issues` | Confirm the constraint with a live check (e.g. an anonymous API call) rather than copying it from a same-pattern sibling | `dataset_msd.py`, [#91](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/91) |

**`remove_unused_files` — retention tiers.** Gated by `BENCHMARK_RETENTION` (`path_constants.py`, default `Retention.STANDARD`): `FULL` deletes nothing, `STANDARD` also deletes reformatted intermediates already captured in the standardized layout, `MINIMAL` also deletes the original source downloads. The full per-tier definitions, the `if BENCHMARK_RETENTION != Retention.FULL` / `== Retention.MINIMAL` code shape, and every scoping gotcha (unlink-path mismatches, missing guards, symlinked raw folders, and the four shared-archive scopes — whole-dataset-only, scene/group-scoped, dataset-wide indefinitely reused, exact-file share) are canonical in `dataset_template.py`'s `remove_unused_files` comment (included above). Not restated here.

### Step 5 — Register it

In `Datasets/get_dataset.py`:
- Add `from Datasets.dataset_files.dataset_<name> import <Name>Dataset` under the correct mode section comment (Monocular / RGBD / Stereo / Stereo-VI / Development).
- Add to the `switcher` dict in `get_dataset()`: `"<name>": lambda: <Name>Dataset(),`.

### Step 6 — Smoke-test config + experiment pair

Reference: `test_config_eth.yaml`/`test_exp_eth.yaml` (eth is `mono`+`rgbd`, 97 sequences) — demonstrates both rules below. Follow it, or a closer step-2 sibling, rather than inventing the shape.

**`configs/test_config_<name>.yaml`** — a small, representative subsample, never the whole dataset (even for a small dataset):
```yaml
<dataset_name>:
  - sequence_01
  - sequence_02
```
Representative = sequences exercising different sizes/conditions if heterogeneous, not just the first N alphabetically. See `test_config_eth.yaml` (2 of 97, different scene categories) and `test_config_sweetcorals.yaml` (4 of 13, one per site group, including the one sequence with real calibration/groundtruth).

**`configs/test_exp_<name>.yaml`** — **one block per mode this dataset supports** (step-1 `modes`), not just one total:
```yaml
test_exp_<name>_<baseline>:
  Config: test_config_<name>.yaml
  NumRuns: 1
  Parameters: {verbose: 1, mode: <one of this dataset's modes>, rgb_idx: [0,2000]}
  Module: <baseline>
```
- `<baseline>` (`Module`) must be a pixi environment name from `pixi.toml`'s `[environments]` table — match the closest sibling's choice where possible (`droidslam`/`dpvo` are common lightweight picks for `mono`/`rgbd`); different modes can use different baselines (`test_exp_eth.yaml` does).
- A second baseline for the same mode is worth adding when convenient, not just when a second mode forces it (`test_exp_soneva.yaml`/`test_exp_sweetcorals.yaml`, both `mono`-only, each run two).
- `rgb_idx: [0,2000]` caps the smoke test to the first ~2000 frames — omit only if the matched sibling's convention doesn't use it (`test_exp_videos.yaml`/`test_exp_strayscanner.yaml`). Some siblings (`eth`/`soneva`/`sweetcorals`) instead use `max_rgb`/`step_size` to spread a fixed frame count across the *whole* sequence — a deliberate choice when late-sequence content (e.g. loop closures) matters, not the default to copy elsewhere.

### Step 7 — Add the README row

**Pick the table first.** `README.md` has two same-shaped tables for this mechanism:
- **Datasets** — a fixed, published benchmark with its own sequences (groundtruth typically shipped or derivable).
- **Tools** — a data-capture app/format whose sequences the user brings themselves (e.g. `strayscanner`; no fixed published sequence set).

Both register through the identical `DatasetVSLAMLAB` subclass + YAML + `get_dataset.py` mechanism — the table choice is a README categorization only. If it's unclear which fits, ask the user rather than guessing.

In the chosen table (`| Datasets | Features | Label | Modes | Camera Models |` or `| Tools | Features | Label | Modes | Camera Models |`), append one row as the last real entry — immediately above that table's commented-out placeholder rows for not-yet-implemented entries:
```
| [**<Display Name>**](<homepage URL>) | <feature emoji(s)> | `<dataset_name>` | <modes> | <cam_models> |
```
- `<Display Name>`/`<homepage URL>` — from the YAML's `about:` block.
- `<feature emoji(s)>` — from the legend below the table (Real 📸/Synthetic 💻; Indoor 🏠/Outdoor 🏞️/Underwater 🌊/Intracorporeal 🫀; Handheld 🤳/Headmounted 🥽/Vehicle 🚗/UAV 🚁/Robot 🤖). Ask the user if unclear, don't guess.
- `<modes>` — step-1 list; a mode + its `-vi` variant collapse to one entry (`` `mono(-vi)` ``), backticked, space-separated.
- `<cam_models>` — step-1 list, backticked, space-separated.

### Step 8 — Simulate the download, function by function (required — do not skip)

Even when steps 1–7 look correct on inspection, only running the code catches a wrong URL, a path typo, or a malformed calibration field. Using the **first sequence** in `sequence_names`, drive `download_process`'s hooks yourself, one at a time (`dataset = get_dataset(dataset_name)`, call each method directly) instead of one opaque CLI call:

1. `download_sequence_data`
2. `create_rgb_folder`
3. `create_rgb_csv`
4. `create_calibration_yaml`
5. `create_imu_csv` *(skip if deleted in step 4)*
6. `create_groundtruth_csv`
7. `remove_unused_files`

Track state as you go — this example illustrates a non-`-vi` dataset (`create_imu_csv` already deleted in step 4, so it's skipped and everything renumbers accordingly); include it as its own row, between `create_calibration_yaml` and `create_groundtruth_csv`, if your dataset has it:

| # | Function | State |
|---|---|---|
| 1 | `download_sequence_data` | processed |
| 2 | `create_rgb_folder` | running |
| 3 | `create_rgb_csv` | to be run |
| 4 | `create_calibration_yaml` | to be run |
| 5 | `create_groundtruth_csv` | to be run |
| 6 | `remove_unused_files` | to be run |

After each function, report in detail:
- **Inputs**: `sequence_name` + state/files it consumed.
- **Output**: exact paths created, file/image counts, folder sizes; row counts + first rows for `create_rgb_csv`/`create_groundtruth_csv`; actual `focal_length`/`principal_point`/`image_dimension` values for `create_calibration_yaml`; paths actually deleted for `remove_unused_files`.

This is slower than letting the pipeline run silently, but it pinpoints which stage produced bad output instead of only learning after the fact.

Only move to step 9 once every row reads `processed`. If a hook fails, fix `dataset_<name>.py`/`.yaml` and re-run from that hook (or from the top if the fix touches `download_sequence_data`/`__init__`) — don't commit an unproven dataset.

### Step 9 — Commit

**Before staging, check whether this run surfaced anything new** — a download sub-pattern, a mode/camera-model nuance, a gotcha, a stale or missing `Model:` citation — that isn't already covered by `SKILL.md`, `CLAUDE.md`, `dataset_template.py`, or `dataset_template.yaml`. All four are out of this skill's file scope (per the hard constraint above), so don't edit them directly — use the Issue exception to file it (`improvement` label) so a future doc pass can fold it in. This is the same final-sweep habit `Datasets/extra-files/dataset_cleanup_log.md` (checklist item 16) codifies for cleanup passes, applied here so it isn't only ad hoc.

Stage exactly the files this skill created/modified, by name — never `git add -A`/`git add .`:
- `Datasets/dataset_files/dataset_<name>.py`
- `Datasets/dataset_files/dataset_<name>.yaml`
- `Datasets/get_dataset.py`
- `configs/test_config_<name>.yaml`
- `configs/test_exp_<name>.yaml`
- `README.md`
- `Datasets/extra-files/dataset_table.md`

Run `git status` first and confirm the staged set matches this list exactly. Commit with a concise message like `Add <name> dataset` (check `git log --oneline -10` for style). Local commit only — never push, force-push, or amend.

---

Full reference docs live on the project's GitHub Wiki if more detail is needed.
