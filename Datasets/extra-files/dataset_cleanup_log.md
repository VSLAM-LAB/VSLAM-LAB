# Dataset cleanup log

Running log of dataset-file cleanup passes, kept so each new dataset's cleanup can reuse what
was already checked/decided elsewhere instead of re-deriving it. Not part of the add-dataset
skill's file scope — this is a separate, ongoing hygiene pass across existing `Datasets/dataset_files/*`.

## Checklist (apply per dataset, in order)

1. **Import grouping** — `from __future__ import annotations` first, then up to three blank-line-separated groups (stdlib, third-party, project-local), plain `import x` lines before `from x import y` lines within each group, alphabetical case-insensitive within each of those two subsets (SKILL.md step 4). Also check exactly two blank lines separate the import block's last module-level statement from the class definition (PEP8), same as any other top-level def.

2. **Header consistency** — `.py` module docstring vs. its own `.yaml` `vslamlab_maintainer:` block vs. `Datasets/extra-files/dataset_template.py`'s header comment. Specifically:
   - `.py` `Author` line == yaml `vslamlab_maintainer.name` (verbatim).
   - `.py` `Assisted by` line == yaml `vslamlab_maintainer.assisted_by`, both set to `None` (not omitted) if no AI agent was involved.
   - `.py` `Created` == yaml `vslamlab_maintainer.date`, *unless* the dataset predates the maintainer-block convention, in which case `Created` = original integration date and yaml `date` = last maintainer touch date (allowed to diverge, but should be a deliberate, logged decision, not drift).
   - `.py` `Updated` line present iff the file has been substantively edited since `Created` (add it the same pass as the edit that prompts it, not proactively).
   - `License` line text matches template exactly (`GPLv3 License`).

3. **`__init__` contents** — check for logic already handled by `DatasetVSLAMLAB.__init__` (base class) being redundantly redone in the subclass, e.g. re-reading `target_resolution` from `cfg` after `super().__init__()` already set it. Only dataset-/download-pattern-specific fields belong in the subclass (`url_download_root`/`hf_repo_id`/mode-specific fields like `depth_factor`) — these should generally stay out of the base class rather than being hoisted there, since they're mutually exclusive per download pattern or specific to one mode/dataset, unlike genuinely universal fields (`sequence_names`, `rgb_hz`, `modes`, `cam_models`, `target_resolution`). Also check the `__init__` signature itself: every dataset file in the repo (checked all ~30, not just eth/soneva/sweetcorals) uses `def __init__(self, dataset_name: str = "<name>") -> None: super().__init__(dataset_name)` — flag any file that hardcodes the name inline or drops the type hints instead.

4. **Redundant yaml reopen** — subclass `__init__` should read dataset-specific fields from `self.cfg` (set by `DatasetVSLAMLAB.__init__`), not reopen/re-parse `self.yaml_file` itself. If a subclass still does `with open(self.yaml_file...) as f: cfg = yaml.safe_load(f)`, switch it to `self.cfg` and drop the now-unused `import yaml` if nothing else in the file needs it.

5. **Unused `sequence_path` locals** — `sequence_path = self.sequence_path(sequence_name)` assigned but never read afterward (the method builds `rgb_path`/other paths via other helpers instead). **Closed**: all instances fixed across `dataset_soneva.py`/`dataset_sweetcorals.py`'s `create_rgb_csv`/`create_calibration_yaml`/`create_groundtruth_csv`.

6. **`check_sequence_integrity` mode coverage (base class)** — `DatasetVSLAMLAB.check_sequence_integrity` only conditionally checks `rgb_1/` (stereo) and IMU CSV (mono-vi); there's no check for `depth_0/` when `'rgbd' in self.modes`. A mono-only download can get marked `"available"` and skip re-download even though `depth_0/` was never fetched, for any rgbd dataset. Tracked in [#76](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/76). Not dataset-file-specific, so this is a `DatasetVSLAMLAB.py` fix, not something a per-dataset pass can resolve on its own.

7. **`remove_unused_files` / `Retention` tiers** — `FULL` deletes nothing; `STANDARD` (default) deletes intermediate files that are pure reformats of data already captured in the standardized layout (no information loss, e.g. eth's per-frame `.txt` files once parsed into `.csv`/`.yaml`); `MINIMAL` additionally deletes the *original source* downloads (archives, un-resized raw images) that would need a fresh download to reproduce. In code: `if BENCHMARK_RETENTION != Retention.FULL: <STANDARD-tier deletes>` then `if BENCHMARK_RETENTION == Retention.MINIMAL: <MINIMAL-tier deletes too>` — a dataset with no purely-reformatted intermediates (e.g. soneva/sweetcorals) only needs the second check. Full definition in `dataset_template.py`'s `remove_unused_files` comment.

8. **`get_download_issues` return shape (base class)** — must return `list[dict]` (`_get_dataset_issue(...)`'s output, one call per issue), never a bare `dict`. `DatasetVSLAMLAB.py`'s default previously mismatched this (`-> dict: return {}`); fixed to `-> List[dict]: return []`. Check any new implementation actually returns a list, not a single dict or a bare `_get_dataset_issue(...)` call unwrapped in a list.

9. **YAML formatting** — `sequence_names` list items indented 2 spaces (`  - name`), matching `dataset_template.yaml`/`dataset_eth.yaml`/the repo-wide majority — not flush-left. File ends with a trailing newline. `about:`/`vslamlab_maintainer:` key order and quoting style should match sibling yamls (see checklist item 2 for the `vslamlab_maintainer:` block specifically).

10. **Template self-containment** — `dataset_template.yaml`/`dataset_template.py` should be usable as a standalone starting point without needing to cross-reference SKILL.md or another dataset's files for basic facts (closed-list values, which yaml field goes with which download pattern, etc.) — a pointer comment alone (e.g. "see dataset_hilti2026.yaml") is a gap unless the actual shape/values are also inlined. Before trusting `dataset_table.md`-derived facts (closed-list values, Download column, etc.), regenerate it first (`python3 Datasets/extra-files/generate_dataset_table.py`) — a stale copy silently drifts from the yaml files it's generated from.

11. *(next checks TBD as they come up, e.g. field-value validation)*

## Entries

### 2026-07-25 — eth / soneva / sweetcorals: header consistency pass

Files: `dataset_eth.py`/`.yaml`, `dataset_soneva.py`/`.yaml`, `dataset_sweetcorals.py`/`.yaml`.

Findings:
- All three `.py` files had `Author: Alejandro Fontan Villacampa`, but all three yamls had `vslamlab_maintainer.name: Alejandro Fontan` (no "Villacampa") — consistent with each other, but violated SKILL.md step 4's "Author = vslamlab_maintainer.name" rule.
- `dataset_eth.py`'s `Created: 2024-07-13` didn't match `dataset_eth.yaml`'s `vslamlab_maintainer.date: 2026-03-07` (~20 months apart) — eth predates the maintainer-block convention, so these two dates carry different meanings.
- `dataset_eth.py` had an `Updated:` header line (not in the template at all); soneva/sweetcorals didn't. CLAUDE.md's stated convention for newer files is `Author/Version/Created/Updated`, so `Updated` is a legitimate field, just not yet reflected in the template/skill docs.

Decisions (user-confirmed):
- Standardized on the short form: all three `.py` `Author` lines changed to `Alejandro Fontan` (dropped "Villacampa"). Yaml `vslamlab_maintainer.name` values were already correct, untouched.
- `dataset_eth.yaml`'s `vslamlab_maintainer.date` changed from `2026-03-07` to `2024-07-13` to match `Created` (treats `date` as "originally added", not "last touched").
- Added `Updated: 2026-07-25` to `dataset_soneva.py`/`dataset_sweetcorals.py` headers, justified since this pass itself edited both files (the Author line change).

Commit: `e0b2b51` (bundled together with the Assisted-by convention change below — both landed in one commit)

Follow-up / not fixed: `Datasets/extra-files/dataset_template.py`'s header comment doesn't mention `Updated` at all, and neither does SKILL.md step 4. Worth adding to the template/skill later so this doesn't need rediscovering per dataset — flagged, not changed here (template is skill-owned, out of scope for this ad hoc pass).

### 2026-07-25 — `Assisted by` convention change: omit → explicit `None`

Prompted by adding `Assisted by: Claude (Sonnet 5)` to `dataset_eth.py`/`.yaml` (this cleanup pass was itself Claude-assisted, so eth should say so like soneva/sweetcorals do).

While doing that, changed the underlying convention itself (user-requested): `Assisted by` should always be present, filled with `None` when no AI agent was involved, rather than omitted entirely. Old convention ("omit the line/field if absent") was ambiguous — a missing field silently drifted into "not established" instead of "confirmed no agent" for older files that predate the convention.

Files touched:
- `dataset_eth.py` — added `- Assisted by: Claude (Sonnet 5)` header line.
- `dataset_eth.yaml` — added `assisted_by: Claude (Sonnet 5)` to `vslamlab_maintainer:`.
- `Datasets/extra-files/dataset_template.py` — placeholder line reworded from "(omit this line entirely if no AI coding agent was involved)" to "<agent name, or None if no AI coding agent was involved>".
- `Datasets/extra-files/dataset_template.yaml` — comment reworded from "omit this line entirely if no agent was involved" to "set to None if no agent was involved".
- `.claude/skills/add-dataset/SKILL.md` step 4 — "(omit the line if absent)" → "(write `None` if absent)".

Not done (scope creep beyond what was asked): existing datasets that currently omit `Assisted by` entirely (i.e., every dataset added before this convention existed, minus eth/soneva/sweetcorals) were **not** retrofitted with `Assisted by: None`. Add that as its own checklist item/pass if the convention should be back-filled repo-wide.

### 2026-07-25 — eth / soneva / sweetcorals / template: import-grouping pass

Files: `dataset_eth.py`, `dataset_soneva.py`, `dataset_sweetcorals.py`, `Datasets/extra-files/dataset_template.py`.

Verified programmatically (not just by eye): stdlib/third-party/project-local grouping, plain-`import`-before-`from`-`import` ordering, and case-insensitive alphabetical order within each subgroup — all four files fully comply with SKILL.md step 4's convention. No changes needed there.

Finding: `dataset_template.py` had only one blank line between the last import and `class TemplateDataset`, while all three real dataset files use two (PEP8 standard for top-level defs). Fixed — template now has two blank lines before the class, matching eth/soneva/sweetcorals.

Commit: `466214e`

### 2026-07-25 — `__init__` review: dropped redundant `target_resolution` re-read, base-class-hoisting question

Files: `Datasets/DatasetVSLAMLAB.py` (read-only reference), `dataset_eth.py`, `dataset_soneva.py`, `dataset_sweetcorals.py`, `Datasets/extra-files/dataset_template.py`, `.claude/skills/add-dataset/SKILL.md`.

Finding: `dataset_template.py`'s `__init__` re-read `target_resolution` from `cfg` and reassigned `self.target_resolution`, even though `DatasetVSLAMLAB.__init__` (base class, run via `super().__init__()`) already sets it from the same yaml field with identical fallback logic (`DatasetVSLAMLAB.py:61-63`) — dead, duplicated logic, apparently left over from before `target_resolution` got hoisted to the base class. The template's own comment cited `dataset_sweetcorals.py` as the model, but that file (and `dataset_soneva.py`) correctly *don't* re-read it — the citation didn't match the code it was pointing at. SKILL.md step 4a had the same stale instruction (telling implementers to re-read `target_resolution` in `__init__`, also (wrongly) citing `dataset_soneva.py`). Fixed both: template's `__init__` no longer re-reads it (replaced with a one-line comment pointing at the base class), and SKILL.md step 4a's instruction updated to match.

Question raised: should `depth_factor` (eth-only)/`url_download_root`/`hf_repo_id` also move to the base class, following `target_resolution`'s precedent? Recommendation: no — `target_resolution` fit the base class because it's genuinely universal (every dataset uses the same key/fallback). These three aren't: `url_download_root`/`hf_repo_id` are mutually exclusive per download pattern (most datasets have one or neither), and moving them would force switching from `cfg["key"]` (fail-fast `KeyError`) to `cfg.get("key")`, losing that validation. `depth_factor` is currently eth-only and rgbd-specific — putting it on the ABC would put an rgbd concept on every mono-only dataset. Left as subclass-owned, per SKILL.md's existing (and correct) step 4a guidance. No code changed for this part — decision only.

Commit: `466214e`

### 2026-07-25 — `__init__` consistency/bug/clarity pass

Files: `dataset_eth.py`, `dataset_soneva.py`, `dataset_sweetcorals.py`, `Datasets/extra-files/dataset_template.py` (+ grepped all ~30 `Datasets/dataset_files/*.py` for the signature check, read-only).

Finding (bug): `dataset_template.py`'s `__init__` didn't match the `__init__` signature every other dataset file in the repo uses (confirmed via grep across all dataset files, not just eth/soneva/sweetcorals) — `def __init__(self, dataset_name: str = "<name>") -> None: super().__init__(dataset_name)`. The template instead had `def __init__(self):` (no parameter, no return hint) with `super().__init__('dataset_name_template')` hardcoded inline. Fixed to match the universal pattern: `def __init__(self, dataset_name: str = "dataset_name_template") -> None:` / `super().__init__(dataset_name)`.

Finding (clarity nit): template docstring's sync note said "same assisted_by (if any)" — stale wording from the old omit-if-absent convention (see the `Assisted by` convention-change entry above). Since `assisted_by` is now always present (agent name or `None`), reworded to "same assisted_by" (dropped "(if any)").

No other issues — structural flow (`super().__init__()` → reopen yaml → pull dataset-specific fields) and the `target_resolution` handling from the previous pass all still consistent across eth/soneva/sweetcorals/template.

Commit: `466214e`

### 2026-07-25 — Drop redundant yaml reopen: subclasses now read `self.cfg`

Files: `Datasets/DatasetVSLAMLAB.py`, `dataset_eth.py`, `dataset_soneva.py`, `dataset_sweetcorals.py`, `Datasets/extra-files/dataset_template.py`, `.claude/skills/add-dataset/SKILL.md`.

User asked whether `super().__init__()` could return the parsed cfg so subclasses don't reopen/re-parse the same yaml file — every one of the ~31 dataset files reopens `self.yaml_file` inside its own `__init__` even though the base class already opened and parsed it. Python constraint: `__init__` can't return a value (`TypeError` if it does), so the literal ask isn't possible — the equivalent is stashing it as an instance attribute instead.

Decision: added `self.cfg: dict = cfg` in `DatasetVSLAMLAB.__init__` (`DatasetVSLAMLAB.py:51`, purely additive/backward-compatible — every existing subclass still works unchanged). Scoped the actual "stop reopening the file" fix to this pass's 4 files only (user's explicit choice over doing all ~31 or none): `dataset_eth.py`/`dataset_soneva.py`/`dataset_sweetcorals.py`/`dataset_template.py` now read `self.cfg["..."]` directly instead of `with open(self.yaml_file...) as f: cfg = yaml.safe_load(f)`, and dropped the now-unused `import yaml` from each (only `Datasets/DatasetVSLAMLAB.py` still needs it). SKILL.md step 4a updated to instruct `self.cfg` instead of reopening the yaml.

Verified: all 5 files parse (`ast.parse`) and `EthDataset()`/`SonevaDataset()`/`SweetcoralsDataset()` instantiate correctly under `pixi run -e vslamlab python` with `self.cfg` populated and all dataset-specific fields (`url_download_root`, `depth_factor`, `hf_repo_id`, `target_resolution`) reading correctly from it.

Not done (scope creep beyond what was asked, logged per user's explicit choice): the other ~27 dataset files still reopen `self.yaml_file` redundantly in their own `__init__` — harmless (still correct, just duplicate I/O+parsing) but inconsistent with these 4 now. Checklist item 4 added above so this gets caught file-by-file as each dataset gets its own cleanup pass, rather than doing a repo-wide sweep in one shot.

Commit: `466214e`

### 2026-07-25 — Final `__init__` pass: base class's own header had gone stale

Files: `Datasets/DatasetVSLAMLAB.py`.

Finding: adding `self.cfg` to `DatasetVSLAMLAB.__init__` (previous entry, same day) was a real code change to this file, but its own header wasn't updated to match — `Updated:` was still `2025-12-30`, violating the same "Updated present/accurate iff substantively edited" rule we've been applying to every dataset file. Fixed: `Updated` bumped to `2026-07-25`. `Author`/`Version` left untouched — `DatasetVSLAMLAB.py` isn't a per-dataset file with a paired yaml `vslamlab_maintainer:` block to sync against (CLAUDE.md's stated header convention for core files is just `Author/Version/Created/Updated`, no `Assisted by`), so that part of the dataset-file convention doesn't apply here by design, not by oversight.

With this, the `__init__` review across `DatasetVSLAMLAB.py`/`dataset_eth.py`/`dataset_soneva.py`/`dataset_sweetcorals.py`/`dataset_template.py` turned up no further issues: signatures, structure, `self.cfg` usage, and header accuracy are all consistent.

Commit: `466214e`

### 2026-07-25 — `download_sequence_data` pass

Files: `dataset_eth.py`, `dataset_soneva.py`, `dataset_sweetcorals.py`, `Datasets/extra-files/dataset_template.py`.

`download_sequence_data` itself is clean across all three real files — each matches its source pattern (website vs. hugging-face) from SKILL.md/the template, idempotency is handled appropriately per source type (eth's explicit `.exists()` checks vs. `ensure_hf_sequence_download`'s built-in resumable/idempotent fetch), no dead code or logic issues.

Finding (minor, fixed): `dataset_soneva.py` named the resolved remote top-level folder `remote_name`; `dataset_sweetcorals.py` — the closest possible sibling, sharing `HFColmapDatasetMixin` and mirroring soneva's structure — named the identical concept `remote_folder`. Renamed soneva's to `remote_folder` to match.

Finding (flagged, not fixed — outside `download_sequence_data` itself): confirmed by reading the code (not just the Pylance diagnostic) that `sequence_path = self.sequence_path(sequence_name)` is genuinely dead in 5 other methods — `dataset_soneva.py`'s `create_rgb_csv`/`create_calibration_yaml`/`create_groundtruth_csv`, `dataset_sweetcorals.py`'s `create_calibration_yaml`/`create_groundtruth_csv` — each uses `rgb_path`/other helpers instead and never reads `sequence_path` again. Added as checklist item 5 for whichever pass reviews those specific hooks.

Commit: `90ac921`

### 2026-07-25 — `create_rgb_folder` pass: near-miss, caught by user before landing

Files: `dataset_eth.py`, `dataset_soneva.py`/`HFColmapDatasetMixin` (shared by `dataset_sweetcorals.py`), `Datasets/extra-files/dataset_template.py`.

eth.py's `create_rgb_folder` (a plain `rgb`→`rgb_0`/`depth`→`depth_0` rename) has no issues.

Near-miss (proposed, then reverted before commit): in `HFColmapDatasetMixin.create_rgb_folder`, `target_size` is computed once from the *first* image and reused for the whole sequence; a later image with a different original size still gets forced into that same `target_size` (the code warns about the mismatch but resizes to the cached size anyway). Read in isolation this looks like a bug — `compute_scaled_size`'s own docstring says "preserving aspect ratio," and it's cheap (pure arithmetic, no I/O), so recomputing per image looked strictly better. Proposed switching to per-image `compute_scaled_size` calls.

**User caught the flaw before it landed**: `create_calibration_yaml` computes `focal_length`/`principal_point` from a *single* reference image's resized dimensions (`next(rgb_path.iterdir())`) and applies that one calibration to every frame in the sequence — valid only if every resized image ends up at the *same* final pixel dimensions. Per-image resizing would let differently-shaped source images produce differently-shaped outputs, silently invalidating calibration for those frames. The original cached-`target_size` behavior (normalize every frame to one common size, warn-but-still-conform on outliers) is the correct one. Change reverted; no net diff to `create_rgb_folder` from this entry (the `remote_folder` rename from the previous pass is unrelated and still in place).

Lesson for future passes: before "fixing" resize/calibration-adjacent logic, check downstream consumers (`create_calibration_yaml` in particular) for assumptions the change might break — a locally-cheap, locally-correct-looking change can violate a whole-sequence invariant a different method depends on.

Commit: `90ac921`

### 2026-07-25 — `create_rgb_folder` template comment: missing depth_0/rgbd coverage

Files: `Datasets/extra-files/dataset_template.py`.

Finding (user-spotted): the template's `create_rgb_folder` comment mentioned `rgb_1/` for stereo modes but said nothing about `depth_0/` for rgbd modes, and didn't cite `dataset_eth.py` (an rgbd dataset, one of our three real models) anywhere — a real gap, since `create_rgb_csv`'s comment right below it *does* cover all three headers (mono/stereo/rgbd) consistently.

First fix attempt was itself wrong and got corrected by the user: I initially wrote that `depth_0/` is generally handled outside the resize branch, "normally just a plain rename/copy at original resolution." User corrected this — rgbd depth *does* need resizing when the source requires it, following the same `self.target_resolution` branch as `rgb_0`/`rgb_1`; `dataset_eth.py`'s depth being an unresized plain rename is specific to ETH3D's source already being close enough to 640x480 that `eth.yaml` sets no `target_resolution` at all (nothing gets resized for eth, not just depth) — not a general rule that depth is exempt from resizing.

Corrected fix: template now says rgbd's `depth_0/` follows the same `target_resolution` branch as `rgb_0`/`rgb_1`, but must use a non-interpolating resample (nearest-neighbor — `Image.NEAREST`/`cv2.INTER_NEAREST`) instead of LANCZOS, since interpolating resample blends depth values across object boundaries and corrupts the metric data. `dataset_eth.py` is cited specifically as the depth_0/ folder-layout model, with an explicit note that its lack of resizing is dataset-specific, not a general pattern.

Commit: `90ac921`

### 2026-07-25 — `create_rgb_csv` pass

Files: `dataset_eth.py`, `dataset_soneva.py`.

Fixed (checklist item 5): removed the dead `sequence_path = self.sequence_path(sequence_name)` from `HFColmapDatasetMixin.create_rgb_csv` (`dataset_soneva.py`, shared by `dataset_sweetcorals.py`) — genuinely unused, the method only ever reads `rgb_path`.

Finding (fixed): `dataset_eth.py`'s `create_rgb_csv` hand-rolled the same open/`csv.writer`/tmp/`.replace()` pattern that `utilities.write_csv_rows` already encapsulates byte-for-byte (confirmed via its docstring: "the atomic write-then-replace pattern used throughout `Datasets/dataset_files/*.py`"). `dataset_soneva.py`/`dataset_sweetcorals.py` already use `write_csv_rows`; `dataset_eth.py` predates it. Refactored `create_rgb_csv` to build a `rows` list and call `write_csv_rows`, matching the newer files. `import csv` stays in `dataset_eth.py` — still used by `create_groundtruth_csv`, out of scope for this pass.

Verified: both files parse; `EthDataset()`/`SonevaDataset()`/`SweetcoralsDataset()` still instantiate correctly under `pixi run -e vslamlab python`; a standalone `write_csv_rows` call with eth-shaped rows produces byte-identical header/row output to the old hand-rolled version.

Not done (scope creep, logged): `write_csv_rows` is only adopted by 2 of ~19 dataset files that hand-roll the same csv-writing pattern (`dataset_kitti.py`, `dataset_hilti2022.py`, `dataset_madmax.py`, etc. — checked via grep) — same "new shared utility, not yet repo-wide" situation as `self.cfg`. Left the other ~17 files alone; only `dataset_eth.py` (in scope for this cleanup) was migrated.

Commit: `d3108dd`

### 2026-07-25 — `create_calibration_yaml` pass

Files: `dataset_eth.py` (reviewed, no changes needed), `dataset_soneva.py`, `dataset_sweetcorals.py`.

Fixed (checklist item 5): removed dead `sequence_path` from both `dataset_soneva.py`'s and `dataset_sweetcorals.py`'s `create_calibration_yaml`.

Finding (fixed, user-approved before implementing): `dataset_soneva.py`'s entire `create_calibration_yaml` body and `dataset_sweetcorals.py`'s pinhole branch were near byte-identical (COLMAP camera lookup → pinhole intrinsics → rescale to resized image → build `rgb` dict) — duplicated logic that `HFColmapDatasetMixin`'s own docstring already flagged as "not byte-identical, kept separate" without noting that everything *except* the `camera_id` resolution actually was identical. Extracted a new `HFColmapDatasetMixin._pinhole_rgb_calibration(sequence_name, camera_id) -> dict[str, Any]` helper covering the shared part; each subclass's `create_calibration_yaml` now only resolves its own `camera_id` (soneva: `raw_to_colmap` mapping; sweetcorals: `_PINHOLE_LEFT_PREFIX` match) and calls the helper.

Import cleanup from the extraction: `dataset_sweetcorals.py` no longer uses `read_colmap_cameras` or `PIL.Image` directly (moved into the helper in `dataset_soneva.py`), and dropped `from typing import Any` (its only remaining use, the `rgb: dict[str, Any]` annotation, was removed along with the inlined dict).

Verified against real previously-downloaded data (not just syntax/instantiation): regenerated `calibration.yaml` for `soneva/hb_20250710`, `sweetcorals/tabuhan_p1` (pinhole branch), and `sweetcorals/watudodol_p1` (non-pinhole/"unknown" branch) against a saved pre-refactor baseline of each — all three byte-identical after the refactor.

`dataset_eth.py`'s `create_calibration_yaml` (per-sequence `calibration.txt` parsing, unrelated pattern) reviewed, no issues found.

Commit: `5173f18`

### 2026-07-25 — `create_groundtruth_csv` pass

Files: `dataset_eth.py`, `dataset_soneva.py`, `dataset_sweetcorals.py`.

Fixed (checklist item 5, now fully closed): removed dead `sequence_path` from `dataset_soneva.py`'s and `dataset_sweetcorals.py`'s `create_groundtruth_csv` — the last two remaining instances.

Bug found and fixed: `dataset_sweetcorals.py`'s `create_groundtruth_csv` did `if sequence_name != _PINHOLE_SEQUENCE: return` — for every non-pinhole sequence (12 of 13) it wrote **no `groundtruth.csv` at all**, violating the template's own documented convention ("Always create this file... write just the header row... rather than... leaving no file at all. Model: `dataset_videos.py`"), which `dataset_videos.py` itself confirms in practice. Confirmed live: the already-downloaded `watudodol_p1` (non-pinhole) had no `groundtruth.csv` on disk. Fixed by writing a header-only `groundtruth.csv` via `write_csv_rows(groundtruth_csv, header, [])` for non-pinhole sequences instead of returning early.

Also refactored (matching the `create_rgb_csv` pass's precedent): `dataset_eth.py`'s `create_groundtruth_csv` hand-rolled the same open/`csv.writer`/tmp/`.replace()` pattern `write_csv_rows` already covers. Refactored to build a `rows` list and call `write_csv_rows`; `import csv` was then fully unused in the file (its only other use was the `create_rgb_csv` this same pattern was already removed from) and was dropped.

Verified: `table_3` (the only locally-downloaded eth sequence) already had its raw `groundtruth.txt` removed by `remove_unused_files` at this retention level, so the eth refactor was verified with a synthetic before/after comparison (crafted `groundtruth.txt` with blank lines/comments/multiple rows) instead — byte-identical old vs. new. soneva/sweetcorals were verified against real downloaded data: regenerated `groundtruth.csv` for `soneva/hb_20250710` and `sweetcorals/tabuhan_p1` (pinhole) — both byte-identical to pre-fix baselines (dead-variable removal doesn't change output) — and `sweetcorals/watudodol_p1` (non-pinhole) now correctly produces a header-only file where none existed before.

Commit: `cce205d`

### 2026-07-25 — `remove_unused_files` pass: documented the 3 `Retention` levels in the template

Files: `Datasets/extra-files/dataset_template.py`.

User asked to write out what each `Retention` level (`path_constants.py`: `MINIMAL`/`STANDARD`/`FULL`, default `STANDARD`) keeps vs. removes, since the enum itself carries no docstring and the convention was only implicit across dataset files. Derived the convention by reading every `BENCHMARK_RETENTION`/`Retention` usage across all dataset files that have one (~15, via grep, not just eth/soneva/sweetcorals) — the pattern is consistent repo-wide: `if BENCHMARK_RETENTION != Retention.FULL:` guards deleting intermediate files that are pure reformats of data already captured in the standardized layout (no information loss); `if BENCHMARK_RETENTION == Retention.MINIMAL:` guards additionally deleting the *original source* downloads (archives, un-resized raw images) that would otherwise require a fresh download to reproduce.

Rewrote `dataset_template.py`'s `remove_unused_files` comment block to spell out all three tiers explicitly (`FULL` = delete nothing, `STANDARD` = delete redundant reformats only, `MINIMAL` = also delete original source downloads), with the two-check code shape and both `dataset_eth.py` (both tiers) and `HFColmapDatasetMixin` (`MINIMAL`-only) cited as models.

Reviewed `remove_unused_files` itself against this now-documented convention: `dataset_eth.py` and `HFColmapDatasetMixin` (soneva/sweetcorals) both already match it exactly — no code changes needed. The mixin's lack of a `!= FULL`/`STANDARD`-tier check isn't an inconsistency: soneva/sweetcorals have no purely-reformatted intermediate files to clean (they don't parse raw per-frame text files the way eth does), only the `MINIMAL`-tier `rgb_0_raw/`.

Commit: `cce205d`

### 2026-07-25 — `get_download_issues` pass: base class return-type mismatch

Files: `Datasets/DatasetVSLAMLAB.py`.

Bug found and fixed: the base class's default `get_download_issues(self, sequence_names: List[str]) -> dict: return {}` was typed and implemented as returning a `dict`, but every actual implementation returns a `list[dict]` — `HFColmapDatasetMixin.get_download_issues` (`dataset_soneva.py`, shared by sweetcorals) returns `[_get_dataset_issue(...)]`, `_get_dataset_issue` (`DatasetVSLAMLAB_issues.py`) itself returns a single dict (not a list), and the sole caller (`vslamlab_utilities.py:373`, `for issue_seq in issues_seq: ... issue_seq['name']`) only works for a list of dicts — iterating an actual non-empty dict would iterate its keys (strings) and crash on `issue_seq['name']`. The template's own comment already correctly said "Return a list of dicts..." — only the base class's signature/default was wrong. Not a live bug today (both `{}` and `[]` are falsy, so the empty default behaves identically either way), but a trap for the next dataset that trusts the type hint literally. Fixed: `-> List[dict]`, `return []`.

Reviewed the 3 real files against this: none needed changes. `dataset_eth.py` correctly has no override (no auth constraint, matches SKILL.md step 1 "leave blank if none"); `dataset_sweetcorals.py` correctly has no override either, inheriting the HF-token check from `HFColmapDatasetMixin` via multiple inheritance rather than duplicating it; the mixin's own implementation was already correct.

Verified: `EthDataset().get_download_issues(['x'])`, `SonevaDataset().get_download_issues(['x'])`, `SweetcoralsDataset().get_download_issues(['x'])` all return `[]` under `pixi run -e vslamlab python` (HF token is set in this environment, so the mixin's no-issue branch is exercised for soneva/sweetcorals; eth exercises the base class default).

Commit: `22bc9e1`

### 2026-07-25 — `.yaml` files pass

Files: `dataset_eth.yaml`, `dataset_soneva.yaml`, `dataset_sweetcorals.yaml`, `Datasets/extra-files/dataset_template.yaml`.

(Header/`vslamlab_maintainer:` block consistency was already covered in the earlier header-consistency and Assisted-by-convention passes — this pass covered everything else: field order, formatting, quoting.)

Finding (fixed): `sequence_names` list-item indentation was inconsistent — `dataset_eth.yaml`/`dataset_template.yaml` indent items 2 spaces (`  - cables_1`), but `dataset_soneva.yaml`/`dataset_sweetcorals.yaml` had them flush left (`- hb_20250710`, no indent). Checked indentation across all ~33 dataset yamls (not just our 4): 28+ use the 2-space style, only soneva/sweetcorals (in scope) plus `dataset_videos.yaml`/`dataset_vitum.yaml` (out of scope, not touched) deviate. Reindented soneva/sweetcorals's `sequence_names` to 2 spaces to match the dominant convention; verified both still parse with identical `sequence_names` content/count (23 and 13 respectively) after the change.

Finding (fixed): `dataset_template.yaml` was missing a trailing newline at end-of-file (the other three all have one). Added.

No other issues: `about:`/`vslamlab_maintainer:` field order, `modes`/`cam_models` values (checked against SKILL.md's closed lists), quoting style, and blank-line section spacing are all already consistent across the four files.

Not done (scope creep, logged): `dataset_videos.yaml` and `dataset_vitum.yaml` have the same `sequence_names` indentation inconsistency (flush-left and 1-space respectively) but are outside this cleanup's eth/soneva/sweetcorals/template scope — left alone.

Commit: `7ffb1f5`

### 2026-07-25 — `dataset_template.yaml`: inlined a real `about:` placeholder block

User-spotted: `dataset_template.yaml` had only a one-line pointer comment for `about:` ("see dataset_hilti2026.yaml for the full shape") instead of an actual fill-in-the-blank block like `sequence_names`/`vslamlab_maintainer` both have — inconsistent with the template's own established pattern of being a concrete, ready-to-edit starting point rather than a set of pointers elsewhere.

Added a real `about:` block (`license`/`summary`/`homepage`/`authors`, matching `dataset_hilti2026.yaml`'s shape and the field order already used by `dataset_eth.yaml`/`dataset_soneva.yaml`/`dataset_sweetcorals.yaml`), placed between `sequence_names` and `vslamlab_maintainer` to match where it sits in all three real files. Kept a short comment clarifying `authors` means the dataset's original creators, not the `vslamlab_maintainer` (the VSLAM-LAB integrator) — a distinction that isn't obvious from the field name alone.

Verified: file still parses; `about` block loads with the expected 4 keys.

Commit: `7ffb1f5`

### 2026-07-25 — `dataset_template.yaml`: documented the download-source field alternatives

User-spotted: `url_download_root: ""` was presented with no indication it's only one of several mutually-exclusive download-source fields depending on pattern (website/google-drive use `url_download_root`, hugging-face uses `hf_repo_id`, local uses `sequence_location`, and website can alternatively use `url_download_sequences` for a per-sequence-URL table) — that mapping was only documented in `dataset_template.py`'s `__init__` comment, not in the yaml template itself, so someone filling out just the yaml first (SKILL.md step 3, before step 4's `.py`) could easily keep the wrong field.

Added a comment above `url_download_root` in `dataset_template.yaml` listing all four patterns and which field(s) each uses, with model citations (`dataset_s3li.yaml` for `url_download_sequences`, `dataset_soneva.yaml`/`dataset_sweetcorals.yaml` for `hf_repo_id`, `dataset_strayscanner.yaml` for `sequence_location`) — condensed from `dataset_template.py`'s existing `__init__` comment rather than duplicating it verbatim, so the two stay easy to keep in sync.

Verified against the actual cited files: `dataset_s3li.yaml`'s `url_download_sequences` is a dict keyed by sequence name; `dataset_strayscanner.yaml`'s `sequence_location` is a plain list, positionally parallel to `sequence_names` (not itself keyed by name) — corrected the comment's wording after an initial draft said "keyed by sequence_name" for the list case, which was imprecise.

Commit: `7ffb1f5`

### 2026-07-25 — Unrelated bug found while regenerating `dataset_table.md`: `generate_dataset_table.py`

Files: `Datasets/extra-files/generate_dataset_table.py`, `Datasets/extra-files/dataset_table.md`.

While regenerating `dataset_table.md` to check the live `cam_models`/`modes` value sets (for the next entry below), the Download column came back wrong for 9 datasets (`ariel`, `hamlyn`, `msd`, `openloris-d400`, `openloris-t265`, `soneva`, `strayscanner`, `sweetcorals`, `videos`) — all showing `other` instead of `hugging-face`. Root cause: `generate_dataset_table.py`'s `_download_labels()` checked `cfg.get("repo_id")`, but every hugging-face dataset yaml in the repo uses `hf_repo_id` (confirmed via grep — zero yaml files use bare `repo_id`) — a permanently-dead check. Not caused by this session's edits; the previously-committed `dataset_table.md` just hadn't been regenerated since the yaml field was renamed `repo_id` → `hf_repo_id` at some earlier point.

Fixed directly (one-line, unambiguous): `cfg.get("repo_id")` → `cfg.get("hf_repo_id")`. Regenerated `dataset_table.md`; all 9 datasets now correctly show `hugging-face`. Also picked up an unrelated stale entry while regenerating: `eth`'s "AI-Assisted" column now correctly shows "Claude (Sonnet 5)" (this session's earlier `Assisted by` addition to `dataset_eth.yaml` had never been reflected in the table until this regeneration).

Filed [#77](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/77) to record the bug and its impact (SKILL.md step 2's cross-check-by-shared-Download-source would have failed to match hugging-face candidates for a new dataset addition).

Commit: `7ffb1f5`

### 2026-07-25 — `dataset_template.yaml`: documented the closed `modes`/`cam_models` value lists

User-spotted, same gap pattern as the `url_download_root` entry above: `modes: ['mono', 'rgbd']` and `cam_models: ['pinhole']` gave no indication these are closed lists (SKILL.md step 1) sourced from `dataset_table.md`'s Modes/Camera Models columns, or what the current valid values are.

Confirmed the live value sets by regenerating `dataset_table.md` (now with the `hf_repo_id` fix from the previous entry): `modes` = `{mono, mono-vi, rgbd, rgbd-vi, stereo, stereo-vi}`, `cam_models` = `{equid4, pinhole, radtan4, radtan5, unknown}` — matches SKILL.md's own hardcoded documentation of both lists exactly. Added comments above each field listing the current values (marked "as of this writing" since SKILL.md says to read the list live, it can grow) plus, for `modes`, the derivation rule from SKILL.md step 1 (native mode + every mode derivable by dropping a channel: stereo/rgbd → mono, -vi → non-vi) with the same worked examples SKILL.md uses.

Commit: `7ffb1f5`

### 2026-07-25 — SKILL.md step 6 rewrite: smoke-test config/exp pair, using eth as the current reference

Files: `.claude/skills/add-dataset/SKILL.md`.

Context: while checking the config-file deletions above, `configs/test_config_eth.yaml`/`test_exp_eth.yaml` had themselves been substantially rewritten by the user during this session — `test_config_eth.yaml` trimmed from 87 to 2 sequences, `test_exp_eth.yaml` gained a second `mode: mono` block (`colmap`) alongside the existing `mode: rgbd` block (`droidslam`), and both blocks were renamed with a `test_exp_` prefix (`test_exp_eth_droidslam`/`test_exp_eth_colmap`, previously `exp_eth_droidslam`). `test_exp_soneva.yaml`/`test_exp_sweetcorals.yaml` got the same block-name prefix change. User then explicitly asked to update SKILL.md step 6 to match, citing eth as the canonical example, and stating two requirements: experiments representative of the dataset's modes, sequences a small representative subsample.

Checked how widespread the `test_exp_` block-name prefix actually is before rewriting: only 5 of 17 `test_exp_*.yaml` files use it (`test_exp_eth.yaml`, `test_exp_soneva.yaml`, `test_exp_sweetcorals.yaml` — all touched this session — plus `test_exp_rgbd.yaml`/`test_exp_stereo.yaml`, pre-existing). The other 12 still use the old unprefixed `exp_<name>_<baseline>:` block name. Same "new convention, not yet backfilled everywhere" situation as `self.cfg`/`write_csv_rows` earlier — SKILL.md governs *future* additions, so it was updated to the new convention without touching the 12 old files.

Rewrote step 6: block-name template changed from `exp_<name>_<baseline>:` to `test_exp_<name>_<baseline>:`; added an explicit rule that `test_exp_<name>.yaml` needs **one block per mode the dataset supports** (not just one experiment total), citing `test_exp_eth.yaml`'s two blocks (`rgbd`/`droidslam`, `mono`/`colmap`) as the model — a single-mode dataset trivially satisfies this with its existing one block. Replaced the `test_config_videos.yaml`/`test_config_strayscanner.yaml` citations for "small representative handful" with `test_config_eth.yaml` (2 of 97 sequences) as the primary example, keeping `test_config_sweetcorals.yaml` (13/13) for the small-dataset-lists-everything exception. Also documented `test_exp_eth.yaml`'s `max_rgb`/`step_size` frame-capping (spreads samples across the whole sequence) as a deliberate alternative to the default `rgb_idx` (early-window truncation) for datasets where late-sequence content matters — not the default to copy without the same reason, matching the "intentional, not a bug" conclusion from the earlier `test_exp_eth.yaml` review in this log.

Also fixed the yaml code-block example's `sequence_names`-equivalent list indentation (`- sequence_01` → `  - sequence_01`, 2-space), matching the yaml-indentation convention fixed in the earlier `.yaml` files pass.

Commit: `00aa003`

### 2026-07-25 — soneva/sweetcorals test_config/test_exp brought in line with the new SKILL.md step 6

Files: `configs/test_config_soneva.yaml`, `configs/test_config_sweetcorals.yaml` (`test_exp_soneva.yaml`/`test_exp_sweetcorals.yaml` reviewed, no changes needed).

Checked both pairs against the just-rewritten SKILL.md step 6:
- Block naming (`test_exp_<name>_<baseline>:`) — already compliant in both (part of the same user edit that prompted the SKILL.md rewrite).
- One experiment block per supported mode — both datasets are mono-only, so their existing single block already satisfies this trivially. No new blocks added.
- `Parameters`/`rgb_idx` shape — already matches the SKILL.md template exactly in both.

Fixed: `test_config_soneva.yaml`/`test_config_sweetcorals.yaml` list-item indentation (flush-left → 2-space), matching `test_config_eth.yaml` and the yaml-indentation convention fixed earlier this log for `Datasets/dataset_files/*.yaml` — these `configs/test_config_*.yaml` files are separate files that pass hadn't touched.

Fixed (user-confirmed): `test_config_soneva.yaml`'s 5 sequences covered 5 of soneva's 6 location groups (`hb`/`ootbm`/`ootbr`/`ootsl1`/`ootsr3`), missing `ootsr2` entirely. Added `ootsr2_20250702` (chronologically first for that location) for full 6/6 location coverage, matching the "representative sequence subsample" principle more completely. `test_config_sweetcorals.yaml` already lists all 13/13 sequences (the small-dataset exception), left as-is.

Verified: both files parse; `test_config_soneva.yaml` now has 6 sequences (was 5), `test_config_sweetcorals.yaml` unchanged content (13), only formatting changed.

Commit: `00aa003`

### 2026-07-25 — test_exp_soneva.yaml/test_exp_sweetcorals.yaml: switched to eth's max_rgb/step_size

Files: `configs/test_exp_soneva.yaml`, `configs/test_exp_sweetcorals.yaml`.

User asked to make these more similar to `test_exp_eth.yaml`. The one remaining stylistic difference (block naming, `Config`/`NumRuns`/`Module` shape were already identical) was the frame-capping parameter: `rgb_idx: [0,2000]` vs. eth's `max_rgb: 200, step_size: 3`. Switched both files' `Parameters` to `max_rgb: 200, step_size: 3`, matching eth's exact values.

Verified: both files parse correctly with the new Parameters.

Commit: `00aa003`

### 2026-07-25 — Added colmap block to test_exp_soneva.yaml/test_exp_sweetcorals.yaml, fixed the now-stale SKILL.md claim

Files: `configs/test_exp_soneva.yaml`, `configs/test_exp_sweetcorals.yaml`, `.claude/skills/add-dataset/SKILL.md`.

User asked to add a `colmap` block too (mirroring `test_exp_eth.yaml`'s two baselines) and confirmed `test_config_sweetcorals.yaml` still has all 13 sequences (verified: yes, only formatting had changed in the earlier pass, not content).

Added `test_exp_soneva_colmap`/`test_exp_sweetcorals_colmap` blocks to both files (same `Config`/`NumRuns`/`Parameters` as their `droidslam` block, `Module: colmap`) — both datasets are mono-only, so this is two baselines testing the *same* mode, not mode-coverage.

This directly contradicted a claim I'd just written into SKILL.md step 6 ("a single-mode dataset (e.g. soneva, mono-only) needs only the one block") — fixed: reworded to say one block per mode is the *minimum*, but a second baseline for the same mode is also worth adding for baseline diversity, citing soneva/sweetcorals's new two-block shape as the example. Also updated the `max_rgb`/`step_size` citation to include soneva/sweetcorals (now 3 files use it, not just eth).

Verified: both `test_exp_soneva.yaml`/`test_exp_sweetcorals.yaml` parse with two top-level blocks each.

Commit: `00aa003`

### 2026-07-25 — test_config_sweetcorals.yaml trimmed to a real subset, dropped the "small dataset lists everything" exception

Files: `configs/test_config_sweetcorals.yaml`, `.claude/skills/add-dataset/SKILL.md`.

User: "the test config should be a subset of sequences" — a general principle, not a sweetcorals-specific fix. This overrides the exception kept from the original SKILL.md text (small dataset → list every sequence, cited via `test_config_sweetcorals.yaml`'s prior 13/13).

Trimmed `test_config_sweetcorals.yaml` from 13 to 4 sequences: one per site group (`banyuwangi_farm`, `pemuteran_p1`, `tabuhan_p1`, `watudodol_p1`), keeping `tabuhan_p1` specifically since it's the only sequence with real calibration/groundtruth (every other sweetcorals sequence is raw uncalibrated fisheye — losing it from the smoke test would mean nothing in the sample ever exercises the calibrated code path).

Updated SKILL.md step 6 to match: removed the "small dataset, list everything" exception entirely, replaced the `test_config_sweetcorals.yaml` citation (now demonstrating "trim to a subset even when the dataset is small," 4/13, one per site group including the calibrated sequence) as the second worked example alongside `test_config_eth.yaml`.

Verified: `test_config_sweetcorals.yaml` parses with 4 sequences; no other stale reference to the old 13/13 citation remains in SKILL.md.

Commit: `00aa003`

### 2026-07-25 — generate_dataset_table.py: fixed Download Issues detection for soneva/sweetcorals

Files: `Datasets/extra-files/generate_dataset_table.py`.

User-spotted: `dataset_table.md`'s Download Issues column was empty for `soneva`/`sweetcorals`, even though both should show `huggingface_token` (`HFColmapDatasetMixin.get_download_issues`, shared by both via `dataset_soneva.py`).

Root cause, three compounding regex/design issues, all triggered by exactly this multi-inheritance case (confirmed via grep: `dataset_soneva.py`/`dataset_sweetcorals.py` are the *only* dataset files using multi-inheritance anywhere in the repo):
1. `_CLASS_RE` required parens (`\(([\w.]+)\)`), so `class HFColmapDatasetMixin:` (no base class, no parens) wasn't matched as a class at all.
2. `_CLASS_RE`'s base-class group (`[\w.]+`) didn't allow commas, so `class SonevaDataset(HFColmapDatasetMixin, DatasetVSLAMLAB):`/`class SweetcoralsDataset(HFColmapDatasetMixin, DatasetVSLAMLAB):` (multiple bases) weren't matched either — `_class_blocks()` returned an empty dict for both files, so every lookup immediately failed.
3. Even with 1/2 fixed, `SweetcoralsDataset`'s real base with the actual `get_download_issues` implementation (`HFColmapDatasetMixin`) lives in a *different* file (`dataset_soneva.py`) than the class being checked — the script's own docstring says "(single-file) inheritance chain," an explicit scope limit that doesn't cover this case.

Fixed all three: `_CLASS_RE` now makes the base-class group optional and captures the full parenthesized content; `_class_blocks()` splits it into a list of base names instead of a single one; walking the chain now tries every base in order, and a new `_local_imports()` helper parses `from Datasets.dataset_files.X import Y` lines so a base class not found in the current file's blocks is looked up in whichever sibling dataset file actually imports/defines it (following exactly the soneva→sweetcorals mixin-sharing pattern).

Verified: script runs clean, `dataset_table.md` diff is exactly the two expected rows — `soneva`/`sweetcorals` now both show `huggingface_token` in Download Issues, every other row unchanged (including `strayscanner`, another hugging-face+local dataset, confirming the fix didn't touch anything outside the multi-inheritance case).

Commit: `00aa003`

### 2026-07-25 — generate_dataset_table.py: added Features/License columns, reordered README's Datasets table

Files: `Datasets/extra-files/generate_dataset_table.py`, `Datasets/extra-files/dataset_table.md`, `README.md`.

User asked for two things: (1) add `Features`/`License` columns to `dataset_table.md` in a specific order (`Features Label Modes Camera Models License Download Download Issues Maintainer AI-Assisted`), Features sourced from README.md, License from each dataset's own yaml; (2) reorder README's Datasets table to match `dataset_table.md`'s alphabetical order.

**Script changes**: renamed the "Dataset" column to "Label" (matching README's own terminology) and reordered columns per spec. Added `_expand_label()` to handle README rows that cover multiple dataset_table.md labels via a shared prefix (`openloris-d400/t265` → `openloris-d400`+`openloris-t265`; `rover-picam/d435i/t265` → all three `rover-*` labels) — the constituents share everything up to the last `-` before the first `/`, varying only the suffix. Added `_readme_features()`, which parses every row (active *and* commented-out placeholder rows — they already carry curated Features tags, no reason to leave them blank) in the region between the `| Datasets` table header and the next `## ` heading, so the differently-shaped Baselines table above it (no Features column, but a similarly-shaped `[**Name**](url) | ... | \`label\`` row structure) is never touched. License comes straight from `cfg.get("about", {}).get("license", "")`.

**Real bug found in README along the way**: the `monotum` commented-out placeholder row was missing its leading `| ` (`<!-- [**Monocular...` instead of `<!-- | [**Monocular...`), unlike all 8 other commented rows — meant its Features were silently unparseable by any row-shaped regex. Fixed the missing pipe while reordering this block anyway.

**Reordered README's Datasets table**: 17 active rows + 9 commented placeholder rows, both blocks independently sorted alphabetically by Label to match `dataset_table.md`'s `sorted(dataset_files_dir.glob(...))` order. Verified row count unchanged (17+9=26 before and after) and no row content altered beyond reordering + the monotum pipe fix.

**Notable finding surfaced by the new License column, not fixed** (needs domain knowledge, not something to guess): `dataset_ariel.yaml`, `dataset_caves.yaml`, `dataset_drunkards.yaml`, `dataset_hamlyn.yaml`, `dataset_hilti2022.yaml`, `dataset_hilti2026.yaml`, `dataset_madmax.yaml`, `dataset_s3li.yaml`, `dataset_videos.yaml`, `dataset_youtube.yaml` all literally have `license: License` in their `about:` block — an unfilled template placeholder, never actually set to a real license identifier. Confirmed this is genuine yaml content, not a script parsing bug. Worth a follow-up pass to fill in real license values.

Verified: script runs clean under the reordered README, all 34 dataset_table.md rows present, `monotum`'s Features now populate (📸🏠🤳), `soneva`/`sweetcorals`/`openloris-*`/`rover-*` all show the expected Features from their (possibly grouped) README row.

Commit: `00aa003`

### 2026-07-25 — generate_dataset_table.py: added leftmost Datasets column (from yaml, not README)

Files: `Datasets/extra-files/generate_dataset_table.py`, `Datasets/extra-files/dataset_table.md`.

Note: user restored `README.md` to its pre-reorder state ("temporarily") before this pass — the earlier README reorder entry above is now reverted in the working tree; not re-applied here, this pass only touched the script.

User asked for a new leftmost "Datasets" column matching README's `[**Name**](URL)` display format, but sourced from each dataset's own yaml this time (`about.summary` for Name, `about.homepage` for URL) rather than cross-referencing README — the opposite direction from the Features column added earlier this session. Added `dataset_link = f"[**{display_name}**]({homepage})"` (falls back to plain `display_name` text, no link, if `homepage` is missing) as the new leftmost column, ahead of Features.

Three real findings surfaced by actually running this against every dataset yaml, not bugs in the implementation:
1. **18 of 34 dataset yamls have no `about:` block at all** (`7scenes`, `euroc`, `iphone`, `kitti`, `monotum`, `msd`, `nuim`, `openloris-d400`/`openloris-t265`, `replica`, `rgbdtum`, `rover-d435i`/`rover-picam`/`rover-t265`, `scannetplusplus`, `sesoko`, `tartanair`, `ut-coda`) — blank Datasets *and* License cells for all of them, consistent with the License-column finding from the earlier pass. These predate the `about:` block convention (SKILL.md step 3) and were never backfilled.
2. Where `about.summary` exists, it doesn't always read like a short display title the way README's hand-curated Datasets-column names do — `soneva`/`sweetcorals` use their full one-line descriptive sentence as `summary` ("Coral reef time-series photogrammetry (Maldives), from Soneva Conservation and Sustainability Maldives and Wildflow" vs. README's actual "Soneva Corals"), and `eth`'s summary text ("ETH3D SLAM & Stereo Benchmarks") differs in wording from README's ("ETH3D SLAM Benchmarks"). Not a bug — this column is defined to pull from the yaml, and the yaml's `summary` field just isn't always title-length.
3. `youtube` correctly falls back to plain (unlinked) text since its yaml has `summary` but no `homepage` key — confirmed intentional fallback behavior, not a bug.

Verified: script runs clean, 34 rows written, spot-checked `eth`/`soneva`/`sweetcorals`/`kitti`/`iphone`/`youtube` rows against their actual yaml `about:` blocks.

Commit: `00aa003`
