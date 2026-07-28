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

11. **`rgb_0`/`depth_0` hardcoded literals** — filesystem paths for these two folders must be built via `self.rgb_path(sequence_name)`/`self.depth_path(sequence_name)` (`DatasetVSLAMLAB` base-class helpers), never by hardcoding the `'rgb_0'`/`'depth_0'` strings yourself. Doesn't apply to CSV *content* — the `f"rgb_0/{filename}"` relative-path string written into a `rgb.csv`/`groundtruth.csv` row is legitimately a literal, since it's data, not a filesystem `Path` construction.

12. **CSV read/write via `utilities.read_csv_rows`/`write_csv_rows`** — build a `rows` list and call `write_csv_rows(path, header, rows)` instead of hand-rolling `csv.writer` + a manual tmp-file/`.replace()` dance; use `read_csv_rows(path)` instead of hand-rolling `csv.reader` when a hook needs to read back an already-written sequence CSV (e.g. `create_groundtruth_csv` reading `rgb.csv` for timestamps).

13. **`sequence_nicknames` — check whether an override is even needed** — `DatasetVSLAMLAB.__init__` already sets a reasonable default (`utilities.default_sequence_nicknames()`: underscore → space). A subclass override that just reproduces that default is dead code. When something fancier genuinely is needed and the transform matches a substring that itself contains an underscore, build it from `self.sequence_names` (raw), not `self.sequence_nicknames` (already underscore-replaced by the time the default ran) — the underscore your match depends on would already be gone.

14. **`remove_unused_files` path-matching bug** — whatever path gets `unlink()`'d at `MINIMAL` retention must be *exactly* where `download_sequence_data` wrote the file. A recurring copy-paste bug: checking `VSLAMLAB_BENCHMARK / <archive>` when the archive actually lives one level deeper at `self.dataset_path / <archive>`. `unlink(missing_ok=True)` silently swallows the mismatch, so nothing is ever actually deleted and it goes unnoticed.

15. **Shared-archive cleanup scope** — an archive shared across multiple sequences needs different handling depending on scope: whole-dataset shares (the source can't be split into per-sequence downloads at all) belong in an overridden `download_process`, run once after the loop over every sequence — never in `remove_unused_files`, which runs per sequence and could delete a still-needed shared resource mid-loop. A scene/group-scoped share (only some sequences, not the whole dataset) instead belongs in `remove_unused_files` itself: delete only that one sequence's exclusive piece, and it's fine to also delete the shared file itself (even before sibling sequences are downloaded) *if* `download_sequence_data` re-downloads it on demand — verify that fallback actually exists before relying on it.

16. **Final step — check whether the shared docs need updating** — once every per-file item above is done for this pass's datasets, check whether anything the pass surfaced should be encoded into `Datasets/extra-files/generate_dataset_table.py`, `dataset_template.py`/`.yaml`, `.claude/skills/add-dataset/SKILL.md`, or `CLAUDE.md` — a new download pattern, a closed-list semantic that wasn't documented, a bug in the table generator, etc. This happened after the #78 pass, the #79 pass, and the squidle/sesoko/eiffel_tower pass (#84/#85), but only because it was asked each time, not because this checklist told the next session to — codified here so it happens by default. Distinct from item 10: that one is an ongoing self-containment property to check per item as you go; this is a deliberate final sweep after the per-file work is done.

17. *(next checks TBD as they come up, e.g. field-value validation)*

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

### 2026-07-26 — replica / rgbdtum / tartanair / nuim / 7scenes: full checklist pass

Files: `dataset_replica.py`/`.yaml`, `dataset_rgbdtum.py`/`.yaml`, `dataset_tartanair.py`/`.yaml`, `dataset_nuim.py`/`.yaml`, `dataset_7scenes.py`/`.yaml` (plus one-off exceptions to `dataset_eth.py` and `Datasets/DatasetVSLAMLAB.py`, see below).

Unlike the eth/soneva/sweetcorals passes above, this pass's findings were tracked as a checklist in [GitHub issue #78](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/78) (21 items, all closed) rather than incrementally in this log — these entries backfill the log after the fact, once asked whether the template/SKILL.md/log needed updating for what this pass turned up. See #78 for the itemized list; this and the following entries summarize the substance.

Ran the full checklist (items 1–10 as they stood at the time) in order, one commit per stage:

- **Header + `about:`/`vslamlab_maintainer:` pass** (`cfe6ebd`) — none of the five had a module header docstring or yaml `about:` block at all (unlike eth, which predates the convention but had already been backfilled). Added headers with `Created` sourced from each file's actual first commit (`git log`, not guessed) and `Updated: 2026-07-26`; `about:` fields (license/summary/homepage/authors) verified via web search rather than trusted from memory. `replica` needed a judgment call: the data it actually downloads (`nice-slam`'s rehosted `Replica.zip`) is really the iMAP paper's rendered RGB-D trajectories over Replica's original 3D scenes, not the raw Replica dataset itself — credited both projects' authors rather than picking one.
- **Import grouping pass** (`d352c4b`) — all five had ungrouped/non-alphabetical imports; also fixed `replica`'s combined `import os, shutil` and `tartanair`'s stray leading blank line before `from __future__ import annotations`. Dropped two genuinely-dead imports found along the way (`rgbdtum`'s unused `Iterable`, `tartanair`'s unused `urljoin`).
- **`self.cfg` / class docstring / `sequence_nicknames` pass** (`5066e97`) — all five `__init__`s still reopened `self.yaml_file` instead of using `self.cfg` (checklist item 4); switched to `self.cfg` and dropped the now-unused `import yaml` from each. Class docstrings used the raw slug/class spelling (`REPLICA`, `TARTANAIR`, `NUIM`, `7scenes`) instead of a brand name — fixed to `Replica`/`TartanAir`/`ICL-NUIM`/`7-Scenes` (`rgbdtum`'s `"TUM RGB-D"` was already correct). Removed `tartanair`'s no-op `self.sequence_nicknames = self.sequence_names` override (the base class's default already produces the same thing, since tartanair's names have no underscores) — the trigger for new checklist item 13.
- **`download_sequence_data` pass** (`664d4a5`) — found `tartanair`'s MINIMAL-retention cleanup unlinking `VSLAMLAB_BENCHMARK / <archive>` when the archive was actually at `self.dataset_path / <archive>` (new checklist item 14) — fixed, dropped the now-unused `VSLAMLAB_BENCHMARK` import. Collapsed `nuim`'s/`7scenes`' redundant `decompressed_folder`/`sequence_path` duplicate variables.
- **Base-class `depth_path()` + `create_rgb_folder` pass** (`fd79114`) — user-spotted: the base class had `rgb_path()` but no `depth_path()` helper, so every rgbd dataset hardcoded `'depth_0'` itself (and `rgbdtum`/`7scenes`/even `eth` also bypassed the existing `rgb_path()` helper for `'rgb_0'`, using string-literal tuples instead) — new checklist item 11. Added `DatasetVSLAMLAB.depth_path()` mirroring `rgb_path()`; switched `replica`/`nuim`/`rgbdtum`/`7scenes` to use both helpers. `eth` is normally left read-only as this pass's reference dataset, but the user granted a one-off exception to fix the identical pattern there too, later extended to `eth`'s `create_rgb_csv` as well (see below). Also dropped `tartanair`'s now-dead `sequence_path` in `create_rgb_folder`, and simplified `7scenes`' copy+`os.remove` into a single `shutil.move` (removing a redundant duplicate `glob.glob()` call in the process).
- **`create_rgb_csv` pass + rgbdtum's RGB/depth sync** (`d97ffb1`) — migrated all five to `utilities.write_csv_rows` instead of hand-rolled `csv.writer`/manual tmp-file boilerplate (new checklist item 12); `replica` previously had no atomic write at all. Along the way: `rgbdtum` turned out to be the first dataset in the repo where RGB and depth come from independently-timestamped streams (TUM's Kinect) rather than a single hardware-synchronized capture, needing a `pandas.merge_asof` nearest-timestamp match instead of the sort-and-zip-by-index every other rgbd dataset here uses — documented this distinction in `dataset_rgbdtum.py`'s docstring, `dataset_template.py`'s `create_rgb_csv` comment, and SKILL.md step 4d, so future additions know to check which case applies.
- **`create_calibration_yaml` pass** (`0e9cfce`) — reformatted all five calibration dict literals from a compact multi-key-per-line style to eth's fully-expanded one-key-per-line style (trailing comma), for consistency. `rgbdtum`'s freiburg1/freiburg2 distortion branch had non-4-space-multiple indentation and used `camera == "freiburg1" or camera == "freiburg2"` instead of `camera in (...)` (already the style used two methods away in the same file) — fixed both.
- **`create_groundtruth_csv` pass** (`9731cb8`) — found and fixed the same "no file written at all" bug as the earlier sweetcorals fix: `rgbdtum`'s validation sequences (no public groundtruth) hit an early `return` before writing anything; now writes a header-only `groundtruth.csv` instead. Also removed a dead `tmp.unlink(missing_ok=True)` call sitting right after `tmp.replace(...)` (the file no longer exists at that point — a no-op left over from a different pattern). `7scenes`' `os.remove(gt0)` ran unconditionally regardless of `BENCHMARK_RETENTION`; gated it behind `!= Retention.FULL`. All five migrated to `write_csv_rows`; `replica` also switched from a hand-rolled `csv.reader` loop to `utilities.read_csv_rows` for reading back `rgb.csv`.
- **`remove_unused_files` pass** (`877f666`) — found the *same* wrong-path bug as `tartanair`'s (checklist item 14) independently in `nuim` too — same fix. `tartanair`'s `download_process` unconditionally `shutil.rmtree`'d its two extracted-archive folders regardless of retention (even at `FULL`, which should delete nothing) — gated behind `!= Retention.FULL`. `7scenes` had no `remove_unused_files` at all and no cleanup of either of its two archive levels (a scene-level zip shared by several sequences, e.g. `chess.zip` shared by all `chess_seq-*`, plus each sequence's own sub-zip inside it) — designing this required working out the shared-archive-scope distinction that became checklist item 15: deleting a *scene-level* shared zip inside one sequence's `remove_unused_files` is safe (verified `download_sequence_data` re-downloads it on demand), but a whole-dataset override like `tartanair`'s would have broken 7-Scenes' per-sequence download semantics (unlike tartanair/replica, 7-Scenes has no `"complete_dataset"` issue — single-sequence downloads must keep working standalone). Added `remove_unused_files` accordingly, including removing the scene folder once it's empty.
- **YAML field-order + trailing-whitespace pass** (`d9c8e18`) — all five put `url_download_root` right after `rgb_hz`, before `modes`/`cam_models`; the template/eth/soneva/sweetcorals order it after `modes`/`cam_models` instead — reordered all five to match. Stripped trailing whitespace from `rgbdtum.yaml` (14 lines, more than the one example originally spotted) and `7scenes.yaml` (3 lines); `nuim`/`replica`/`tartanair` were already clean (fixed incidentally by the earlier `about:`-block edits).
- **`_find_sequence_group` fix** (`22e7156`) — the last open item: `7scenes.py` had this as a stray module-level function with non-4-space-multiple indentation and no error handling (silently returned `None` on no match, which would crash downstream with a confusing `TypeError`). Moved onto the class as a `@staticmethod`, matching `rgbdtum`'s `_nickname`/`_camera_from_sequence` precedent exactly, including an explicit `ValueError` on no match.
- **Final full re-read + trailing whitespace/blank-line pass** — a last read-through of all five files (prompted by asking "is there anything else") turned up scattered trailing whitespace across all five `.py` files (not yet touched — separate from the yaml whitespace above) and a stray blank line right after `def download_sequence_data(...):` in `replica.py`/`tartanair.py` (the same pattern already fixed in `nuim`/`7scenes`' `create_calibration_yaml` earlier, just missed there since it wasn't the method under review). No new bugs, no leftover debug code. Commit: `e0ea1b2`.

Verified throughout: every stage confirmed via `ast.parse` + `pixi run -e vslamlab python` instantiation for all five classes, plus synthetic round-trip tests (temp directories with fabricated source files) for every hook that touches on-disk data — `create_rgb_csv`/`create_groundtruth_csv`/`create_calibration_yaml` output checked byte-for-byte against hand-computed expected values, `remove_unused_files`/`download_process` retention behavior checked at all three `Retention` tiers where applicable (`tartanair`'s `FULL`/`STANDARD`/`MINIMAL`, `7scenes`' `FULL`/`STANDARD`, `7scenes`' shared-vs-sibling-sequence cleanup ordering).

### 2026-07-26 — Template/SKILL.md updated to encode this session's findings

Files: `Datasets/extra-files/dataset_template.py`, `.claude/skills/add-dataset/SKILL.md`, this log.

Once the pass above was done, asked whether anything should be added to the template/SKILL.md/log given what it turned up — four gaps confirmed, all backed by findings above rather than speculative:

- `dataset_template.py`'s `create_rgb_folder` comment didn't mention `self.rgb_path()`/`self.depth_path()` at all — added, matching the `depth_path()` base-class addition above. Also updated SKILL.md step 4c to match.
- `dataset_template.py`'s `__init__` "Sequence nicknames" comment didn't warn against a no-op override, nor explain the underscore-ordering constraint when something fancier is needed — added both, citing tartanair (no-op) and rgbdtum/7scenes (ordering) by name. Also updated SKILL.md step 4a.
- `dataset_template.py`'s `remove_unused_files` comment had no warning about the wrong-path bug pattern (hit independently in both tartanair and nuim this session) — added. Also updated SKILL.md step 4g.
- `dataset_template.py`'s `remove_unused_files` comment had no guidance on cleaning up an archive shared across multiple sequences — added the whole-dataset-vs-scene-group distinction worked out for 7-Scenes above. Also updated SKILL.md step 4g.

`CLAUDE.md` and `dataset_template.yaml` were checked too — no gaps traced back to either (CLAUDE.md is intentionally high-level and already delegates this level of detail to SKILL.md; the yaml template's field order/shape was already correct, this pass's yaml fix was bringing five *existing* files into line with it, not a template gap).

Commit: `e0ea1b2`

### 2026-07-26 — google-drive vs. website: split into two yaml fields, not one host-sniffed URL

Files: `dataset_drunkards.py`/`.yaml`, `dataset_hilti2026.py`/`.yaml` (out of this session's `#78` scope, touched here by explicit request), `dataset_tartanair.py`/`.yaml` (unchanged), `Datasets/extra-files/generate_dataset_table.py`, `Datasets/extra-files/dataset_template.py`/`.yaml`, `.claude/skills/add-dataset/SKILL.md`.

User-spotted, prompted by asking about tartanair's `google-drive` label in `dataset_table.md`: tartanair's `download_sequence_data` doesn't use `gdown` at all — it downloads its yaml's `url_download_root` (a *pre-resolved* `drive.usercontent.google.com/download?...&confirm=t&...` direct-download link) via plain `utilities.downloadFile`, mechanically identical to a `website` dataset. It was only labeled `google-drive` because `generate_dataset_table.py`'s `_download_labels()` classified purely by sniffing the URL's host (`_is_google_drive_url()`) rather than by which yaml field was used or whether the dataset's `.py` actually needs `gdown`.

Checked scope first: exactly 3 datasets showed `google-drive` in `dataset_table.md` — `tartanair` (the false positive above) and `drunkards`/`hilti2026` (both genuine — real `drive.google.com/drive/folders` share links, both `.py` files `import gdown` and call `gdown.download_folder`).

Decision: split the single `url_download_root`-plus-host-sniffing scheme into two distinct yaml fields, so the label is driven by field name, not URL content:
- `url_download_root` — a website-style URL, full stop, regardless of what host it happens to be on. Covers tartanair's pre-resolved Drive link.
- `google_drive_link` (new) — a real Drive share link that needs `gdown` to get past Drive's virus-scan interstitial for anything non-trivially sized.

Renamed `drunkards`/`hilti2026`'s field from `url_download_root` to `google_drive_link` (both `.yaml` and the corresponding `self.url_download_root` → `self.google_drive_link` in each `.py`) — tartanair itself needed no change at all, it just stopped being misclassified once the classification logic no longer sniffed URL hosts. Rewrote `_download_labels()` in `generate_dataset_table.py` accordingly and deleted `_is_google_drive_url()`/the now-unused `urlparse` import entirely — no host inspection left anywhere in the script.

Updated the google-drive pattern's documentation in three places to match (`dataset_template.yaml`'s field-choice comment, `dataset_template.py`'s `__init__`/`download_sequence_data` comments, SKILL.md step 3's yaml-field guidance) — each now explains both fields and the don't-confuse-them rule ("if the URL works with a plain HTTP GET, it's `website`, not `google-drive`").

Verified: regenerated `dataset_table.md` — diff is exactly the expected one-line change (`tartanair`'s Download column `google-drive` → `website`), `drunkards`/`hilti2026` unchanged (still `google-drive`), every other row untouched. All three `.py` files parse and instantiate correctly under `pixi run -e vslamlab python` with `self.google_drive_link`/`self.url_download_root` populated as expected.

Commit: `e0ea1b2`

### 2026-07-26 — euroc / kitti / madmax: full checklist pass

Files: `dataset_euroc.py`/`.yaml`, `dataset_kitti.py`/`.yaml`, `dataset_madmax.py`/`.yaml` (plus `dataset_template.py`/`.yaml` and `.claude/skills/add-dataset/SKILL.md`, see the doc-update entry below).

Same shape as the replica/rgbdtum/tartanair/nuim/7scenes pass above: findings tracked as a checklist in [GitHub issue #79](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/79), worked one checklist item/function at a time with findings presented before each fix, one commit per stage. This entry backfills the log.

- **Header + `about:`/`vslamlab_maintainer:` pass** (`cf6f14f`) — euroc/kitti had no module header docstring and no yaml `about:`/`vslamlab_maintainer:` block at all; madmax had a yaml `about:` block but no `.py` header and no `assisted_by`. Added headers with `Created` sourced from `git log --diff-filter=A --follow` (2024-07-13 for euroc/kitti, matching madmax's existing yaml `date: 2026-03-07`); `about:` fields verified via web search (EuRoC: no explicit license found on ASL's page or the ETH Research Collection record, which 500'd on every fetch attempt — written descriptively per user's call rather than guessing an SPDX tag; KITTI: confirmed CC BY-NC-SA 3.0 directly from cvlibs.net; madmax: license rewritten from the unfilled `License` placeholder to a descriptive value, author list kept collapsed to `Others` per user's call rather than expanding to the paper's full 12-author byline).
- **Import grouping pass** (same commit) — all three had ungrouped/unordered stdlib+third-party imports; kitti also had a stray leading blank line before `from __future__ import annotations` (same bug class fixed in tartanair during #78) and a double-space typo (`from typing import  Any`); madmax was missing `from __future__ import annotations` entirely despite otherwise-correct grouping.
- **`self.cfg` / class docstring pass** (same commit) — all three `__init__`s still reopened `self.yaml_file` instead of using `self.cfg`; switched, and dropped kitti's now-unused `import yaml` (euroc/madmax still need it for calibration parsing). Class docstrings: euroc's `"EUROC"` was a genuine misspelling (the real name is "EuRoC", confirmed against the ASL homepage and the paper title) — fixed; kitti/madmax's all-caps forms are their real acronym-style names (unlike replica/tartanair/nuim's slug artifacts from #78), so those got optional disambiguating context instead (`"KITTI Odometry"`, `"MADMAX Mars rover navigation"`). kitti/madmax's `self.sequence_nicknames = self.sequence_names` overrides were the same no-op pattern as tartanair's (`'00'..'21'`/`'A-0'` etc. have no underscores to swap) — removed.
- **`url_download_root` redesign for euroc** (same commit) — user-spotted: euroc's `url_download_sequences` yaml field had only 3 distinct URLs duplicated across 11 sequence keys (one per `MH_`/`V1_`/`V2_` group, repeated per sequence in that group), not "each sequence has its own unrelated URL" as that field's documented purpose states — a duplicate-source-of-truth risk against the `.py`'s own `_GROUP_INFO` dict (already keyed by the same prefixes, for `subfolder`/`file_size`). Redesigned: renamed back to the canonical `url_download_root` field, restructured as a dict keyed by group prefix (3 entries) instead of full sequence name (11 entries); `_subfolder_for` renamed to `_group_for` (now also returns the matched prefix so the URL can be looked up from `self.url_download_root[prefix]`). This is a genuinely new sub-shape for `url_download_root` not previously documented — see the doc-update entry below.
- **`create_rgb_folder`/`create_rgb_csv` pass** (`7260262`) — euroc/kitti both hardcoded the `rgb_0` folder path instead of `self.rgb_path(sequence_name)` (item 11); fixed (`rgb_1` stays a literal in both — no base-class helper exists for a second-camera folder anywhere in the repo). All three migrated `create_rgb_csv` from hand-rolled `csv.writer`/`pandas.to_csv` to `utilities.write_csv_rows`; madmax's `create_rgb_csv` also switched its `rgb_0` lookup to `self.rgb_path()` for the same reason. kitti's dead `sequence_path` local removed. Flagged but left alone (user's explicit call): madmax's `download_sequence_data` hardcodes `"rgb_0"`/`"rgb_1"`/`"depth_0"` as a folder-name list — not a live bug today (the literal happens to equal what the helpers would return), a single-source-of-truth risk if the naming convention ever moved into a `path_constants.py` constant.
- **`create_imu_csv`/`create_calibration_yaml` pass** (`c0561e8`) — euroc/madmax's `create_imu_csv` migrated to `write_csv_rows`, using `df[cols].astype(object).values.tolist()` (not a plain `.values.tolist()`) so pandas doesn't upcast the mixed int64 timestamp + float64 rate columns into a single dtype, which would corrupt the nanosecond timestamp's precision — verified byte-identical to the pre-refactor `pandas.to_csv` baseline. **Bug found and fixed**: kitti's `create_calibration_yaml` read `fx`/`fy`/`cx`/`cy` via `file.readline().split()` and never cast to `float()`, so `calibration.yaml`'s `focal_length`/`principal_point` were written as quoted strings (`['718.856000', ...]`) instead of numbers — confirmed by actually round-tripping through `yaml.safe_load` before and after the fix. Also removed euroc's dead `imu0/sensor.yaml` read (parsed but immediately overwritten by hardcoded IMU-noise constants, never consulted) and reformatted euroc/kitti's calibration dict literals to the one-key-per-line style `dataset_eth.py`/madmax already use.
- **`create_groundtruth_csv` pass** (`261ffc3`) — **two more "no file written at all" bugs**, same class already fixed for sweetcorals/rgbdtum in earlier passes: euroc returned early with no `groundtruth.csv` at all when TUM's supplemental groundtruth file was missing; kitti did the same for sequences `11`-`21` (KITTI's held-out benchmark test set, poses never published — a genuine per-sequence gap, not just "not downloaded yet"). Both fixed to write a header-only file instead. Migrated both plus madmax to `write_csv_rows`; dropped kitti's dead `sequence_path` local and the now-fully-unused `import csv` from both euroc.py and kitti.py (every CSV-writing method in both files had by then migrated to `write_csv_rows`).
- **`remove_unused_files`/`download_process` pass** (`7a71dc8`) — three retention-tier bugs, verified across all three `Retention` tiers with a synthetic per-tier check after fixing: euroc's `mav0`/`__MACOSX` delete had no `BENCHMARK_RETENTION` gate at all, running even at `Retention.FULL`; kitti's `download_process` unlinked `VSLAMLAB_BENCHMARK / <archive>` (same wrong-path bug class as tartanair/nuim in #78) when `download_sequence_data` actually wrote everything to `self.dataset_path / <archive>` — confirmed directly against `dataset_tartanair.py`'s already-correct model — compounded by calling `.unlink()` on `"dataset"` (a directory, not a file); madmax's STANDARD/MINIMAL tiers were reversed (`*.zip` archives deleted at STANDARD instead of MINIMAL-only; `calibration`/`groundtruth` folders gated at MINIMAL instead of STANDARD), plus `imu_raw.csv` added to MINIMAL after determining it has no separate zip backing it (unlike `rgb_0`/`rgb_1`/`depth_0`), making it play the same role the zips play for those modalities.
- **YAML formatting pass** (`42aad15`) — kitti's `url_download_root`/`url_download_root_gt` still preceded `modes`/`cam_models` (the exact ordering bug fixed repo-wide for replica/rgbdtum/tartanair/nuim/7scenes in #78, missed for kitti at the time since it wasn't in that pass's scope); reordered. Stripped trailing whitespace from 2 `sequence_names` entries in kitti.yaml, an extra trailing blank line in euroc.yaml, and a missing comma-space in madmax's `modes` list.
- **`get_download_issues` pass** — reviewed, no changes needed for any of the three: euroc/kitti already match `dataset_replica.py`'s model verbatim; madmax is the literal reference model SKILL.md cites for the `api_token` pattern.
- **Final full re-read + trailing whitespace/blank-line pass** (`c4cd54f`) — same closing pattern as #78: stripped scattered trailing whitespace (5 lines in euroc.py, 2 in kitti.py) and a stray blank line right after kitti's `def download_sequence_data(...):` (same pattern fixed in replica.py/tartanair.py during #78). Also found and fixed a latent bug outside the 15-item checklist: madmax's `_get_file_url` chained six independent `if`s with no `elif`/`else` to build `ids` — an unrecognized `sequence_name` would leave `ids` unbound and raise a confusing `UnboundLocalError` at the `return` line. Replaced with a `_SEQUENCE_FILE_IDS` class dict (placed alongside `__init__`, matching `_GROUP_INFO`'s placement convention) + direct lookup, now raising a clear `KeyError` naming the bad sequence; verified byte-identical URLs for every real sequence name.

Verified throughout: every stage confirmed via `ast.parse`/`yaml.safe_load` + `pixi run -e vslamlab python` instantiation for all three classes, plus synthetic round-trip tests (temp directories with fabricated source files) for every hook that touches on-disk data — output checked byte-for-byte or numerically against hand-computed/pre-refactor-baseline values, `remove_unused_files`/`download_process` retention behavior checked at all three `Retention` tiers for all three datasets.

Commits: `cf6f14f`, `7260262`, `c0561e8`, `261ffc3`, `7a71dc8`, `42aad15`, `c4cd54f`

### 2026-07-26 — Template/SKILL.md updated to encode the euroc/kitti/madmax pass's findings

Files: `Datasets/extra-files/dataset_template.py`, `Datasets/extra-files/dataset_template.yaml`, `.claude/skills/add-dataset/SKILL.md`, this log.

Asked (same as after the #78 pass) whether anything from the pass above should be added to the template/SKILL.md/`utilities.py`/`CLAUDE.md` — three real gaps confirmed, all backed by bugs actually hit above rather than speculative:

- `create_calibration_yaml`'s template comment had no warning about type-casting raw-parsed values — kitti's bug (see above) is exactly this: `write_calibration_yaml`/`_get_rgb_yaml_section` (`Datasets/DatasetVSLAMLAB_calibration.py`, out of this pass's file scope) just f-string-embeds whatever type it's given, so an uncast string list gets Python-repr'd into `calibration.yaml` as quoted strings, with no error anywhere. Added a warning to both `dataset_template.py`'s `create_calibration_yaml` comment and SKILL.md step 4e.
- `remove_unused_files`'s existing "Common bug" paragraph (from #78) only covered the wrong-path/`unlink(missing_ok=True)` mistake. Added two more variants hit this session: an ungated STANDARD-tier delete (no `BENCHMARK_RETENTION` check at all, so it deletes even at `Retention.FULL`) and calling `.unlink()` on a path that's actually a directory (`IsADirectoryError` — needs `shutil.rmtree`). Added to both `dataset_template.py`'s comment and SKILL.md step 4g.
- `url_download_root` was only documented as a single string or a dict keyed by full `sequence_name` (`url_download_sequences`). euroc's redesign introduced a third shape — dict keyed by group prefix — not previously documented anywhere. Added to `dataset_template.yaml`'s field-choice comment and SKILL.md step 3, citing `dataset_euroc.yaml` as the model.

`CLAUDE.md` checked, no gap found (same conclusion as after #78 — it stays intentionally high-level). `utilities.py` checked too: the `df[cols].astype(object).values.tolist()` pattern used 3x this session (to stop pandas from upcasting mixed int64/float64 columns and corrupting the ns timestamp) is a real repeated pattern and a plausible future helper, but was explicitly not added — user's call to keep this pass scoped to euroc/kitti/madmax plus doc updates, not a new shared utility.

Commit: `c4cd54f` (bundled with the final re-read pass above — both landed in one commit)

### 2026-07-26 — Post-pass sanity check: regenerated `dataset_table.md`

Files: `Datasets/extra-files/dataset_table.md` (checked, not modified — see below).

Per checklist item 10's "regenerate before trusting `dataset_table.md`-derived facts" rule, re-ran `python3 Datasets/extra-files/generate_dataset_table.py` after the euroc/kitti/madmax pass (and its template/SKILL.md follow-up) to confirm the script still runs clean against the changed files and the table is accurate.

Result: exit code 0, "Wrote 34 dataset rows", **zero diff** against the already-committed `dataset_table.md` — it had already been kept in sync via the regenerations run mid-pass for verification (see the `url_download_root` redesign entry above), so nothing was stale. Specifically confirmed `euroc`'s row still correctly shows `website` in the Download column: `_download_labels()`'s `any([cfg.get("url_download_root"), cfg.get("url_download_sequences")])` check is satisfied by truthiness, so euroc's new dict-keyed-by-group-prefix shape for `url_download_root` (3 entries, not a plain string) doesn't trip up the script — a non-empty dict is still truthy. `kitti`/`madmax` rows also spot-checked (modes, cam_models, license, download issues, maintainer, AI-assisted all correct).

No commit — no file changed.

### 2026-07-28 — openloris / ut-coda: full checklist pass

Files: `dataset_openloris.py` (`OpenlorisDataset` shared base + both registered children `OpenlorisD400Dataset`/`OpenlorisT265Dataset` — unlike the squidle/sesoko/eiffel_tower pass, no unregistered downstream child to exclude here), `dataset_openloris-d400.yaml`, `dataset_openloris-t265.yaml`, `dataset_ut_coda.py` (`UtCodaDataset`), `dataset_ut-coda.yaml`. Findings tracked upfront as a checklist in [GitHub issue #88](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/88) (mirroring #78/#79/#85's shape), then worked one checklist item/function at a time in conversation, with findings presented before each fix — this entry backfills the log to match.

**Note:** the three openloris files already had an uncommitted diff in the working tree at the start (`repo_id` → `hf_repo_id`, matching the `hf_repo_id` convention used by soneva/sweetcorals) — confirmed with the user this was prior work to build on, not something to set aside, so this pass's findings reflect the current (uncommitted) state rather than a fresh baseline. `dataset_ut_coda.py`/`dataset_ut-coda.yaml` had no pending changes.

- **Header consistency (item 2)** — neither `.py` file had a module header docstring, and none of the 3 yamls had `about:`/`vslamlab_maintainer:` blocks at all (both predate the convention). Added both, with two judgment calls resolved with the user rather than guessed: maintainer identity (Alejandro Fontan / Claude (Sonnet 5) / `alejandro.fontan@qut.edu.au`, matching every other dataset's block exactly) and `dataset_ut_coda.py`'s `Created` date — its git history is unusually split (`dataset_ut-coda.yaml` first added 2025-05-25, but `dataset_ut_coda.py` under its current name/content only appears 2025-12-30, a reversed version of the usual predates-the-convention gap) — resolved as 2025-05-25 (treats the `.py` as a continuation of the same integration, not a new one). `about:` license/summary/authors were verified via live web search (OpenLORIS-Scene: CC BY-ND 4.0 per its actual Hugging Face dataset card, not the code repo's separate MIT license; UT CODA: CC BY-NC-SA 4.0 per its own homepage), not guessed. Long author lists trimmed to lead-author(s)-plus-`Others`, matching `dataset_eth.yaml`'s precedent (user's explicit choice over madmax's plain `Others`-only style or listing everyone).
- **Import grouping (item 1)** — both files had their whole import block as one undifferentiated blob (no blank line splitting stdlib from third-party) with several misordered names within it; regrouped and reordered both to match the repo's established 3-group/alphabetical convention (verified via grep across ~20 sibling files, e.g. `BENCHMARK_RETENTION, Retention` and `decompressFile, downloadFile` name ordering).
- **Class docstrings** (not itself a numbered item, but the same casing issue flagged for replica/tartanair/nuim in #78 and sesoko in #85) — `"""OPENLORIS..."""`/`"""OPENLORIS_d400..."""`/`"""OPENLORIS_t265..."""`/`"""UT_CODA..."""` all used raw slug/all-caps-underscore spelling. Fixed to `"""OpenLORIS-Scene..."""` (+ `D400`/`T265` suffix per subclass, matching `dataset_rover.py`'s exact base+sensor-suffix precedent) and `"""CODa..."""` (the dataset's own short/citable name, matching the `dataset_eth.py`→`"""ETH3D..."""` precedent of using the compact brand name over the longer descriptive title) — both confirmed with the user before renaming, since exact display wording is a judgment call.
- **`__init__` contents (item 3)** — `OpenlorisDataset.__init__`'s `self.sequence_nicknames = self.sequence_names` was a no-op override (openloris names have no underscores to swap) — same dead pattern as tartanair's in #78, removed. `UtCodaDataset`'s `[f"seq{s}" for s in self.sequence_names]` override is genuine, correctly built from raw `self.sequence_names` — no violation. Signatures already correct on all four classes.
- **Redundant yaml reopen (item 4)** — all three `__init__`s (`OpenlorisDataset`, `OpenlorisD400Dataset`, `UtCodaDataset`) still reopened `self.yaml_file` instead of reading `self.cfg`; switched all three. `import yaml` dropped from `dataset_openloris.py` (became fully unused); kept in `dataset_ut_coda.py` (`create_calibration_yaml` separately parses each sequence's own calib yaml file).
- **Unused `sequence_path` locals (item 5)** — 4 dead instances found and removed in `dataset_openloris.py` (`create_imu_csv`, `create_groundtruth_csv`, both subclasses' `create_rgb_csv`); `dataset_ut_coda.py` was already clean here (its `download_sequence_data` had a *different* bug instead — see below).
- **`remove_unused_files`/Retention tiers (item 7)** — both were under-implemented, and neither was fixed without checking with the user first since neither could be verified against real downloaded data (no `OPENLORIS`/`UT-CODA` folders exist locally in this environment). `OpenlorisDataset.remove_unused_files` was a complete no-op; implemented STANDARD-tier deletion of the reformatted-away `groundtruth.txt`/`{camera}_accelerometer.txt`/`{camera}_gyroscope.txt`/`sensors.yaml`/`trans_matrix.yaml`, plus MINIMAL-tier deletion of the per-sequence `.7z` (safe per the group-scoped-share fallback verified under item 15 below; a no-op specifically for `corridor1-1`, whose raw data never produces that file on disk in the first place — noted in a code comment, not treated as a bug). `UtCodaDataset.remove_unused_files` only ever ran at MINIMAL; moved `calibrations/`/`timestamps/`/`poses/` to STANDARD (pure reformats, per convention) and added MINIMAL-tier deletion of the downloaded `.zip` (previously never cleaned at any tier) — left `metadata/` at MINIMAL-only since nothing in the file ever reads it, so its role couldn't be confirmed.
- **`get_download_issues` (item 8)** — `dataset_ut_coda.py` correctly has no override (plain public HTTPS download, no token/license constraint). `dataset_openloris.py` had no override either, despite being a genuine Hugging Face Hub download (`hf_hub_download`) — but the *user* flagged this as suspicious given `dataset_table.md` classifies it `hugging-face`. Investigated against `dataset_soneva.py`'s own two HF-fetch shapes: `ensure_hf_sequence_download()` for a whole directory of ready files vs. plain `hf_hub_download(..., token=hf_token())` for one specific named file (`_fetch_colmap_file`) — openloris's per-sequence compressed-archive fetch matches the *second* shape, so the download mechanism itself was already right, just missing `token=hf_token()` and a `get_download_issues` override. Added both, matching `HFColmapDatasetMixin.get_download_issues` exactly; verified live (both the "token present" and "token absent" branches) under `pixi run -e vslamlab python`.
- **YAML formatting (item 9)** — `sequence_names` indentation (2-space) and trailing newlines were already fine in all 3 files by the time this item was reached (the newline gap closed incidentally once the `about:`/`vslamlab_maintainer:` blocks were appended). Real bug found: `dataset_ut-coda.yaml`'s `sequence_names` had `'6 '` (trailing space inside the quotes) — the only one of 23 entries not a clean numeric string, feeding directly into `UtCodaDataset.sequence_nicknames` (`"seq6 "`) and the download URL/path built from that raw sequence name. Fixed to `'6'`; verified `sequence_nicknames` now correctly reads `"seq6"`.
- **`rgb_0`/`depth_0` hardcoded literals (item 11)** — found and fixed in `create_rgb_folder` (both `OpenlorisD400Dataset`/`OpenlorisT265Dataset`'s `new_folders` string lists, `UtCodaDataset`'s `rgb_path_0`) and in `create_rgb_csv` (`UtCodaDataset`'s `rgb_path_0` again) — all switched to `self.rgb_path()`/`self.depth_path()`. `rgb_1` stays a literal everywhere (no base-class helper for it, matching the euroc/kitti precedent).
- **CSV read/write via `write_csv_rows` (item 12)** — `create_rgb_csv` (all three subclasses), `create_imu_csv`, and `create_groundtruth_csv` (both files) all migrated off plain `df.to_csv(...)`/hand-rolled `csv.writer` onto `write_csv_rows`. Two of these (`create_imu_csv`, `create_groundtruth_csv`, both mixing an int64 ns-timestamp column with float64 sensor/pose columns) hit the same upcast risk already fixed for euroc/madmax in #79 — verified with a standalone before/after test that a plain `.values.tolist()` corrupts a realistic ns timestamp (`1.7537...e+18`, wrong past 2^53) while `df[cols].astype(object).values.tolist()` preserves it exactly. `dataset_ut_coda.py`'s `create_groundtruth_csv` also dropped an unused `idx` from `enumerate(source_file, start=0)`, and its `import csv` (now fully unused after the migration).
- **Duplicate-variable bug (not itself a numbered item)** — `UtCodaDataset.download_sequence_data`'s `decompressed_folder`/`decompressed_name` were an exact duplicate of `sequence_path`/`sequence_name` under different names — same bug class already fixed for nuim/7scenes in #78. Collapsed to just `sequence_path`.
- **Calibration dict formatting** — reformatted all four calibration dicts (D400's `rgbd0`/`imu`, T265's `rgb0`/`rgb1`/`imu`, ut-coda's `rgb`) from a compact multi-key-per-line style to `dataset_eth.py`'s one-key-per-line-with-trailing-comma style, matching the precedent set for replica/rgbdtum/tartanair/nuim/7scenes.
- **Shared-archive cleanup scope (item 15)** — `dataset_openloris.py`'s `dataset_path_raw` archives are a scene/group-scoped share (several sequences per group tar, e.g. 4 `corridor1-*` sequences per tar), which per this item belongs in `remove_unused_files` itself. Verified the required on-demand-redownload fallback already exists in `download_sequence_data` before relying on it for the item-7 fix above. `dataset_ut_coda.py` has no shared-archive scenario (one `.zip` per sequence) — not applicable.
- **Trailing whitespace / stray formatting sweep** — both files had scattered trailing whitespace throughout, plus a few real formatting bugs of the same classes fixed in earlier passes: a stray blank line right after `def _get_imu_noise_parameters(...):` (openloris.py, same pattern as nuim/7scenes' `create_calibration_yaml` in #78), a missing blank line before `class OpenlorisD400Dataset`, and a doubled/whitespace-only blank line in `UtCodaDataset.__init__`. All swept in one pass once the yaml-reopen fix (item 4) was also done.

Verified throughout: `ast.parse` after every edit, plus `pixi run -e vslamlab python` instantiation of all three classes (`OpenlorisD400Dataset`/`OpenlorisT265Dataset`/`UtCodaDataset`) after each stage, `self.cfg`-sourced fields spot-checked post-item-4, `get_download_issues`' both branches exercised live, and the `.astype(object)` timestamp-precision claim verified with a standalone before/after numeric test (not just read by eye).

Not verified (explicitly out of scope for this pass, flagged to the user): no `OPENLORIS`/`UT-CODA` data exists in either local benchmark directory, so nothing here has been run against real files — SKILL.md step 8's function-by-function live simulation hasn't been performed. The `remove_unused_files` implementations in particular are reasoned from code alone, not confirmed against a real download.

No commit yet — pending.

### 2026-07-28 — Template/SKILL.md updated: symlinked-vs-copied raw source under `remove_unused_files`

Files: `Datasets/extra-files/dataset_template.py`, `.claude/skills/add-dataset/SKILL.md`.

Prompted by designing `OpenlorisDataset.remove_unused_files`/`UtCodaDataset.remove_unused_files` above: both datasets' `create_rgb_folder` *symlinks* `rgb_0`/`depth_0`/`rgb_1` directly onto their raw source folders (`color/`/`aligned_depth/`/`fisheye1/`/`fisheye2/`, `2d_rect/`) rather than copying them — meaning those raw folders can never be deleted at any `Retention` tier, including `MINIMAL`, without leaving the standardized layout's own symlinks dangling. The template's existing `remove_unused_files` comment already documented `MINIMAL` deleting "un-resized raw images" (citing `HFColmapDatasetMixin`'s `rgb_0_raw/`, which *is* a copy and genuinely safe to delete) without ever noting that this only holds for the copy case — a real gap, not speculative, since two datasets in this same pass hit it independently.

Added a caveat paragraph to both `dataset_template.py`'s `remove_unused_files` comment (right after the three-tier bullet list) and SKILL.md step 4g (appended to the existing sentence), citing `dataset_openloris.py`/`dataset_ut_coda.py` as the models for the symlink case and `HFColmapDatasetMixin` for the safe-to-delete copy case, so the next dataset with a symlink-based `create_rgb_folder` doesn't have to rediscover this from scratch.

`CLAUDE.md` and `dataset_template.yaml` checked, no gap found (same conclusion as every prior pass — `CLAUDE.md` stays intentionally high-level, and this finding is pure `.py`-level `remove_unused_files` logic with nothing yaml-field-shaped about it).

No commit yet — pending (uncommitted along with the pass above).

### 2026-07-28 — rover / msd: full checklist pass

Files: `dataset_rover.py` (`RoverDataset` shared base + all three registered children `RoverT265Dataset`/`RoverD435iDataset`/`RoverPicamDataset` — no unregistered downstream child, same shape as the openloris pass), `dataset_rover-t265.yaml`, `dataset_rover-d435i.yaml`, `dataset_rover-picam.yaml`, `dataset_msd.py` (`MsdDataset`), `dataset_msd.yaml`. Findings tracked upfront as a checklist in [GitHub issue #90](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/90), then worked one checklist item/function at a time in conversation, with findings presented before each fix — this entry backfills the log to match.

**Note:** `dataset_msd.py`/`dataset_msd.yaml` already had an uncommitted diff in the working tree at the start (`repo_id` → `hf_repo_id`, matching the same rename already done for openloris/soneva/sweetcorals) — confirmed with the user this was prior work to build on, so this pass's findings reflect the current (uncommitted) state. The three rover files had no pending changes.

- **Header consistency (item 2)** — neither `.py` file had a module header docstring, and none of the 4 yamls had `about:`/`vslamlab_maintainer:` blocks at all (both predate the convention — rover/msd were among the "18 of 34 datasets with no `about:` block" noted after an earlier table regeneration). Added both, with `Created`/`vslamlab_maintainer.date` sourced from each file's *original* integration commit via `git log --diff-filter=A --follow` across renames (rover: 2025-09-05, predating the later split into per-sensor yamls; msd: 2025-11-06, predating its rename from `dataset_msdmi.py`) — not the later refactor commits. `about:` fields (license/summary/homepage/authors) verified via live web fetch: ROVER — MIT license (per its Hugging Face dataset card, not just the paper page), authors led by Fabian Schmidt (IEEE T-RO 2025), trimmed to lead author + `Others` per the eth/madmax precedent; MSD — CC-BY 4.0, three authors (Mateo de Mayo, Daniel Cremers, Taihú Pire, IROS 2025), short enough to list all three. One judgment call resolved with the user rather than assumed: both files' original integration commits were authored by external contributors (`niyolu` for rover, Mateo de Mayo — also an MSD paper author — for msd), a new wrinkle beyond the existing "external committer, still credit Alejandro Fontan" precedent (ut-coda) since Mateo de Mayo's case is *also* the "integrator was one of the dataset's original creators" scenario SKILL.md step 2 treats as a stronger claim. User's call: keep `vslamlab_maintainer.name: Alejandro Fontan` for both, consistent with every other dataset.
- **Class docstrings** (not itself a numbered item, same casing pattern flagged for replica/tartanair/nuim/sesoko/openloris) — `"""ROVER..."""`/`"""ROVER T265/D435i/Picam..."""`/`"""MSD..."""` used the raw all-caps acronym with no disambiguating context, unlike kitti/madmax's precedent of adding a few descriptive words to a real acronym-style name. User picked `"ROVER multiseason dataset helper..."` (base + subclass sensor suffix, e.g. `"ROVER T265 multiseason dataset helper..."`) and `"Monado SLAM Dataset (MSD) helper..."` from a set of options rather than guessing.
- **`__init__` contents / redundant yaml reopen (items 3, 4)** — `RoverDataset.__init__`, `RoverD435iDataset.__init__` (reopened a *second* time just for `depth_factor`), and `MsdDataset.__init__` all still reopened `self.yaml_file` instead of reading `self.cfg`; switched all three (`import yaml` stays in `dataset_rover.py`, still needed by the three subclasses' `create_calibration_yaml`; dropped from `dataset_msd.py`, fully unused after the switch). Separately, `RoverDataset.__init__` set `self.sequence_nicknames = self.sequence_names[:]` — not a no-op (rover's names do contain underscores the base class default would swap), but a *worse* nickname than the default (`"campus_small_autumn_t265"` vs. `"campus small autumn t265"`), reading like a pre-`default_sequence_nicknames()` leftover rather than a deliberate choice. User confirmed removing it. `MsdDataset`'s `[s.split("_")[0] for s in self.sequence_names]` reviewed, correctly built from raw `sequence_names` — no issue.
- **`download_sequence_data` pass** — found and fixed a real bug in `RoverDataset._sequence_data_from_name` (called from `download_sequence_data`): a `for...else` that only `print()`ed a warning on no location match, leaving `location = None` and crashing one line later with a confusing `TypeError` — same bug class already fixed for 7-Scenes' `_find_sequence_group` in #78; now raises a clear `ValueError`. Bigger finding, resolved with the user via two design questions: (1) `RoverDataset.seq2group`, a *class-level* (not instance) mutable dict populated by `download_sequence_data` and read by `create_groundtruth_csv` — removed entirely in favor of a new `_sequence_group_path(sequence_name)` helper both hooks call independently, so `create_groundtruth_csv` no longer depends on `download_sequence_data` having already run on the same instance; (2) added a `.download_complete` completion-marker file to both `RoverDataset._ensure_data_exists` and `MsdDataset.download_sequence_data`, closing the "plain `.exists()` check can't tell fully-downloaded from crashed-partway" gap the template already flagged as worth considering — verified with synthetic before/after tests (repeat call is a no-op with the marker present; removing the marker to simulate a crash correctly triggers a full re-run for both).
- **`create_rgb_folder` pass (item 11)** — `RoverT265Dataset`/`RoverD435iDataset`/`RoverPicamDataset` all hardcoded `"rgb_0"`/`"depth_0"` instead of `self.rgb_path()`/`self.depth_path()`; fixed (`rgb_1` stays literal everywhere, matching the euroc/kitti precedent — no base-class helper exists for it). `MsdDataset.create_rgb_folder` builds `f"rgb_{cam}"` dynamically; special-cased `cam == 0` to use `self.rgb_path()` too, for consistency. Verified via synthetic temp-directory round-trips for all four — identical resulting layout (symlinks for T265/Picam/Msd, a real move for D435i) to before the change.
- **`create_rgb_csv` pass (item 12)** — found and fixed, with the user's explicit confirmation, a real bug in `RoverT265Dataset.create_rgb_csv`: it replaced `'depth/'` → `'rgb_1/'` in the right camera's raw path column, even though T265 has no depth sensor (a copy-paste leftover from the D435i sibling script, which genuinely does have `"depth/"`-prefixed raw paths). Rebuilt both `cam_left`/`cam_right` path columns from each row's filename (`Path(p).name`) instead of assuming any specific raw prefix — robust regardless of what the raw metadata actually uses, verified against synthetic input carrying two different wrong prefixes. Migrated all four hooks (`RoverT265Dataset`/`RoverD435iDataset`/`RoverPicamDataset`/`MsdDataset`) to `utilities.write_csv_rows`; two of them (T265, Picam) had no atomic tmp-file+replace at all beforehand — a real robustness gap, not just style, now closed.
- **`create_imu_csv` pass** — re-hit the mixed-int64/float64-column precision trap already fixed for euroc/madmax/openloris/ut-coda, this time while *migrating* rather than in the original code: both `RoverDataset.create_imu_csv` and `MsdDataset.create_imu_csv` build a DataFrame mixing an int64 ns-timestamp column with float64 sensor columns, currently safe because pandas' own `to_csv` formats per-column — but a naive `.values.tolist()` migration to `write_csv_rows` would have silently corrupted the timestamp. Proved this with a synthetic test (a realistic ns timestamp came back wrong via plain `.values`, exact via `.astype(object).values.tolist()`) before applying the fix to both. `RoverPicamDataset.create_imu_csv`'s `pass` no-op reviewed, correctly a no-op (no `-vi` mode for that sensor).
- **`create_calibration_yaml` pass** — no correctness bugs (both read from already-typed YAML/JSON, so the raw-text/CSV `float()`-cast warning doesn't apply). Reformatted all calibration dict literals (T265's `rgb0`/`rgb1`/`imu`, D435i's `rgbd0`/`imu`, Picam's `rgb0`, MSD's `cam`/`imu`) to the one-key-per-line, trailing-comma style already applied to eth/replica/rgbdtum/tartanair/nuim/7scenes; folded D435i's post-hoc `rgbd0["distortion_type"] = ...` assignments into the dict literal; added the missing `dict[str, Any]` annotation to Picam's `rgb0` and MSD's `imu`; simplified MSD's redundant `np.array(np.eye(4)).reshape((4, 4))` to `np.eye(4)`; MSD's `imu["fps"]` switched from re-reading the raw JSON int directly to `float(imu_hz)`, confirmed via test to fix a latent int-vs-float inconsistency (`fps: 1000` instead of `fps: 1000.0`). Verified byte-for-byte against fabricated calib YAML/JSON fixtures.
- **`create_groundtruth_csv` pass** — two more real bugs beyond the already-flagged missing-atomicity gap. `RoverDataset.create_groundtruth_csv`'s `if not parts or len(parts) < 1` guard was dead code (`"".split(" ")` returns `['']`, a 1-element list, so the condition is never true) — a blank line in the raw `groundtruth.txt` crashed on `float('')`; fixed with a proper skip. `MsdDataset.create_groundtruth_csv` had the "no file written at all" bug already fixed for sweetcorals/rgbdtum/euroc/kitti/openloris (`if not gt_csv.exists(): return` skipped writing `groundtruth.csv` entirely) and routed an already-integer ns timestamp through `float()` unnecessarily (`int(float(parts[0]))`), risking the same precision loss just fixed for `create_imu_csv` — switched to `int(parts[0])` directly. Both migrated to `write_csv_rows`; `dataset_msd.py`'s `import csv` dropped, fully unused after the migration. Verified with synthetic tests covering blank/comment/malformed lines, a missing source file (now header-only instead of no-file-or-crash), and a large ns timestamp round-tripping exactly.
- **`remove_unused_files`/`Retention` tiers + shared-archive scope (items 7, 15)** — both were under-implemented (`RoverDataset.remove_unused_files` was entirely commented out; `MsdDataset.remove_unused_files` only handled `MINIMAL`), and neither could be verified against real downloaded data (no `ROVER`/`MSD` folders exist locally), so the design was worked through structurally rather than empirically, mirroring the openloris pass's approach. Traced the full symlink/move chain for every sensor class first: `sequence_path` itself is *always* a symlink into the shared group folder for every rover sensor (not just the ones whose `create_rgb_folder` symlinks `rgb_0`/`rgb_1`), so the decompressed group folder and `master_calibration_path` can never be deleted at any tier, for any sensor. Implemented: `imu/imu.txt` deletion (exclusive to one sensor-sequence, safe) plus the group/calibration `.zip` archives at `MINIMAL` (each only ever read once, during their own marker-gated extraction — safe regardless of processing order). Deliberately did **not** delete `groundtruth.txt` (shared at the group root by every sensor sub-sequence in that group) after identifying, and confirming with the user, that the usual "on-demand redownload fallback" doesn't actually cover a single file being deleted from an otherwise-intact group folder — only the whole group folder plus its completion marker being gone. `MsdDataset` additionally now cleans `mav0/cam*/data.csv`/`imu0/data.csv`/`gt/data.csv` at `STANDARD`+ (no cross-sequence sharing risk there, unlike rover, since each MSD sequence downloads its own independent zip) and reads `self.calibration_file` instead of a hardcoded `"calibration.json"` literal. All three retention tiers verified via synthetic tests for both files, confirming symlink targets and shared/reused resources are never touched at any tier, and confirmed the real `VSLAM-LAB-Benchmark/ROVER/` directory (accidentally touched by an earlier improperly-sandboxed test in this same pass, cleaned up once noticed) ends up untouched by the final test run.
- **`get_download_issues` pass (item 8)** — investigated rather than pattern-matched from a sibling. `RoverDataset` correctly has no override: confirmed live that `https://fdm.hs-esslingen.de/schmidt2025rover/calibration.zip` downloads via a plain anonymous HTTPS GET (200 OK, no auth). `MsdDataset` was missing `token=hf_token()` on both `hf_hub_download` calls — added (harmless, uses a configured token if present) — but a planned `get_download_issues` override modeled on `HFColmapDatasetMixin`/openloris's `huggingface_token` pattern was *not* added: verified via `huggingface_hub.HfApi().dataset_info("collabora/monado-slam-datasets", token=None)` that the repo is fully public (`gated: False`, anonymous `list_repo_files()` succeeds with 375 files), so reporting that issue would have been a false positive. While looking for a genuinely-correct comparison example, the same live check was run against `dataset_soneva.py`/`dataset_sweetcorals.py`/`dataset_openloris.py`'s own repos — **all three also came back `gated: False`, anonymously listable**, despite each reporting a `huggingface_token` issue today. This is bigger than the rover/msd pass and wasn't resolved here (soneva/sweetcorals/openloris are reference datasets, out of scope) — filed as [#91](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/91) per the user's explicit call to log-and-flag rather than investigate now or silently pick a side.
- **YAML formatting (item 9)** — all 4 yamls placed their download-source field (`url_download_root`/`hf_repo_id`) *before* `modes`/`cam_models`, the same field-order bug already fixed for kitti/replica/euroc/etc.; reordered to match. `dataset_rover.py`/`dataset_msd.py` were both missing a final trailing newline; `dataset_msd.yaml` had one *extra* trailing blank line (`\n\n` at EOF) instead of a single newline — all fixed. The three rover yamls' newly-added `about:`/`vslamlab_maintainer:` blocks were already correctly 2-space-indented with no other issues.

Verified throughout: `ast.parse`/`yaml.safe_load` after every edit, plus `pixi run -e vslamlab python` instantiation of all four classes at each stage; every hook touching on-disk data was checked against synthetic temp-directory fixtures (not just read by eye), including two precision-sensitive claims proven with before/after numeric comparisons (the `.astype(object)` timestamp fix, and rover's raw-seconds-format rounding being pre-existing and unaffected by the migration) and three retention-tier behaviors proven per-file across `FULL`/`STANDARD`/`MINIMAL`.

Not verified (explicitly out of scope, flagged to the user, same caveat as the openloris pass): no `ROVER`/`MSD` folders exist in the local benchmark directory, so nothing here has been run against a real download — SKILL.md step 8's live simulation hasn't been performed.

No commit yet — pending.

### 2026-07-28 — Template/SKILL.md updated to encode the rover/msd pass's findings

Files: `Datasets/extra-files/dataset_template.py`, `.claude/skills/add-dataset/SKILL.md`, this log.

Same closing pattern as every prior full-checklist pass — asked whether anything from the pass above should be added to the template/SKILL.md/`CLAUDE.md`/`dataset_template.yaml`, three real gaps confirmed (all backed by findings above, not speculative):

- `download_sequence_data`'s completion-marker paragraph only cited `dataset_squidle.py` (the `api` pattern) — added `dataset_rover.py`'s `_ensure_data_exists` (`website`) and `dataset_msd.py`'s `download_sequence_data` (`hugging-face`, single-named-file) as models showing the same pattern applies outside the `api` case. Also added a new paragraph warning against stashing per-sequence derived state on `self` for a *different* hook to read later (the `seq2group` bug above) — each hook must stay independently callable; give a later hook needing the same value its own helper that recomputes it from `sequence_name`.
- `remove_unused_files`'s shared-archive scope guidance had two gaps the rover design surfaced: the scene/group bullet didn't note that the "re-downloads on demand" fallback usually only covers the *whole* group folder (plus its marker) being gone, not a single file deleted from an otherwise-intact one — added as a caveat, citing rover's `groundtruth.txt`. And there was no guidance at all for a shared resource that's reused *indefinitely* by every future sequence rather than scoped to one download batch (rover's `master_calibration_path`) — added as a third category alongside the existing whole-dataset/scene-group split.
- `get_download_issues`'s guidance said what each issue type means but never warned against copying one from a sibling sharing the same download pattern without verifying the constraint actually holds — added, citing msd's false-positive-avoided case and pointing at #91 for the now-open question about whether soneva/sweetcorals/openloris's existing checks are themselves correct.

`CLAUDE.md` and `dataset_template.yaml` checked, no gap found (same conclusion as every prior pass — no new yaml-field shape or closed-list value was discovered in this pass, and `CLAUDE.md` stays intentionally high-level).

`Datasets/extra-files/dataset_table.md` regenerated — diff includes this pass's `msd`/`rover-d435i`/`rover-picam`/`rover-t265` rows (now showing their `about:`-derived Dataset/License/Maintainer/AI-Assisted columns) plus `openloris-d400`/`openloris-t265`/`ut-coda` rows that were already uncommitted from the earlier #88 pass and had never been regenerated since — the table reflects the whole yaml corpus, not just one pass's scope, so this mixed diff is expected.

No commit yet — pending (uncommitted along with the pass above).
