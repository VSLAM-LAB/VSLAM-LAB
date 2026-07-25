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

5. **Unused `sequence_path` locals** — `sequence_path = self.sequence_path(sequence_name)` assigned but never read afterward (the method builds `rgb_path`/other paths via other helpers instead). Confirmed present in `dataset_soneva.py`'s `create_rgb_csv`/`create_calibration_yaml`/`create_groundtruth_csv` and `dataset_sweetcorals.py`'s `create_calibration_yaml`/`create_groundtruth_csv` (Pylance flagged these repeatedly across sessions) — not yet fixed, scoped to whichever pass reviews those specific hooks.

6. **`check_sequence_integrity` mode coverage (base class)** — `DatasetVSLAMLAB.check_sequence_integrity` only conditionally checks `rgb_1/` (stereo) and IMU CSV (mono-vi); there's no check for `depth_0/` when `'rgbd' in self.modes`. A mono-only download can get marked `"available"` and skip re-download even though `depth_0/` was never fetched, for any rgbd dataset. Tracked in [#76](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/76). Not dataset-file-specific, so this is a `DatasetVSLAMLAB.py` fix, not something a per-dataset pass can resolve on its own.

7. *(next checks TBD as they come up — e.g. YAML field shape, `remove_unused_files`/`BENCHMARK_RETENTION` handling, etc.)*

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

Commit: *(pending — not yet committed)*

### 2026-07-25 — `create_rgb_folder` pass: near-miss, caught by user before landing

Files: `dataset_eth.py`, `dataset_soneva.py`/`HFColmapDatasetMixin` (shared by `dataset_sweetcorals.py`), `Datasets/extra-files/dataset_template.py`.

eth.py's `create_rgb_folder` (a plain `rgb`→`rgb_0`/`depth`→`depth_0` rename) has no issues.

Near-miss (proposed, then reverted before commit): in `HFColmapDatasetMixin.create_rgb_folder`, `target_size` is computed once from the *first* image and reused for the whole sequence; a later image with a different original size still gets forced into that same `target_size` (the code warns about the mismatch but resizes to the cached size anyway). Read in isolation this looks like a bug — `compute_scaled_size`'s own docstring says "preserving aspect ratio," and it's cheap (pure arithmetic, no I/O), so recomputing per image looked strictly better. Proposed switching to per-image `compute_scaled_size` calls.

**User caught the flaw before it landed**: `create_calibration_yaml` computes `focal_length`/`principal_point` from a *single* reference image's resized dimensions (`next(rgb_path.iterdir())`) and applies that one calibration to every frame in the sequence — valid only if every resized image ends up at the *same* final pixel dimensions. Per-image resizing would let differently-shaped source images produce differently-shaped outputs, silently invalidating calibration for those frames. The original cached-`target_size` behavior (normalize every frame to one common size, warn-but-still-conform on outliers) is the correct one. Change reverted; no net diff to `create_rgb_folder` from this entry (the `remote_folder` rename from the previous pass is unrelated and still in place).

Lesson for future passes: before "fixing" resize/calibration-adjacent logic, check downstream consumers (`create_calibration_yaml` in particular) for assumptions the change might break — a locally-cheap, locally-correct-looking change can violate a whole-sequence invariant a different method depends on.

Commit: *(pending — not yet committed)*

### 2026-07-25 — `create_rgb_folder` template comment: missing depth_0/rgbd coverage

Files: `Datasets/extra-files/dataset_template.py`.

Finding (user-spotted): the template's `create_rgb_folder` comment mentioned `rgb_1/` for stereo modes but said nothing about `depth_0/` for rgbd modes, and didn't cite `dataset_eth.py` (an rgbd dataset, one of our three real models) anywhere — a real gap, since `create_rgb_csv`'s comment right below it *does* cover all three headers (mono/stereo/rgbd) consistently.

First fix attempt was itself wrong and got corrected by the user: I initially wrote that `depth_0/` is generally handled outside the resize branch, "normally just a plain rename/copy at original resolution." User corrected this — rgbd depth *does* need resizing when the source requires it, following the same `self.target_resolution` branch as `rgb_0`/`rgb_1`; `dataset_eth.py`'s depth being an unresized plain rename is specific to ETH3D's source already being close enough to 640x480 that `eth.yaml` sets no `target_resolution` at all (nothing gets resized for eth, not just depth) — not a general rule that depth is exempt from resizing.

Corrected fix: template now says rgbd's `depth_0/` follows the same `target_resolution` branch as `rgb_0`/`rgb_1`, but must use a non-interpolating resample (nearest-neighbor — `Image.NEAREST`/`cv2.INTER_NEAREST`) instead of LANCZOS, since interpolating resample blends depth values across object boundaries and corrupts the metric data. `dataset_eth.py` is cited specifically as the depth_0/ folder-layout model, with an explicit note that its lack of resizing is dataset-specific, not a general pattern.

Commit: *(pending — not yet committed)*
