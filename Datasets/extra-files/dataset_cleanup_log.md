# Dataset cleanup log

Running log of dataset-file cleanup passes, kept so each new dataset's cleanup can reuse what
was already checked/decided elsewhere instead of re-deriving it. Not part of the add-dataset
skill's file scope — this is a separate, ongoing hygiene pass across existing `Datasets/dataset_files/*`.

## Checklist (apply per dataset, in order)

1. **Header consistency** — `.py` module docstring vs. its own `.yaml` `vslamlab_maintainer:` block vs. `Datasets/extra-files/dataset_template.py`'s header comment. Specifically:
   - `.py` `Author` line == yaml `vslamlab_maintainer.name` (verbatim).
   - `.py` `Assisted by` line == yaml `vslamlab_maintainer.assisted_by`, omitted in both if no AI agent was involved.
   - `.py` `Created` == yaml `vslamlab_maintainer.date`, *unless* the dataset predates the maintainer-block convention, in which case `Created` = original integration date and yaml `date` = last maintainer touch date (allowed to diverge, but should be a deliberate, logged decision, not drift).
   - `.py` `Updated` line present iff the file has been substantively edited since `Created` (add it the same pass as the edit that prompts it, not proactively).
   - `License` line text matches template exactly (`GPLv3 License`).

2. *(next checks TBD as they come up — e.g. YAML field shape, import-grouping order per SKILL.md step 4, `remove_unused_files`/`BENCHMARK_RETENTION` handling, etc.)*

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

Commit: *(pending — not yet committed)*

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
