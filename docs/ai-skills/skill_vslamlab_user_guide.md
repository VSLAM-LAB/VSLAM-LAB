---
description: How to use these VSLAM-LAB skill docs with Cursor (rules) and how to prompt effectively.
globs: []
---

# VSLAM-LAB skill docs user guide

## Using these files in Cursor

These docs are written so they can be used either as:

- **Standalone docs** (read and follow manually), or
- **Cursor rules** (auto-applied based on file globs).

If you want Cursor to treat them as rules:

- Copy or symlink them into `.cursor/rules/`
- Rename to `.mdc` (Cursor rule file extension)

## “Co-pilot” workflow expectation

When integrating baselines/datasets, a good agent workflow is:

- use repo files as truth (`pixi.toml`, `Baselines/*`, `Datasets/*`, `vslamlab_utilities.py`)
- state any assumptions about external repos / dataset formats
- ask for the smallest possible verification artifacts (build logs, a few lines of `rgb.csv`, folder listing)
- for transparency, write the final end-to-end plan into `docs/plans/<topic>.md` before implementation (keep the plan file immutable after; subsequent changes should be tracked as updates/notes rather than silently changing the original plan)

## Recommended prompts (repo-accurate)

### Integrate a new baseline

“I want to integrate **<baseline_name>** into VSLAM-LAB. Here is its repo/docs: **LINK**.  
Please update `pixi.toml` (env + `git-clone`/`install`/`execute-*` tasks), add `Baselines/baseline_files/baseline_<name>.py`, register it in `Baselines/get_baseline.py`, and give me a minimal `configs/exp_<name>_smoke.yaml` to run with `pixi run run-exp`.”

### Integrate a new dataset

“I want to integrate **<dataset_name>**. Here’s a small sample of its file layout + timestamp format (paste).  
Please add `Datasets/dataset_files/dataset_<name>.yaml`, implement `Datasets/dataset_files/dataset_<name>.py` (matching `DatasetVSLAMLab` abstract methods), register it in `Datasets/get_dataset.py`, and add a `configs/config_<name>.yaml` for sequences.”
