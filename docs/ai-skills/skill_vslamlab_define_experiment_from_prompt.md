---
description: Convert a natural-language user request into repo-correct VSLAM-LAB experiment config files (config_*.yaml + exp_*.yaml).
globs:
  - "configs/**/*.yaml"
  - "vslamlab_utilities.py"
  - "vslamlab_gui.py"
  - "Baselines/get_baseline.py"
  - "Datasets/get_dataset.py"
  - "path_constants.py"
---

# Skill: define an experiment configuration from a user prompt

## What this skill is for

Given a prompt like:

- “Run `orbslam2` on `rgbdtum` `freiburg1_xyz`, 3 runs, mode rgbd, then evaluate.”
- “Compare `dpvo` vs `droidslam` on `euroc` `MH_01_easy` stereo, 5 runs each.”

Produce ready-to-run VSLAM-LAB YAML files:

- `configs/config_<name>.yaml` (dataset → sequences)
- `configs/exp_<name>.yaml` (experiment(s) using `Module`, `Config`, `NumRuns`, `Parameters`)

## Ground rules (repo-specific)

- Use the actual schema consumed by `vslamlab_utilities.py`:
  - experiment keys: `Config`, `NumRuns`, `Module`, `Parameters` (optional: `Ablation`)
- Baseline and dataset names must match registries:
  - baselines: `Baselines/get_baseline.py` keys (also surfaced by `pixi run print-baselines`)
  - datasets: `Datasets/get_dataset.py` keys / dataset YAMLs (also surfaced by `pixi run print-datasets`)
- Always include `Parameters.mode` unless the intent is truly ambiguous.

## Defaults (when the user prompt is underspecified)

If the user doesn’t specify:

- **Runs**: default `NumRuns: 1`
- **Mode**: default to `mono` unless the dataset/baseline implies otherwise
- **Verbose**: default to `verbose: 1` (matches common baseline defaults)
- **Overwrite**: default to “no overwrite” (user can pass `--overwrite` when running)

State the defaults you chose in the produced YAML (don’t hide them).

## Output format (what to write)

### 1) Config YAML: dataset → sequences

Create `configs/config_<tag>.yaml`:

```yaml
euroc:
  - MH_01_easy
```

Rules:

- One dataset per config is the simplest default.
- If the user requests multiple datasets, include all under the same config only if you intend to run them together.

### 2) Experiment YAML: one or more experiments

Create `configs/exp_<tag>.yaml`:

```yaml
my_exp:
  Config: config_<tag>.yaml
  NumRuns: 3
  Module: orbslam2
  Parameters:
    mode: rgbd
    verbose: 1
```

For comparisons (multiple baselines), emit one experiment block per baseline (same `Config`, same `NumRuns`, baseline-specific `Module` and `Parameters`).

## How to translate a prompt → YAML (step-by-step)

1. **Extract intent**
   - Baseline(s): single (`orbslam2`) or list (`dpvo` vs `droidslam`)
   - Dataset: (`euroc`, `rgbdtum`, …)
   - Sequence(s): names requested
   - Mode: `mono`, `stereo`, `rgbd`, `mono-vi`, `stereo-vi` (as used in this repo)
   - Runs: integer
   - Actions: run only vs run+evaluate vs compare

2. **Normalize names**
   - Lowercase dataset/baseline keys as used in registries (e.g. `ut-coda`, `rover-t265`).
   - Keep sequence names as-is (they’re dataset-specific).

3. **Choose file tag + experiment names**
   - File tag: short + descriptive, e.g. `<dataset>_<seq>_<mode>_<baselines>`
   - Experiment key names: stable, e.g. `exp_<baseline>_<mode>`

4. **Write `configs/config_<tag>.yaml`**
   - Put dataset key at top level and list sequences.

5. **Write `configs/exp_<tag>.yaml`**
   - For each baseline:
     - set `Module` to the baseline key
     - set `Parameters.mode` to the chosen mode
     - include any explicit parameters from the user prompt

6. **Provide the exact run commands**
   - `pixi run validate-experiment-yaml configs/exp_<tag>.yaml`
   - `pixi run run-exp configs/exp_<tag>.yaml`
   - optionally:
     - `pixi run evaluate-exp configs/exp_<tag>.yaml`
     - `pixi run compare-exp configs/exp_<tag>.yaml`

## Compatibility guardrails (don’t generate broken configs)

Before finalizing, sanity check:

- Mode requested is supported by baseline (`baseline.modes`) and dataset (`dataset.modes`).
- Baseline camera models (`baseline.camera_models`) intersect dataset camera models (`dataset.cam_models`).

If a prompt requests an incompatible combo, generate the closest valid alternative and clearly state what changed (e.g. stereo → mono).
