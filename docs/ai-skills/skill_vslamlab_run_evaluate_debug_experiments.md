---
description: Run, evaluate, compare, and debug VSLAM-LAB experiments (configs, logs, overwrite semantics, and common failure points).
globs:
  - "pixi.toml"
  - "vslamlab_gui.py"
  - "vslamlab_utilities.py"
  - "Run/**/*.py"
  - "Evaluate/**/*.py"
  - "configs/**/*.yaml"
  - "path_constants.py"
---

# Skill: run, evaluate, and debug experiments

## What this skill is for

Use this when you already have:

- a baseline key (registered in `Baselines/get_baseline.py`)
- a dataset key (registered in `Datasets/get_dataset.py`)
- some sequences listed in a `configs/config_*.yaml`

…and you want to run the pipeline reliably and diagnose failures.

If you want to generate `configs/config_*.yaml` and `configs/exp_*.yaml` from a natural-language request, see `skill_vslamlab_define_experiment_from_prompt.md`.

## The “truth” entrypoints

- CLI dispatcher: `vslamlab_gui.py`
- Implementation: `vslamlab_utilities.py`
- Default paths: `path_constants.py` (`VSLAMLAB_BENCHMARK`, `VSLAMLAB_EVALUATION`, defaults)

## Minimal experiment YAML (correct schema)

Create `configs/exp_<name>.yaml` like:

```yaml
my_exp:
  Config: config_debug.yaml
  NumRuns: 1
  Module: orbslam2
  Parameters:
    mode: mono
    verbose: 1
```

Notes:

- The key is **`Module`** (baseline), not `Method`.
- `Config` is a *config YAML filename* under `configs/` that maps dataset → sequences.

## Run / evaluate / compare commands

- Validate config + compatibility:
  - `pixi run validate-experiment-yaml <exp_yaml>`
- Run (creates/updates experiment log CSV, installs missing baselines, downloads missing sequences, then executes):
  - `pixi run run-exp <exp_yaml> [--overwrite]`
- Evaluate:
  - `pixi run evaluate-exp <exp_yaml> [--overwrite]`
- Compare:
  - `pixi run compare-exp <exp_yaml>`
- Full pipeline (run + evaluate + compare):
  - `pixi run vslamlab <exp_yaml> [--overwrite]`

## Where outputs and logs show up

Per experiment name (top-level key in the exp YAML), VSLAM-LAB uses:

- `VSLAMLAB_EVALUATION/<exp_name>/vslamlab_exp_log.csv` as the execution log
- Per-run command output logs are written by `BaselineVSLAMLab.execute(...)` as:
  - `.../<exp_it>/system_output_<exp_it>.txt` (exact location depends on the baseline’s `exp_folder`)

## What `--overwrite` actually does

- `--overwrite` triggers `overwrite_exp(exp_yaml)` which deletes prior experiment artifacts under the experiment folder(s) in `VSLAMLAB_EVALUATION`.
- Running without overwrite will skip work in some cases (notably evaluation can skip if it sees already-evaluated rows and overwrite is false).

## Common pitfall: stale experiment logs after changing config files

`run-exp` iterates the experiment log at `VSLAMLAB_EVALUATION/<exp_name>/vslamlab_exp_log.csv`.  
If you change `configs/config_*.yaml` (add/remove sequences) but reuse the same experiment name, you can end up with **stale rows** referencing sequences that are no longer in the config.

Fast fix:

- `pixi run overwrite-exp <exp_yaml>`
- then `pixi run run-exp <exp_yaml>`

## Debug checklist (tight + actionable)

When an experiment fails, the fastest triage path is:

- Baseline name not found:
  - `pixi run print-baselines`
  - fix: add to `Baselines/get_baseline.py` (registry)
- Dataset name / sequence not found:
  - `pixi run print-datasets`
  - fix: correct `configs/config_*.yaml` entries or dataset’s `sequence_names`
- Mode mismatch (`mono`/`stereo`/`rgbd`/`mono-vi`):
  - `validate-experiment-yaml` checks baseline/dataset compatibility (modes + camera models)
  - fix: align `Parameters.mode`, baseline `self.modes`, dataset `modes`
- Baseline installed but runtime fails:
  - inspect `system_output_*.txt` produced during execution
  - common fix points:
    - baseline’s `build_execute_command(...)` arguments
    - baseline `execute-*` tasks in `pixi.toml`
    - missing baseline settings yaml (`BaselineVSLAMLab.download_vslamlab_settings`)
- Dataset “available” check fails after download:
  - `DatasetVSLAMLab.check_sequence_integrity(...)` requires `rgb_0/`, `rgb.csv`, `calibration.yaml` (+ extras for stereo/vi)
  - fix: dataset class methods `create_rgb_folder/create_rgb_csv/create_calibration_yaml`
