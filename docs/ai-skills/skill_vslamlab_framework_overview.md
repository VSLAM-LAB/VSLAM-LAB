---
description: High-level architecture and “where things live” in VSLAM-LAB.
globs:
  - "pixi.toml"
  - "vslamlab_gui.py"
  - "vslamlab_utilities.py"
  - "Baselines/**/*.py"
  - "Datasets/**/*.py"
  - "configs/**/*.yaml"
  - "path_constants.py"
---

# VSLAM-LAB architecture (context)

## What VSLAM-LAB is

VSLAM-LAB is a framework to **install baselines**, **download/standardize datasets**, **run experiments**, and **evaluate/compare results** through a single `pixi`-driven CLI.

## Core ideas that show up in code

- **Reproducibility via `pixi` environments**: each baseline has its own `pixi` environment (e.g. `orbslam2`, `dpvo-dev`) with tasks like `git-clone`, `install`, and `execute-*` defined in `pixi.toml`.
- **Standard dataset layout**: datasets are downloaded and standardized into a benchmark folder (`VSLAMLAB_BENCHMARK` in `path_constants.py`) with required files like `rgb.csv` and `calibration.yaml`.
- **Standardized execution contract**: baselines are run via a wrapper class (`Baselines/BaselineVSLAMLab.py`) that constructs a command and expects an output trajectory file per run.

## Where to look (repo-accurate)

- **CLI entrypoint**: `vslamlab_gui.py` dispatches subcommands.
- **Pipeline implementation**: `vslamlab_utilities.py` implements `install_baseline`, `download_sequence`, `run_exp`, `evaluate_exp`, `compare_exp`, validation, and experiment log management.
- **Baseline base class**: `Baselines/BaselineVSLAMLab.py` (build/execute contract + `pixi run --frozen -e <baseline> ...` integration).
- **Baseline registry**: `Baselines/get_baseline.py` (string name → class instance).
- **Dataset base class**: `Datasets/DatasetVSLAMLab.py` (download + standardization hooks).
- **Dataset registry**: `Datasets/get_dataset.py` (string name → class instance).
- **Configs**: `configs/*.yaml` (experiment YAMLs and dataset/sequence config YAMLs).
- **Paths**: `path_constants.py` (benchmark/evaluation dirs, filenames, defaults).

## “How do I run it?” (actual tasks in `pixi.toml`)

The main pipeline tasks are:

- `pixi run install-baseline <baseline_name>`
- `pixi run download-sequence <dataset_name> <sequence_name>`
- `pixi run run-exp <exp_yaml> [--overwrite]`
- `pixi run evaluate-exp <exp_yaml> [--overwrite]`
- `pixi run compare-exp <exp_yaml>`
