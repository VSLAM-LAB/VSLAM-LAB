---
description: Critical integration surfaces (base classes, registries, YAML schema, and pixi tasks) for VSLAM-LAB.
globs:
  - "pixi.toml"
  - "vslamlab_gui.py"
  - "vslamlab_utilities.py"
  - "Baselines/BaselineVSLAMLab.py"
  - "Baselines/get_baseline.py"
  - "Baselines/baseline_files/*.py"
  - "Datasets/DatasetVSLAMLab.py"
  - "Datasets/get_dataset.py"
  - "Datasets/dataset_files/*"
  - "configs/**/*.yaml"
  - "path_constants.py"
---

# Critical components for integration (repo-accurate)

## `pixi.toml` (environments + tasks)

- Baselines are installed/executed through **per-baseline environments** (e.g. `orbslam2`, `dpvo-dev`) defined under `[environments]`.
- Each baseline environment typically defines tasks under `[feature.<baseline>.tasks]`, commonly:
  - `git-clone`
  - `install` (optional depending on baseline)
  - `execute-mono` / `execute-rgbd` / `execute-stereo` / `execute-mono-vi` / etc.
- The VSLAM-LAB pipeline entrypoints are exposed as `pixi run ...` tasks under `[feature.vslamlab.tasks]` and dispatch through `vslamlab_gui.py`.

## Base classes you must match

- **`Baselines/BaselineVSLAMLab.py`**:
  - You implement a concrete baseline class that provides:
    - `build_execute_command(...)` (typically delegates to `build_execute_command_cpp` or `build_execute_command_python`)
    - `is_installed()`
  - VSLAM-LAB execution is ultimately a `pixi run --frozen -e <baseline> execute-<mode> ...` command.
- **`Datasets/DatasetVSLAMLab.py`**:
  - You implement a concrete dataset class that provides:
    - `download_sequence_data(sequence_name)`
    - `create_rgb_folder(sequence_name)`
    - `create_rgb_csv(sequence_name)`
    - `create_calibration_yaml(sequence_name)`
  - Optional hooks exist (`create_imu_csv`, `create_groundtruth_csv`, `remove_unused_files`).

## Registries (what makes things “discoverable”)

- **Baselines**: `Baselines/get_baseline.py`
  - Add an import for your new baseline class and add a new key to the `get_baseline_switcher()` mapping.
- **Datasets**: `Datasets/get_dataset.py`
  - Add an import and add a new key to the `switcher` mapping in `get_dataset(...)`.

## YAML schema (what `vslamlab_utilities.py` actually reads)

Experiment YAMLs (`configs/exp_*.yaml`) look like:

```yaml
my_experiment_name:
  Config: config_easy.yaml        # dataset → sequences mapping file under configs/
  NumRuns: 10
  Parameters: { verbose: 0, mode: mono }
  Module: orbslam2                # baseline key as registered in Baselines/get_baseline.py
  Ablation: configs/config_ablation.yaml  # optional
```

Notes:

- The key is **`Module`** (not `Method`).
- `Parameters` are merged into baseline defaults by `BaselineVSLAMLab.build_execute_command_*`.

## Standard dataset output structure (what integrity checks require)

After `pixi run download-sequence <dataset> <sequence>` the dataset sequence folder must include (see `Datasets/DatasetVSLAMLab.py` integrity checks):

- `<VSLAMLAB_BENCHMARK>/<DATASET_FOLDER>/<sequence>/rgb_0/` (folder)
- `<...>/<sequence>/rgb.csv` (file)
- `<...>/<sequence>/calibration.yaml` (file)
- plus:
  - `rgb_1/` if the dataset supports `stereo`
  - `imu_0.csv` if the dataset supports `mono-vi`

Groundtruth is optional depending on dataset; evaluation logic may require it for some comparisons.
