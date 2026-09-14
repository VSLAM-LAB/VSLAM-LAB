---
description: Skill guide for adding a new dataset to VSLAM-LAB (dataset class + YAML + registry + download verification).
globs:
  - "Datasets/DatasetVSLAMLab.py"
  - "Datasets/get_dataset.py"
  - "Datasets/dataset_files/*"
  - "Datasets/extra-files/*"
  - "configs/*.yaml"
  - "vslamlab_gui.py"
  - "vslamlab_utilities.py"
  - "path_constants.py"
---

# Skill: integrate a new dataset

## What this skill is for

Add a new dataset so it can be:

- listed by `pixi run print-datasets`
- downloadable via `pixi run download-sequence <dataset_name> <sequence_name>`
- referenced in `configs/config_*.yaml` (dataset → sequence lists)

For running/evaluating/debugging the experiment pipeline, see `skill_vslamlab_run_evaluate_debug_experiments.md`.

## Ground rules

- Don’t guess coordinate frames, time units, or depth scale. State assumptions and validate with small samples.
- The dataset must satisfy the integrity checks in `Datasets/DatasetVSLAMLab.py` (folders/files that must exist).
- **Important**: If you intend to use this dataset with `pixi run run-exp`, the run pipeline currently expects a per-sequence `groundtruth.csv` to exist (it is copied unconditionally by `Run/run_functions.py:get_sequence_data_for_evaluation`). If the dataset has no GT, you must still generate a placeholder `groundtruth.csv` (header-only) or `run-exp` will fail.

## Step-by-step (repo-accurate)

### Step 1: Create the dataset YAML (`dataset_<name>.yaml`)

Create `Datasets/dataset_files/dataset_<dataset_name>.yaml` with at least:

- `sequence_names`
- `rgb_hz`
- optional: `modes` (e.g. `["mono"]`, `["stereo","mono-vi"]`)
- optional: `cam_models` (defaults to `["pinhole"]`)

This YAML is read by `Datasets/DatasetVSLAMLab.py` during dataset initialization.

### Step 2: Implement the dataset class

Create a python file in `Datasets/dataset_files/`, e.g. `dataset_<dataset_name>.py`, and implement a class that subclasses `DatasetVSLAMLab`.

You must implement these abstract methods (names are exact in this repo):

- `download_sequence_data(sequence_name)`
- `create_rgb_folder(sequence_name)`
- `create_rgb_csv(sequence_name)`
- `create_calibration_yaml(sequence_name)`

Optional (implement if applicable):

- `create_imu_csv(sequence_name)` for `mono-vi`
- `create_groundtruth_csv(sequence_name)` if groundtruth exists
- `remove_unused_files(sequence_name)` to shrink disk usage
- `get_download_issues(sequence_names)` to surface manual steps / auth problems

Reference patterns:

- the base contract in `Datasets/DatasetVSLAMLab.py`
- existing datasets in `Datasets/dataset_files/`
- helper templates in `Datasets/extra-files/`

### Step 3: Register the dataset in the registry

Edit `Datasets/get_dataset.py`:

- add an import for your dataset class
- add a key in `get_dataset(... )`’s `switcher` mapping:
  - `"mydataset": lambda: MYDATASET_dataset(benchmark_path),`

#### Verify registry

- `pixi run print-datasets` should list your dataset key.

### Step 4: Add a config YAML that references your sequences

Add sequences to a `configs/config_*.yaml` file:

```yaml
mydataset:
  - sequence_01
  - sequence_02
```

### Step 5: Download + validate the standardized output

Run:

- `pixi run download-sequence mydataset sequence_01`

Verify the created structure under `VSLAMLAB_BENCHMARK` (from `path_constants.py`) includes:

- `<...>/<sequence>/rgb_0/`
- `<...>/<sequence>/rgb.csv`
- `<...>/<sequence>/calibration.yaml`
- plus `rgb_1/` if `stereo` and `imu_0.csv` if `mono-vi`

If you plan to run experiments (`pixi run run-exp`), also verify:

- `<...>/<sequence>/groundtruth.csv` exists (even if header-only)
