# Prompt used (for traceability)

```text
You are an AI coding agent in the repo /home/a.lunawat/ws/VSLAM-LAB (Linux, bash). Your task is to integrate TUM Mono Dataset (URL: https://cvg.cit.tum.de/data/datasets/mono-dataset ) as a new VSLAM-LAB dataset named tum_mono, end-to-end, using the repo’s dataset integration skill doc as your source of truth.

Use the skills in @docs/ai-skills to refine this prompt to come up with an end to end integration plan with verification of the integration.
```

# Integrate `tum_mono` dataset (TUM Mono Dataset)

## Overview
Integrate the TUM Monocular Visual Odometry dataset as a new VSLAM-LAB dataset key `tum_mono`, including download + standardization to the benchmark layout, registration, a smoke config/experiment, and concrete verification steps (download → integrity check → run baseline → evaluate).

## Assumptions (will be validated during implementation)
- The TUM Mono dataset sequences follow the DSO / TUM monoVO on-disk layout: `images.zip`, `times.txt`, `camera.txt`, plus optional `pcalib.txt` / `vignette.png`.
- `times.txt` is whitespace-separated with at least `frame_id timestamp exposure_ms` (timestamp is POSIX seconds), per TUM mono dataset / DSO conventions.
- `camera.txt` follows DSO geometric calibration format (Pinhole / RadTan / EquiDistant / FOV), per DSO README.

## Target: VSLAM-LAB standardized output per sequence
Under `VSLAMLAB_BENCHMARK/TUM_MONO/<sequence_name>/` we will produce:
- `rgb_0/` (extracted images)
- `rgb.csv` (timestamps in ns + relative paths)
- `calibration.yaml` (camera intrinsics + distortion if present)

## Files to add / change
- Add `Datasets/dataset_files/dataset_tum_mono.yaml`
- Add `Datasets/dataset_files/dataset_tum_mono.py`
- Update dataset registry `Datasets/get_dataset.py`
- Add sequences config `configs/config_tum_mono.yaml` (smoke subset)
- Add smoke experiment `configs/exp_tum_mono_smoke.yaml` (for end-to-end verification)

## Implementation plan
### 1) Dataset YAML
Create `Datasets/dataset_files/dataset_tum_mono.yaml`:
- `dataset_name: tum_mono`
- `modes: ['mono']`
- `cam_models`: include `['pinhole', 'radtan4', 'equid4', 'fov']` as applicable; final list will be limited to what we actually parse from `camera.txt`.
- `rgb_hz`: set based on dataset docs (most sequences are ~25–30fps, varies). Since the pipeline uses timestamps from `times.txt`, `rgb_hz` is mainly informational; we’ll set it conservatively (e.g. `30.0`) unless the dataset provides a single canonical FPS.
- `url_download_root`: base URL for per-sequence zip downloads (from the dataset site).
- `sequence_names`: include all official sequences (but we will only reference a small smoke subset in `configs/config_tum_mono.yaml`).

Verification:
- `pixi run print-datasets` should later show `tum_mono` (YAML discoverability is via `Datasets/get_dataset.py`, but this file must exist for dataset init).

### 2) Implement `TUM_MONO_dataset` adapter
Create `Datasets/dataset_files/dataset_tum_mono.py` implementing the `DatasetVSLAMLab` abstract methods:

- `download_sequence_data(sequence_name)`
  - Download `<sequence_name>.zip` from `url_download_root` into the dataset folder.
  - Decompress into `.../TUM_MONO/<sequence_name>/`.

- `create_rgb_folder(sequence_name)`
  - Extract `images.zip` into a working folder.
  - Move the extracted images into `<sequence>/rgb_0/`.
  - Keep original filenames (alphabetical sort defines frame order in many consumers).
  - Optional (only if needed for consistency): delete the intermediate extracted folder, and optionally remove `images.zip` depending on `BENCHMARK_RETENTION` policy (match how other datasets do this).

- `create_rgb_csv(sequence_name)`
  - Parse `<sequence>/times.txt`.
  - For each image in sorted `rgb_0/`, pair with the corresponding `times.txt` row by index.
  - Write `<sequence>/rgb.csv` with header consistent with other mono datasets:
    - `ts_rgb_0 (ns)`, `path_rgb_0`
  - Convert `timestamp_seconds → ts_ns = int(timestamp * 1e9)`.

- `create_calibration_yaml(sequence_name)`
  - Parse `<sequence>/camera.txt` according to DSO’s “Geometric Calibration File” format.
  - Populate a single camera entry (since dataset is mono):
    - `cam_name: rgb_0`
    - `cam_type: gray` or `rgb` (depending on image encoding; likely gray but we can keep `gray` for VO).
    - `cam_model: pinhole`
    - `focal_length: [fx, fy]`
    - `principal_point: [cx, cy]`
    - If `camera.txt` is `RadTan ... k1 k2 r1 r2`, set:
      - `distortion_type: radtan4`
      - `distortion_coefficients: [k1, k2, r1, r2]`
    - If `EquiDistant ... k1 k2 k3 k4`, set `distortion_type: equid4` + coeffs.
    - Set `fps: self.rgb_hz` (informational).
    - Set `T_BS: identity(4)` (dataset does not provide body frame; standard practice in repo when only a single camera exists).
  - Write via `DatasetVSLAMLab.write_calibration_yaml(sequence_name=..., rgb=[rgb0])`.

- Optional: `get_download_issues(sequence_names)`
  - Return a single “complete dataset is large” issue (similar to other datasets) and/or warn if user requests many sequences.

Verification:
- `pixi run download-sequence tum_mono sequence_01` creates required files/folders.
- `DatasetVSLAMLab.check_sequence_integrity(...)` returns complete.

### 3) Register dataset
Update `Datasets/get_dataset.py`:
- Add `from Datasets.dataset_files.dataset_tum_mono import TUM_MONO_dataset`.
- Add switcher entry:
  - `"tum_mono": lambda: TUM_MONO_dataset(benchmark_path),`

Verification:
- `pixi run print-datasets` lists `tum_mono`.

### 4) Add a smoke config
Add `configs/config_tum_mono.yaml` with the smoke subset:

```yaml
tum_mono:
  - sequence_01
  - sequence_02
  - sequence_03
```

Verification:
- `pixi run validate-experiment-yaml` on the smoke experiment (next step) passes dataset/sequence existence checks.

### 5) Add an end-to-end smoke experiment
Add `configs/exp_tum_mono_smoke.yaml` that runs a mono-capable baseline compatible with radtan cameras (e.g. `orbslam3` supports `pinhole/radtan4/radtan5/equid4`).

Example shape (values will mirror repo conventions):
- `Config: config_tum_mono.yaml`
- `NumRuns: 1`
- `Module: orbslam3`
- `Parameters: { mode: mono, verbose: 0 }` (keep minimal)

Verification:
- `pixi run validate-experiment-yaml configs/exp_tum_mono_smoke.yaml` succeeds.

## Verification checklist (end-to-end)
- **Discoverability**: `pixi run print-datasets` includes `tum_mono`.
- **Download+standardize**: `pixi run download-sequence tum_mono sequence_01` produces:
  - `.../TUM_MONO/sequence_01/rgb_0/` with images
  - `.../TUM_MONO/sequence_01/rgb.csv`
  - `.../TUM_MONO/sequence_01/calibration.yaml`
- **Integrity check**: re-running download should short-circuit as “available”.
- **Run**: `pixi run run-exp configs/exp_tum_mono_smoke.yaml`
- **Evaluate (if groundtruth is available in this dataset integration)**: `pixi run evaluate-exp configs/exp_tum_mono_smoke.yaml --overwrite`
  - If TUM Mono provides ground truth segments in a separate bundle, we’ll either:
    - integrate `create_groundtruth_csv` from the official supplementary material, or
    - scope evaluation to “run-only smoke” (trajectory generated) and mark groundtruth as a follow-up.

## Notes on ground truth
The TUM Mono dataset site references “supplementary material with ORB-SLAM and DSO results” and Matlab evaluation code. During implementation we will confirm whether per-sequence ground truth poses are directly downloadable and in what format; only then will we implement `create_groundtruth_csv`.

## Learnings (from end-to-end integration)
- **`run-exp` requires `groundtruth.csv`**: The run pipeline copies `<sequence>/groundtruth.csv` unconditionally (`Run/run_functions.py:get_sequence_data_for_evaluation`). If a dataset has no ground truth, you still need to generate a placeholder `groundtruth.csv` (header-only) or `run-exp` will fail before the baseline starts.
- **Stale experiment logs can reference removed sequences**: `run-exp` iterates `VSLAM-LAB-Evaluation/<exp_name>/vslamlab_exp_log.csv`. If you change `configs/config_*.yaml` but keep the same experiment name, old rows can remain and cause surprising “missing rgb.csv” errors for sequences no longer in the config. Fix: `pixi run overwrite-exp <exp_yaml>` then `pixi run run-exp <exp_yaml>`.
- **Pixi sandbox can fail on global cache lock**: When running `pixi run ...` from a sandboxed environment, we saw `failed to acquire global cache lock ... Permission denied` on `~/.cache/rattler/...`. Workaround is to run pixi tasks outside the sandbox or ensure cache directory permissions are correct.

