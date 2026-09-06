# Capabilities

A **capability** is a learned or algorithmic per-sequence preprocessing step that VSLAM-LAB can
run on a benchmark sequence to produce an extra data stream or estimate a SLAM system can consume:
semantic masks, generated depth, estimated intrinsics, a place-recognition distance matrix, ...
Each one lives in its own file here, runs in its own pixi environment (its model stack never has to
agree with the `vslamlab` environment's dependencies), and writes its result as a **per-sequence
artifact next to the sequence's data** that `Run/run_functions.py` wires into an experiment on
demand.

| Capability | Script | pixi env / task | Artifact (inside `<sequence>/`) | Experiment parameter | Run-side hook |
|---|---|---|---|---|---|
| Static/dynamic masks (Mask2Former) | `mask2former.py` | `mask2former` / `mask-inference` | `mask2former_<i>/` + `.mask2former_complete`, one PNG per `path_rgb_<i>` frame | `segmentation: mask2former` | `append_mask2former_columns` → `ts_mask_<i> (ns)`/`path_mask_<i>` in `rgb_exp.csv` |
| Stereo depth (Fast-FoundationStereo) | `fastfoundationstereo.py` | `fastfoundationstereo` / `stereo-inference` | `fastfoundationstereo_0/` + `.fastfoundationstereo_complete` (records `depth_factor`) | `depth: fastfoundationstereo` | `append_stereo_depth_columns` → `ts_depth_0 (ns)`/`path_depth_0` + `register_depth_stream` in `calibration_exp.yaml` |
| Intrinsics estimation (AnyCalib) | `anycalib.py` | `anycalib` / `calib-inference` | `anycalib/calibration.yaml` + `anycalib/estimates.csv` | `calibration: anycalib` | `create_calibration_exp_yaml` seeds `calibration_exp.yaml` from the artifact |
| VPR distance matrix (VPR-LAB) | `vpr.py` | `vpr-lab` / `vpr` | `vpr-lab/D.npy` | `rgb_vpr: <n>` | `create_rgb_exp_csv` downsamples `rgb_exp.csv` with `sample_vpr`'s sampler |
| Flat-port refraction removal (Refrax) | `refrax.py` | `refrax` / `refrax-inference` | `refrax_0/` (corrected `<stem>.png` per rgb_0 frame, `mask.png`, `zoom_sweep.csv`, `calibration.yaml`) + `.refrax_complete` (zoom/z0/canvas/crop/housing metadata; defaults from `Baselines/Refrax/configs/vslamlab.yaml`, incl. `fit_canvas: true` = output sized to the whole corrected image, principal point shifted accordingly) | `refraction: refrax` | `replace_rgb_with_refraction_corrected` → `path_rgb_0` repointed in `rgb_exp.csv`, `ts_mask_0 (ns)`/`path_mask_0` → `refrax_0/mask.png`, `calibration_exp.yaml` replaced by the artifact's |
| Monocular depth (Depth Anything 3) | `depth_anything.py` | `depth-anything` / `depth-inference` | `depth_anything_0/` | — (not wired yet) | — |

`depth_anything.py` predates the contract below (positional `dataset sequence` arguments, writes
`path_depth_0` into the sequence's `rgb.csv` instead of an artifact + run-side hook) and is
listed here as the next migration target, not as a model to copy.

## The contract

Every capability script follows the same shape; copy `fastfoundationstereo.py` (the most complete
one) when adding a new one.

### 1. Command line

- Sequence targets come from CLAUDE.md's **sequence-target argument convention**:
  `add_sequence_target_args(parser)` + `resolve_sequence_targets_or_exit(args, parser)` from
  `utilities.py`. A capability never invents its own way to name sequences.
- Standard flags, present on every script: `--device` (default `cuda`), `--overwrite`
  (recompute even if the artifact exists) and `--prefetch` (download/cache weights and exit
  without needing any sequence target - this is what the pixi `install` task runs).
- Model-selection flags (`--model-id`, `--checkpoint`, ...) and per-capability knobs are
  hyphenated (`--depth-factor`, `--n-images`), with `dest=` set to the snake_case name.
- Output folder prefix is overridable (`--mask-folder-base`, `--depth-folder-base`) and
  defaults to the capability's own name.

### 2. Input

- Frames are read from the **full** frame list: `rgb_raw.csv` when it exists (the backup that
  `sample_vpr.py`/`synch_gt.py` leave behind after downsampling/syncing `rgb.csv`), else
  `rgb.csv`. A downsampled `rgb.csv` must still end up with complete artifact coverage.
- Streams are discovered from the header (`path_rgb_<i>` → stream `i`); a capability states
  whether it runs on every stream (masks, intrinsics) or only on `rgb_0`/the `rgb_0`+`rgb_1`
  pair (stereo depth).
- Calibration, when needed, is parsed from the sequence's `calibration.yaml`; unsupported camera
  models are a warning + skip, never a crash.
- Missing `rgb.csv` (sequence not downloaded) is a warning + skip.

### 3. Output - the artifact

- Everything is written **inside the sequence folder**, in a folder or file named after the
  capability: `<name>_<i>/` for per-frame, per-stream outputs (one file per frame, keeping the
  source frame's stem), or `<name>/` for per-sequence outputs (`anycalib/calibration.yaml`,
  `vpr-lab/D.npy`).
- A capability **never modifies `rgb.csv`, `groundtruth.csv` or `calibration.yaml`** of the
  sequence. The only things that are allowed to change are its own artifact and (see below) the
  per-experiment copies the run pipeline makes.
- Per-frame artifacts end with a hidden completion marker `.<name>_complete`. The marker is the
  "done" signal the run pipeline checks; it may carry metadata the run side needs to interpret
  the artifact (e.g. `depth_factor: 256.0`) as `key: value` lines.
- Idempotent by default: a sequence whose marker/artifact exists is skipped with an info message
  pointing at `--overwrite`. Per-frame capabilities should also **resume**: skip frames whose
  output file already exists, so an interrupted run continues where it stopped
  (`fastfoundationstereo.py` does this; `mask2former.py` does not yet).
- Load the model **lazily** (`functools.cache` around `load_model`) so a batch of already-complete
  sequences never pays the model load.
- Encodings are documented in the module docstring: masks are 8-bit `L` PNGs with
  `1 = static, 0 = dynamic`; depth is 16-bit PNG with `depth (m) = value / depth_factor` and
  `0 = invalid`.

### 4. pixi wiring

One feature + environment per capability in `pixi.toml`, named after it:

```toml
[environments]
<name> = { features = ["<name>", "cuda126", "py11"], solve-group = "<name>" }

[feature.<name>.tasks]
fetch-source = ...                                              # only if the upstream repo is imported from Baselines/<Upstream>/
install = { cmd = "python Capabilities/<name>.py --prefetch", depends-on = ["fetch-source"] }
<task> = { cmd = "python Capabilities/<name>.py", depends-on = ["install"] }
```

Upstream code that is not pip-installable is cloned into `Baselines/<Upstream>/` (gitignored)
and put on `sys.path` by the script; weights go to the HuggingFace/torch-hub cache or
`Baselines/<Upstream>/weights/`, never into this repo.

### 5. Run-side hook (`Run/run_functions.py`)

An experiment opts into a capability through a `Parameters:` key (`segmentation:`, `depth:`,
`calibration:`, `rgb_vpr:`). The hook

1. checks the artifact's marker for the sequence;
2. if missing, runs the capability in its own environment:
   `subprocess.run(["pixi", "run", "-e", "<name>", "<task>", dataset, sequence], check=True)`;
3. exposes the artifact to the baseline by editing **only the per-experiment copies**:
   appends `ts_<kind>_<i> (ns)`/`path_<kind>_<i>` columns to `rgb_exp.csv` and/or patches
   `calibration_exp.yaml` (e.g. `register_depth_stream`).

A capability may also *replace* a stream rather than add one (`refrax` rewrites
`path_rgb_0` and the rgb_0 calibration entry); such capabilities run before the additive ones,
which are keyed by frame name and geometry.

`run_functions.py` runs in the `vslamlab` environment, so it must **not import the capability
scripts** (they pull in torch); it duplicates the folder/marker constants instead. Pure,
dependency-free helpers (like `sample_vpr`'s sampler) may be imported.

## Adding a capability - checklist

1. `Capabilities/<name>.py` with the module header docstring (Author/Assisted by/Version/
   Created/Updated/License, then what it produces, the encoding, and the run-side parameter).
2. `SCRIPT_LABEL` + `print_info, print_warning = make_printers(SCRIPT_LABEL)`.
3. `main()`: `add_sequence_target_args`, standard flags, `--prefetch` early exit, lazy model
   loader, `for dataset_name, sequence_name in pairs: <name>_pair(...)`.
4. `<name>_pair()`: source csv resolution → stream discovery → marker/overwrite/resume checks →
   inference → artifact + marker.
5. `pixi.toml`: feature, environment, `install` + `<task>` tasks.
6. `Run/run_functions.py`: constants, the `Parameters:` key, the hook that shells out and appends
   columns / patches `calibration_exp.yaml`.
7. A `configs/test_exp_<name>.yaml` smoke test running a baseline with the parameter set.
8. A row in the table at the top of this file.
