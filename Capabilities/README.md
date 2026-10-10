# Capabilities

A **capability** is a learned or algorithmic per-sequence preprocessing step that VSLAM-LAB can
run on a benchmark sequence to produce an extra data stream or estimate a SLAM system can consume:
semantic masks, generated depth, estimated intrinsics, a place-recognition distance matrix, ...
Each one is split in two: its own **repository** (the model code and its pixi environment, so the
model stack never has to agree with the `vslamlab` environment's dependencies), and a **driver**
here, `Capabilities/<name>.py`, that runs it on VSLAM-LAB sequences. The result is a **per-sequence
artifact next to the sequence's data** that `Run/run_functions.py` wires into an experiment on
demand.

| Capability | Script | pixi env / task | Artifact (inside `<sequence>/`) | Experiment parameter | Run-side hook |
|---|---|---|---|---|---|
| Static/dynamic masks (Mask2Former) | `mask2former.py` (driver) + repo [VSLAM-LAB/mask2former](https://github.com/VSLAM-LAB/mask2former) | `mask2former` (`fetch-source`/`install`) / `mask-inference` (vslamlab env) | `mask2former_<i>/` + `.mask2former_complete`, one PNG per `path_rgb_<i>` frame | `segmentation: mask2former` | `append_mask2former_columns` → `ts_mask_<i> (ns)`/`path_mask_<i>` in `rgb_exp.csv` |
| Stereo depth (Fast-FoundationStereo) | `fastfoundationstereo.py` (driver) + repo [VSLAM-LAB/fastfoundationstereo](https://github.com/VSLAM-LAB/fastfoundationstereo) | `fastfoundationstereo` (`fetch-source`/`install`) / `stereo-inference` (vslamlab env) | `fastfoundationstereo_0/` + `.fastfoundationstereo_complete` (records `depth_factor`) | `depth: fastfoundationstereo` | `append_depth_columns` → `ts_depth_0 (ns)`/`path_depth_0` + `register_depth_stream` in `calibration_exp.yaml` |
| Intrinsics estimation (AnyCalib) | `anycalib.py` (driver) + repo [VSLAM-LAB/anycalib](https://github.com/VSLAM-LAB/anycalib) | `anycalib` (`fetch-source`/`install`) / `calib-inference` (vslamlab env) | `anycalib/calibration.yaml` + `anycalib/estimates.csv` | `calibration: anycalib` | `create_calibration_exp_yaml` seeds `calibration_exp.yaml` from the artifact |
| VPR distance matrix (VPR-LAB) | `vpr.py` (driver) + repo [alejandrofontan/VPR-methods-evaluation](https://github.com/alejandrofontan/VPR-methods-evaluation) (fork of gmberton's, entry point `vslamlab_vpr.py`) | `vpr` (`fetch-source`/`install`) / `vpr` (vslamlab env) | `vpr-lab/D.npy` | `rgb_vpr: <n>` | `create_rgb_exp_csv` downsamples `rgb_exp.csv` with `sample_vpr`'s sampler |
| Information-based frame selection (placecell) | `placecell.py` (driver) + repo [VSLAM-LAB/placecell](https://github.com/VSLAM-LAB/placecell) (the [placecell](https://github.com/alejandrofontan/placecell) library as a conda package) | `placecell` (`fetch-source`/`install`) / `placecell-select` (vslamlab env) | reuses `vpr-lab/D.npy` (generated via `vpr` when missing); standalone mode writes `placecell/rgb_placecell.csv` | `rgb_placecell: <n>` | `select_frames_with_placecell` → `rgb_exp.csv` keeps the `n` least redundant frames (placecell's greedy joint-information culler, `CullParameters.target_alive`), removal order in `<exp_folder>/rgb_placecell.csv`; mutually exclusive with `rgb_vpr` |
| Flat-port refraction removal (Refrax) | `refrax.py` (driver) + repo [VSLAM-LAB/refrax](https://github.com/VSLAM-LAB/refrax) (fork of [cllim118/Refrax](https://github.com/cllim118/Refrax), entry point `vslamlab_refrax.py`) | `refrax` (`fetch-source`/`install`) / `refrax-inference` (vslamlab env) | `refrax_0/` (corrected `<stem>.png` per rgb_0 frame, `mask.png`, `zoom_sweep.csv`, `calibration.yaml`) + `.refrax_complete` (zoom/z0/canvas/crop/housing metadata; defaults from the capability's `configs/vslamlab.yaml`, incl. `fit_canvas: true` = output sized to the whole corrected image, principal point shifted accordingly) | `refraction: refrax` | `replace_rgb_with_refraction_corrected` → `path_rgb_0` repointed in `rgb_exp.csv`, `ts_mask_0 (ns)`/`path_mask_0` → `refrax_0/mask.png`, `calibration_exp.yaml` replaced by the artifact's |
| Monocular depth (Depth Anything 3) | `depth_anything.py` (driver) + `vslamlab_depth_anything.py` in [VSLAM-LAB/depthanything3](https://github.com/VSLAM-LAB/depthanything3) (the da3 baselines' checkout) | `da3` (`fetch-source`/`install`) / `depth-inference` (vslamlab env) | `depth_anything_0/` + `.depth_anything_complete` (records `depth_factor`), one 16-bit PNG per `path_rgb_0` frame | `depth: depth_anything` | `append_depth_columns` → `ts_depth_0 (ns)`/`path_depth_0` + `register_depth_stream` in `calibration_exp.yaml` |

## Layout

Every capability follows this split (the same one as the baselines): VSLAM-LAB's `pixi.toml`/`pixi.lock` never
carry a model stack.

- **Repository** `github.com/VSLAM-LAB/<name>` (a fork of the upstream model repo when one exists), cloned to
  `Capabilities/sources/<name>/` (gitignored). It holds:
  - `vslamlab_<name>.py`, the **entry point**: takes explicit `--sequence-path <dir> [<dir> ...]` VSLAM-LAB
    sequence folders (no dataset names, no VSLAM-LAB imports), plus `--prefetch`, `--overwrite`, `--device` and
    its model flags; writes the artifact and marker described in "Output" below. Small helpers it needs (reading
    `rgb.csv`, the `_raw` backup name) are copied in, so the repo works standalone.
  - `pixi.toml` + `pixi.lock`: only the model stack (CUDA 12.9 / PyTorch 2.7, to share the pixi cache with the
    baselines), tasks `install` (= `--prefetch`, weights into the repo: `TORCH_HOME`/`HF_HOME` under `weights/`),
    `inference` and `test`. Nothing is published, so no GitHub Actions are needed. If the model uses
    `torch.compile` or Triton kernels, add `cuda-driver-dev` and `cuda-cudart-dev`: Triton builds its launcher
    against `cuda.h` at run time (`fatal error: cuda.h: No such file or directory`).
- **Driver** `Capabilities/<name>.py` (vslamlab environment, no torch): keeps the sequence-target argument
  convention, resolves targets into sequence folders and runs the repo through
  `CapabilityVSLAMLAB(<name>, "VSLAM-LAB/<name>").run(folders, extra_args)`
  (`Capabilities/CapabilityVSLAMLAB.py`: clones/installs on first use, then
  `pixi run --manifest-path <repo>/pixi.toml --frozen inference --sequence-path ...`). Anything that touches
  VSLAM-LAB itself stays here (e.g. `add_rgbd_modes`). It exposes a function the run pipeline calls
  (`generate_stereo_depth(pairs)`, `estimate_intrinsics(pairs)`, `compute_d_matrix(pairs)`, ...).

Variations in use: a capability whose input is not a sequence folder calls `CapabilityVSLAMLAB.run_args(args)`
instead (placecell takes `--d-matrix`/`--indices`); one that shares a baseline's checkout passes `path=` and its task
(`depth_anything` runs `depth-inference` in `Baselines/Depth-Anything-3`); a library the capability depends on can
be a conda package (placecell's from prefix.dev/vslamlab/vslamlab).

## The contract

`fastfoundationstereo` (driver `Capabilities/fastfoundationstereo.py`, entry point `vslamlab_fastfoundationstereo.py`
in its repository) is the most complete example; copy it when adding a new one.

### 1. Command line

- **Driver**: sequence targets come from CLAUDE.md's **sequence-target argument convention**:
  `add_sequence_target_args(parser)` + `resolve_sequence_targets_or_exit(args, parser)` from `utilities.py`. A
  capability never invents its own way to name sequences. Model flags are forwarded to the entry point only when
  given, so the entry point's defaults stay the single source; `--prefetch` runs `CapabilityVSLAMLAB.install()`.
- **Entry point**: `--sequence-path`, `--device` (default `cuda`), `--overwrite` (recompute even if the artifact
  exists) and `--prefetch` (download/cache weights and exit without any sequence - what the repo's `install` task
  runs).
- Model-selection flags (`--model-id`, `--checkpoint`, ...) and per-capability knobs are hyphenated
  (`--depth-factor`, `--n-images`), with `dest=` set to the snake_case name.
- The output folder prefix is overridable (`--mask-folder-base`, `--depth-folder-base`, `--folder-base`) and
  defaults to the capability's own name.

### 2. Input

- Frames are read from the **full** frame list: `rgb_raw.csv` when it exists (the backup that
  `sample_vpr.py`/`synch_gt.py` leave behind after downsampling/syncing `rgb.csv`), else `rgb.csv`. A downsampled
  `rgb.csv` must still end up with complete artifact coverage.
- Streams are discovered from the header (`path_rgb_<i>` -> stream `i`); a capability states whether it runs on
  every stream (masks, intrinsics) or only on `rgb_0`/the `rgb_0`+`rgb_1` pair (stereo depth).
- Calibration, when needed, is parsed from the sequence's `calibration.yaml`; unsupported camera models are a
  warning + skip, never a crash.
- Missing `rgb.csv` (sequence not downloaded) is a warning + skip.

### 3. Output - the artifact

- Everything is written **inside the sequence folder**, in a folder or file named after the capability:
  `<name>_<i>/` for per-frame, per-stream outputs (one file per frame, keeping the source frame's stem), or
  `<name>/` for per-sequence outputs (`anycalib/calibration.yaml`, `vpr-lab/D.npy`).
- A capability **never modifies `rgb.csv`, `groundtruth.csv` or `calibration.yaml`** of the sequence. The only
  things allowed to change are its own artifact and (see below) the per-experiment copies the run pipeline makes;
  a standalone mode that rewrites `rgb.csv` (placecell's `placecell-select`) keeps an `rgb_raw.csv` backup and
  offers `--revert`.
- Per-frame artifacts end with a hidden completion marker `.<name>_complete`. The marker is the "done" signal the
  run pipeline checks; it may carry metadata the run side needs to interpret the artifact (e.g.
  `depth_factor: 256.0`) as `key: value` lines.
- Idempotent by default: a sequence whose marker/artifact exists is skipped with an info message pointing at
  `--overwrite`. Per-frame capabilities also **resume**: skip frames whose output file already exists, so an
  interrupted run continues where it stopped.
- Load the model **lazily** (`functools.cache` around `load_model`) so a batch of already-complete sequences never
  pays the model load.
- Encodings are documented in the entry point's module docstring: masks are 8-bit `L` PNGs with
  `1 = static, 0 = dynamic`; depth is 16-bit PNG with `depth (m) = value / depth_factor` and `0 = invalid`.

### 4. pixi wiring

VSLAM-LAB's `pixi.toml` gets an environment with only the clone/install tasks, and the user-facing task in the
`vslamlab` environment (it runs the driver):

```toml
[environments]
<name> = { features = ["<name>"] }

[feature.<name>]
platforms = ["linux-64"]

[feature.<name>.tasks]
fetch-source = "test -d Capabilities/sources/<name> || git clone --recursive https://github.com/VSLAM-LAB/<name>.git Capabilities/sources/<name>"
install = { cmd = "pixi run install", cwd = "Capabilities/sources/<name>", depends-on = ["fetch-source"] }

[feature.vslamlab.tasks]
<task> = "python Capabilities/<name>.py"     # e.g. stereo-inference, mask-inference, calib-inference
```

The repository's own `pixi.toml` carries the model stack and the `install` / `inference` / `test` tasks; weights
go into the repository (`weights/`), never into VSLAM-LAB.

### 5. Run-side hook (`Run/run_functions.py`)

An experiment opts into a capability through a `Parameters:` key (`segmentation:`, `depth:`, `calibration:`,
`refraction:`, `rgb_vpr:`, `rgb_placecell:`). The hook

1. checks the artifact's marker for the sequence;
2. if missing, calls the driver function (e.g. `generate_stereo_depth([(dataset, sequence)])`), which runs the
   capability in its own environment;
3. exposes the artifact to the baseline by editing **only the per-experiment copies**: appends
   `ts_<kind>_<i> (ns)`/`path_<kind>_<i>` columns to `rgb_exp.csv` and/or patches `calibration_exp.yaml` (e.g.
   `register_depth_stream`).

A capability may also *replace* a stream rather than add one (`refrax` rewrites `path_rgb_0` and the rgb_0
calibration entry); such capabilities run before the additive ones, which are keyed by frame name and geometry.

`run_functions.py` runs in the `vslamlab` environment and imports the drivers directly: they carry no model
dependencies.

## Adding a capability - checklist

1. Repository `VSLAM-LAB/<name>` (fork the upstream model repo if there is one): `vslamlab_<name>.py` (entry point,
   module docstring with what it produces and the encoding) and `pixi.toml` with `install` / `inference` / `test`.
2. Driver `Capabilities/<name>.py` with the module header docstring (Author/Assisted by/Version/Created/Updated/
   License, then what it produces and the run-side parameter): `CAPABILITY = CapabilityVSLAMLAB(...)`, the driver
   function, `main()` with `add_sequence_target_args`, the forwarded flags and `--prefetch`.
3. `pixi.toml`: the `<name>` environment (`fetch-source`, `install`) and the user-facing task in the vslamlab
   feature (see "pixi wiring").
4. `Run/run_functions.py`: the `Parameters:` key and the hook (marker check -> driver function -> columns /
   `calibration_exp.yaml`).
5. A `configs/test_exp_<name>.yaml` smoke test running a baseline with the parameter set.
6. A row in the table at the top of this file.
