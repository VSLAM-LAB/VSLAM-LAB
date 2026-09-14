---
description: Skill guide for adding a new SLAM baseline to VSLAM-LAB (pixi env + baseline class + registry + smoke test).
globs:
  - "pixi.toml"
  - "Baselines/BaselineVSLAMLab.py"
  - "Baselines/get_baseline.py"
  - "Baselines/baseline_files/*.py"
  - "vslamlab_gui.py"
  - "vslamlab_utilities.py"
  - "configs/*.yaml"
---

# Skill: integrate a new VSLAM baseline

## What this skill is for

Add a new baseline so it can be:

- installed via `pixi run install-baseline <baseline_name>`
- selected in experiment YAMLs via `Module: <baseline_name>`
- executed during `pixi run run-exp <exp_yaml>`

For running/evaluating/debugging the experiment pipeline, see `skill_vslamlab_run_evaluate_debug_experiments.md`.

## Ground rules

- Don’t invent build steps. Use the baseline’s docs or existing baselines in `Baselines/baseline_files/` as patterns.
- Keep changes minimal: `pixi.toml` + a new `baseline_*.py` + one registry entry in `Baselines/get_baseline.py` is the usual shape.

## Step-by-step (repo-accurate)

### Step 1: Add a `pixi` environment + tasks

In `pixi.toml`:

- Add an environment under `[environments]` (this becomes the baseline name users type):
  - example pattern: `orbslam2 = { features = ["orbslam2"], solve-group = "orb" }`
- Add a feature section `[feature.<baseline_name>]`:
  - define dependencies under `[feature.<baseline_name>.dependencies]`
  - define tasks under `[feature.<baseline_name>.tasks]`
    - `git-clone` (required if the baseline is fetched)
    - `install` (optional)
    - one or more `execute-*` tasks that match supported modes (e.g. `execute-mono`, `execute-stereo`, `execute-mono-vi`)

#### Verify install

- `pixi run install-baseline <baseline_name>` should clone+install (via `vslamlab_utilities.install_baseline`).

### Step 2: Implement the baseline wrapper class

Create a new file in `Baselines/baseline_files/`, e.g. `baseline_mybaseline.py`.

Implement a class that subclasses `BaselineVSLAMLab`:

- define `self.modes` (e.g. `['mono']`, `['mono','stereo']`, etc.)
- define `self.camera_models` (what dataset camera models are compatible)
- implement:
  - `build_execute_command(self, exp_it, exp, dataset, sequence_name)`:
    - usually `return super().build_execute_command_cpp(...)` or `..._python(...)`
  - `is_installed(self) -> tuple[bool, str]`:
    - follow the patterns in existing baseline files (some are “conda package available”, others check a built binary)

#### Important contract

- The baseline must produce the trajectory file expected by `BaselineVSLAMLab.execute(...)`:
  - `<exp_folder>/<exp_it>_<TRAJECTORY_FILE_NAME>.csv`
  - where `TRAJECTORY_FILE_NAME` is defined in `path_constants.py` (default: `KeyFrameTrajectory`)

### Step 3: Register the baseline name

Edit `Baselines/get_baseline.py`:

- add an import for your class
- add a key in `get_baseline_switcher()`:
  - `"mybaseline": lambda: MYBASELINE_baseline(),`

#### Verify registry

- `pixi run print-baselines` should list it.
- `pixi run baseline-info mybaseline` should show default parameters/modes.

### Step 4: Smoke test with a minimal exp YAML

Create a small experiment YAML under `configs/` (or reuse `configs/exp_debug.yaml`) using:

```yaml
my_smoke_test:
  Config: config_debug.yaml
  NumRuns: 1
  Module: mybaseline
  Parameters:
    mode: mono
    verbose: 1
```

#### Verify run

- `pixi run run-exp configs/<your_exp>.yaml --overwrite`
- then `pixi run evaluate-exp configs/<your_exp>.yaml --overwrite` (if your dataset+baseline support evaluation inputs)
