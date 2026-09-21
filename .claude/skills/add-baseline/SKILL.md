---
name: add-baseline
description: Add a new SLAM baseline to VSLAM-LAB. Use when the user asks to add/integrate a new SLAM system, wire up a baseline for the pipeline, or asks "how do I add a baseline".
---

Adding a baseline means creating a `BaselineVSLAMLAB` subclass, registering a pixi feature/environment for its dependencies, and wiring both into `Baselines/get_baseline.py`.

1. **Implement the class**: create `Baselines/baseline_files/baseline_<name>.py`, subclassing `BaselineVSLAMLAB` (`Baselines/BaselineVSLAMLAB.py`). There is no template file for baselines — copy the closest existing baseline of similar type (e.g. `baseline_droidslam.py` for a learned/CUDA method, `baseline_orbslam2.py` for a classical feature-based method) and adapt it. At minimum implement the abstract hooks:
   - `__init__(self, baseline_name, baseline_folder, default_parameters=None)` — pass the baseline's default parameters (must include `mode`), then set `self.color`, `self.modes`, `self.cam_models` and `self.command_style` (`'cpp'` for `key:value` entry points, `'python'` for `--key value` ones).
   - `build_execute_command(self, exp_it, exp, dataset, sequence_name)` — not abstract: the base builds the `pixi run --frozen -e <env> execute-<mode> ...` command from `command_style`, the fixed sequence/experiment paths and `default_parameters` overridden by the experiment's `Parameters:`. Override only to add a side step (e.g. colmap downloads a vocabulary) and call `super().build_execute_command(...)`. To derive one parameter from another, override `resolve_parameters(self, exp) -> dict` instead (e.g. allfeature fills `feature_yaml` from `feature`).
   - `is_installed(self) -> tuple[bool, str]` — not abstract: the base default returns `has_source()`, which is right for conda-package baselines (no `install` pixi task; the executable ships in the env). A baseline that builds from source **must** override it to check a build artifact (e.g. `bin/<executable>`), otherwise a failed build is reported as installed.

2. **Add a pixi feature** in `pixi.toml` for the baseline's dependencies (mirror `[feature.<name>]` blocks like `[feature.droidslam]`):
   - `[feature.<name>]` — channels/platforms (e.g. `platforms = ["linux-64-cuda"]` if it needs CUDA).
   - `[feature.<name>.tasks]` — at minimum a `fetch-source` task pointing at the baseline's source repo, plus `execute-mono`/`execute-rgbd`/`execute-stereo` tasks (whichever modes the baseline supports) that invoke its executable/entrypoint.
   - `[feature.<name>.dependencies]` — conda/pip packages the baseline needs.
   - Register the environment in the top-level `[environments]` table: `<name> = { features = ["<name>", ...], solve-group = "<name>" }` (pin shared `cuda*`/`py*` features and a `solve-group` the way `droidslam` does, to reuse dependency solves across baselines).

3. **Register it** in `Baselines/get_baseline.py`: import the class and add it to the baseline switcher/lookup, following the existing pattern (mirrors `Datasets/get_dataset.py`'s `switcher` dict).

4. **Verify**: run `pixi run install-baseline` to clone/build the new baseline, then `pixi run demo <name> <dataset> <sequence> <mode>` or a `configs/test_exp_<name>.yaml` via `pixi run vslamlab configs/test_exp_<name>.yaml` to confirm it executes end-to-end and produces a trajectory output.

Full reference docs live on the project's GitHub Wiki if more detail is needed.
