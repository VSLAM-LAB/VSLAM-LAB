---
name: add-baseline
description: Add a new SLAM baseline to VSLAM-LAB. Use when the user asks to add/integrate a new SLAM system, wire up a baseline for the pipeline, or asks "how do I add a baseline".
---

Adding a baseline means creating a `BaselineVSLAMLAB` subclass, registering a pixi feature/environment for its dependencies, and wiring both into `Baselines/get_baseline.py`.

1. **Implement the class**: create `Baselines/baseline_files/baseline_<name>.py`, subclassing `BaselineVSLAMLAB` (`Baselines/BaselineVSLAMLAB.py`). There is no template file for baselines — copy the closest existing baseline of similar type (e.g. `baseline_droidslam.py` for a learned/CUDA method, `baseline_orbslam2.py` for a classical feature-based method) and adapt it. At minimum implement the abstract hooks:
   - `__init__(self, baseline_name, baseline_folder, default_parameters='')` — set `baseline_path`, `settings_yaml`, labels/colors.
   - `build_execute_command(self, exp_it, exp, dataset, sequence_name)` — construct the shell command that runs the baseline on a sequence.
   - `is_installed(self)` — check whether the baseline's environment/weights are already set up.

2. **Add a pixi feature** in `pixi.toml` for the baseline's dependencies (mirror `[feature.<name>]` blocks like `[feature.droidslam]`):
   - `[feature.<name>]` — channels/platforms (e.g. `platforms = ["linux-64-cuda"]` if it needs CUDA).
   - `[feature.<name>.tasks]` — at minimum a `git-clone` task pointing at the baseline's source repo, plus `execute-mono`/`execute-rgbd`/`execute-stereo` tasks (whichever modes the baseline supports) that invoke its executable/entrypoint.
   - `[feature.<name>.dependencies]` — conda/pip packages the baseline needs.
   - Register the environment in the top-level `[environments]` table: `<name> = { features = ["<name>", ...], solve-group = "<name>" }` (pin shared `cuda*`/`py*` features and a `solve-group` the way `droidslam` does, to reuse dependency solves across baselines).

3. **Register it** in `Baselines/get_baseline.py`: import the class and add it to the baseline switcher/lookup, following the existing pattern (mirrors `Datasets/get_dataset.py`'s `switcher` dict).

4. **Verify**: run `pixi run install-baseline` to clone/build the new baseline, then `pixi run demo <name> <dataset> <sequence> <mode>` or a `configs/test_exp_<name>.yaml` via `pixi run vslamlab configs/test_exp_<name>.yaml` to confirm it executes end-to-end and produces a trajectory output.

Full reference docs live on the project's GitHub Wiki if more detail is needed.
