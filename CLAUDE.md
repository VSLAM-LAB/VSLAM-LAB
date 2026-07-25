# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VSLAM-LAB is a framework for compiling, configuring, running, and evaluating Visual SLAM baselines against benchmark datasets from the CLI. It's Python, managed entirely through **pixi** (conda-based package manager) — there is no `pyproject.toml`/`requirements.txt`/`Makefile` at the top level.

## Environment and commands

Install pixi first (`curl -fsSL https://pixi.sh/install.sh | bash`), then all commands run via `pixi run <task>` from the repo root. Key tasks (defined in `pixi.toml`):

- `pixi run demo <baseline> <dataset> <sequence> <mode>` — quick end-to-end demo run.
- `pixi run vslamlab configs/exp_vslamlab.yaml` — run a full experiment (default config `configs/exp_vslamlab.yaml`).
- `pixi run evaluate configs/exp_debug.yaml` — evaluate results of an experiment.
- `pixi run compare configs/exp_debug.yaml` — compare results across runs/baselines.
- `pixi run install-baseline` / `install-baselines` — install one or all SLAM baselines (clones third-party source into `Baselines/<Name>/`, gitignored).
- `pixi run download-sequence` / `download-sequences` — fetch dataset sequences.
- `pixi run set-benchmark-path` / `set-evaluation-path` — change the benchmark/evaluation data directories (defaults to sibling dirs `../VSLAM-LAB-Benchmark` and `../VSLAM-LAB-Evaluation`; these tasks rewrite `path_constants.py` in place).
- `pixi run print-baselines` / `print-datasets` / `baseline-info` — introspection helpers.

`pixi.toml` defines a separate pixi environment per baseline (`droidslam`, `orbslam2`, `mast3rslam`, etc., plus `-dev` variants) since each SLAM system has different dependency/CUDA requirements.

There is **no automated test suite**. To verify a change works, run the relevant experiment config, e.g. `pixi run vslamlab configs/test_exp_<name>.yaml` or `pixi run vslamlab configs/exp_debug.yaml`, targeting the baseline/dataset you touched.

There's no linter/formatter configured yet. `ruff` is a reasonable option if one gets added later (as a pixi feature/environment) — not currently set up.

## Project structure

- `Baselines/baseline_files/*.py` — one class per SLAM baseline, subclassing `BaselineVSLAMLAB` (`Baselines/BaselineVSLAMLAB.py`).
- `Baselines/<Name>/` — actual third-party baseline source, cloned in by `install-baseline` at setup time; gitignored, not part of this repo's history.
- `Datasets/dataset_files/*.py` + paired `.yaml` — one class per dataset, subclassing `DatasetVSLAMLAB`. `Datasets/extra-files/dataset_template.py`/`.yaml` is the starting template for a new dataset.
- `configs/` — experiment YAMLs (`exp_*.yaml`, referencing a `Config:` block that lists dataset:sequence pairs, `NumRuns`, `Parameters`, `Module`) and sequence-list configs (`config_*.yaml`). `test_*` files here are test experiment configs, not pytest.
- `Run/` — pipeline execution logic.
- `Evaluate/` — metrics and evaluation logic.
- `vslamlab_gui.py`, `vslamlab_utilities.py`, `utilities.py`, `path_constants.py` — CLI entry point and shared utilities at the repo root.

## Adding a baseline or dataset

This is a plugin architecture: new datasets/baselines are added by subclassing the respective base class and registering config/pixi entries — see the `add-dataset` and `add-baseline` skills for the full workflow, or the project's GitHub Wiki.

## Sequence-target argument convention

Any pixi task or script that operates on one or more dataset sequences (downloading, running, evaluating, syncing groundtruth, etc.) should accept its targets in this shape:

- `<dataset> [<sequence> ...]` — positional, and the *only* shape that stays positional: one dataset, and (optionally) specific sequences of it. Zero sequences given means every downloaded sequence of that dataset. This is the one case that's unambiguous and needs no filesystem/repo-state lookup to parse.
- `--datasets <dataset1> <dataset2> ...` — every downloaded sequence of each named dataset.
- `--sequences <dataset> <sequence1> <sequence2> ...` — repeatable; explicit sequences of one dataset per use, e.g. `--sequences kitti 05 07 --sequences eth table_3` to mix per-dataset sequence subsets in one call.
- `--exp <exp.yaml>` — every dataset:sequence pair referenced by an experiment yaml's `Config:` file(s).
- `--configs <config.yaml>` — every pair listed directly in a config yaml (`dataset: [sequence, ...]`).

All five are additive — combine as many as apply in one invocation and their results concatenate. Everything except the bare `<dataset> [<sequence> ...]` case requires an explicit flag rather than being guessed from the argument shape: how a command parses should never depend on repo state (e.g. a stray file on disk sharing a dataset's name), which an earlier heuristic-based version of this convention was vulnerable to.

Don't hand-roll this per script — call `utilities.add_sequence_target_args(parser)` on your `argparse.ArgumentParser` to wire up all of the above, then pass the parsed args straight through to `utilities.resolve_sequence_targets(targets=args.targets, datasets=args.datasets, sequences=args.sequences, exp=args.exp, configs=args.configs, benchmark_path=...)`, which returns a flat `list[tuple[dataset_name, sequence_name]]` ready to iterate over. See `Datasets/extra-files/synch_gt.py` for a worked example, and the `pixi run synch-gt`/`pixi run vpr`/`pixi run sample-vpr` tasks for the calling convention this produces.

## Conventions

- Feature branches off `main`, PRs into `main`. All GitHub PRs (including from forks) target `main` — it's the repo's default/integration branch.
- `dev` is a personal working branch, not a shared integration branch — it isn't a PR target on GitHub. Don't assume work should branch off `dev` or merge into it; treat it like any other local/scratch branch.
- Work items are tracked as GitHub issues, labeled `baseline` / `dataset` / `capability` / `improvement` / `bug` (the last reuses GitHub's stock `bug` label). Use `gh issue list --label <label>` to browse by category, `gh issue create --label <label> ...` to file new ones.
- Logging uses a per-file `SCRIPT_LABEL` ANSI-colored prefix pattern — follow existing baseline/dataset files for style when adding new ones.
- Newer files (e.g. `BaselineVSLAMLAB.py`, `baseline_orbslam2.py`) are fully type-hinted with module header docstrings (Author/Version/Created/Updated) — match this style in new files rather than older untyped ones.
