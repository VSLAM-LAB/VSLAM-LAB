# VSLAM-LAB AI skills (human guide)

This folder contains small “skill” docs that help an AI assistant work correctly in the VSLAM-LAB repo (paths, schemas, commands, and integration steps).

## What skills exist

- **`skill_vslamlab_framework_overview.md`**: architecture + where things live.
- **`skill_vslamlab_critical_components.md`**: base classes, registries, YAML schema, required dataset outputs.
- **`skill_vslamlab_integrate_baseline.md`**: add a new baseline (pixi env/tasks + wrapper class + registry + smoke test).
- **`skill_vslamlab_integrate_dataset.md`**: add a new dataset (dataset yaml + class + registry + download validation).
- **`skill_vslamlab_define_experiment_from_prompt.md`**: turn a natural-language request into `configs/config_*.yaml` + `configs/exp_*.yaml`.
- **`skill_vslamlab_run_evaluate_debug_experiments.md`**: validate/run/evaluate/compare + where logs are + debug checklist.

## How to use / trigger them

You have two simple options:

### 1) Manual (works in any chat)

- Reference the file(s) or folder in your prompt, e.g. `@docs/ai-skills/` or `@docs/ai-skills/skill_vslamlab_integrate_baseline.md`.
- Then ask for the task, e.g. “integrate baseline X” or “create an exp yaml for …”.

### 2) Cursor auto-rules (recommended for daily work)

- Copy the files into `.cursor/rules/`
- Rename them from `.md` → `.mdc`
- Cursor will auto-apply them based on the `globs:` in each file’s frontmatter.

## Handy prompts

- **Define an experiment from text**: “Use `skill_vslamlab_define_experiment_from_prompt.md` to generate `configs/config_*.yaml` + `configs/exp_*.yaml` for: …”
- **Run + debug**: “Use `skill_vslamlab_run_evaluate_debug_experiments.md` and tell me exactly which `pixi run ...` commands to run; then help interpret failures from logs.”
