# Baselines/BaselineVSLAMLAB.py


**Section:** [`# Ensure Installed`](BaselineVSLAMLAB.py#L65)
```mermaid
flowchart LR
    EI([ensure_installed]) --> FS[fetch_source] --> HS1{has_source?}
    HS1 -- no --> ENV1[ensure_pixi_env] --> FETCH["pixi run --frozen -e env fetch-source"] --> HS2{has_source?}
    HS2 -- no --> EXIT1([exit 1])
    HS1 -- yes --> INST[install]
    HS2 -- yes --> INST
    INST --> II1{is_installed?}
    II1 -- no --> ENV2[ensure_pixi_env] --> BUILD["pixi run --frozen -e env install -v<br/>(log: install_&lt;name&gt;.txt)"] --> II2{is_installed?}
    II2 -- no --> EXIT2([exit 1])
    II1 -- yes --> DONE([done])
    II2 -- yes --> DONE

    %% VSLAM-LAB logo squares: cyan #b5f3f9, periwinkle #8195fb, lavender #a59ddf
    classDef entry fill:#8195fb,stroke:#5f74d6,color:#fff
    classDef step fill:#b5f3f9,stroke:#7fcfd8,color:#1b2a4a
    classDef check fill:#a59ddf,stroke:#7e75c4,color:#1b2a4a
    classDef cmd fill:#fff,stroke:#8195fb,color:#1b2a4a
    classDef stop fill:#fff,stroke:#c0392b,color:#c0392b

    class EI,DONE entry
    class FS,INST,ENV1,ENV2 step
    class HS1,HS2,II1,II2 check
    class FETCH,BUILD cmd
    class EXIT1,EXIT2 stop
```

```python
def ensure_installed(self) -> None
```
- orchestrator: `fetch_source()` then `install()`; both early-return when nothing to do, so the steady-state cost is two filesystem stats per run.
- called from: `run_sequence` (every run), `vslamlab_utilities.install_baseline` (`pixi run install-baseline`), which is now just this call.

```python
def ensure_pixi_env(self) -> None
```
- `pixi install -e <env>`: solves/installs the env (refreshing `pixi.lock` if `pixi.toml` changed) so the `--frozen` tasks below can run; output stays on the terminal. Runs once per process (`_pixi_env_ready` flag).

```python
def has_source(self) -> bool
```
- the `fetch-source` pixi task must leave a `.git` at `baseline_path`, or the subclass overrides `has_source`.
- called from: `info_print` (path status).

```python
def fetch_source(self) -> None
```
- streams the `fetch-source` task to the terminal (no log file); exits with an error if `has_source()` is still false afterwards.

```python
def is_installed(self) -> tuple[bool, str]
```
- not abstract anymore: base default returns `has_source()`, right for conda-package baselines (no `install` pixi task; the executable ships in the env). Source-built baselines (`-dev`, allfeature, vggt, pycuvslam) override it with a build-artifact check.
- called from: `info_print`, `vslamlab_utilities.check_experiment_baselines_installed` (pre-flight status list before an experiment).

```python
def install(self) -> None
```
- same shape as `fetch_source`, but keeps its log at `<baseline_path>/install_<name>.txt`; exits with an error (including the `is_installed` message) if still not installed afterwards. The trailing `-v` is forwarded to the task (`./build.sh -v`, `pip -v`), not to pixi.