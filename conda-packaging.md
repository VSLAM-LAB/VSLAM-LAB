1) Create new prefix.dev account
    email: vslamlab@gmail.com
    username: vslamlab

2) Create channel
    name: vslamlab (namespace: vslamlab)
    logo: https://github.com/VSLAM-LAB.png
    description: Prebuilt conda packages for the Visual SLAM baselines and tools used by VSLAM-LAB.
    visibility: public
    page: https://prefix.dev/channels/@vslamlab/vslamlab
    conda url (for pixi.toml): https://prefix.dev/vslamlab/vslamlab  (namespace/channel)
    publish: rattler-build auth login prefix.dev
             rattler-build publish recipe.yaml --to https://prefix.dev/vslamlab/vslamlab

3) Package lietorch (package definition lives in the fork: github.com/VSLAM-LAB/lietorch pixi.toml + variants.yaml)
    history: first built from VSLAM-LAB recipes/lietorch (git source @ bf1bcfe), uploaded by hand as 1.0.1;
             recipes/ removed from VSLAM-LAB once the definition moved to the fork (c95efb0)
    fork changes: pyproject.toml (metadata), setup.py without hard-coded -gencode (TORCH_CUDA_ARCH_LIST),
                  pixi.toml [package] + dev env via [dev], .github/workflows/conda-package.yml (5dfc831)
    variants: CUDA 12.6 / 12.9 (zip_keys in variants.yaml), python 3.11, pytorch-gpu 2.7.*
    CI: same as droidslam (step 4) but branch `master`, checkout with submodules (eigen), test = SE3 op on CPU;
        trusted publisher VSLAM-LAB / lietorch / conda-package.yml on vslamlab/vslamlab
    login (manual uploads only): pixi auth login prefix.dev   (browser must be signed in as vslamlab)
    note:   pixi 0.81.0 `pixi publish --to https://prefix.dev/vslamlab/vslamlab` -> 401: it keeps only
            the last URL segment as channel ("vslamlab"), dropping the namespace. Use `pixi upload prefix`.
            Reported: https://github.com/prefix-dev/pixi/issues/7188
    uploaded 2026-10-07 by hand: lietorch-1.0.1-py311h5d61490_0 (cuda 12.6), -py311he38a381_0 (cuda 12.9)
    released 2026-10-07 by CI: lietorch 1.0.2 (tag v1.0.2 -> ff409f5; only build change: no hard-coded -gencode,
             so built for all TORCH_CUDA_ARCH_LIST targets), both CUDA variants

4) Package droidslam (package definition lives in the fork: github.com/VSLAM-LAB/droidslam pixi.toml + variants.yaml)
    fork changes: pyproject.toml (metadata, console scripts), pkg_resources -> importlib.resources,
                  hard-coded -gencode removed (TORCH_CUDA_ARCH_LIST), pixi.toml [package] + dev [workspace]
    CI: .github/workflows/conda-package.yml in the fork (fa7b00c)
             push to main -> build both CUDA variants, inspect packages, test imports/--help without GPU (~12 min)
             push tag vX.Y -> same + check tag == pixi.toml [package] version == pyproject.toml version, then
             publish with `rattler-build upload prefix --skip-existing --channel vslamlab/vslamlab` (no API key)
    prefix.dev trusted publisher: Settings > Repository Access > GitHub: VSLAM-LAB / droidslam /
             conda-package.yml, no environment, read/write on vslamlab/vslamlab
    release routine:
             1. bump version in pixi.toml [package] and pyproject.toml, commit, push main (CI builds + tests)
             2. GPU check: pixi run vslamlab configs/exp_debug.yaml --overwrite with droidslam-dev on that commit
             3. git tag -a vX.Y -m "droidslam X.Y" && git push origin vX.Y   (CI publishes)
             4. VSLAM-LAB: pixi update -e droidslam droidslam, exp_debug with droidslam, commit pixi.lock
             never move/reuse a tag or delete/overwrite a published package (pixi.lock pins file sha256)
    manual fallback: pixi publish --clean --target-dir output  +  pixi upload prefix --channel vslamlab/vslamlab ...
    uploaded 2026-10-07 by hand (built from 664f9ef): droidslam-1.0-py311h5d61490_0 (cuda 12.6), -py311he38a381_0 (cuda 12.9)
    released 2026-10-07 by CI: droidslam 1.1 (tag v1.1 -> dda8f3f, no code changes vs 1.0), both CUDA variants;
             VSLAM-LAB lock updated (be35adc), exp_debug table_3 ATE 2.93 mm (user) / 2.90 mm (dev)
    released 2026-10-08 by CI: droidslam 1.2 (tag v1.2 -> 31ddda0): lietorch pin 1.0.1.* -> >=1.0.1,<1.1
             VSLAM-LAB lock: droidslam 1.2 + lietorch 1.0.2 (8a300c8), exp_debug ATE 3.00 mm (user) / 2.93 mm (dev)
    dev env deps come from [package] via the [dev] table (a4eafd1): no duplicated dependency list in the fork
    VSLAM-LAB user env `droidslam`: droidslam = { version = "*", channel = "https://prefix.dev/vslamlab/vslamlab" }
    VSLAM-LAB dev env `droidslam-dev`: no dependencies of its own; tasks forward to the clone's own tasks
             ({ cmd = "pixi run [--frozen] <task>", cwd = "Baselines/DROID-SLAM-DEV" }, like allfeature-dev);
             `install` = pip install -e in place, re-run it after editing src/*.cu / *.cpp (Python edits apply immediately)
    not a pixi-build path dependency on purpose: an editable pixi-build install links droid_backends.so against
             pixi's hidden build env (RPATH .pixi/bld/...), loading a 2nd abseil/protobuf -> segfault
    open: droidslam env lists both CUDA platforms but both solve to the 12.9 builds (A/B choice pending)

5) Third-party packages: github.com/VSLAM-LAB/conda-recipes (clone in VSLAM-LAB/recipes/, git-ignored)
    for upstream code packaged unchanged; one folder per package with a pixi.toml [package] (upstream git + rev)
    CI: push to main -> build + test every package folder; tag <package>-vX.Y.Z -> check version, build, test,
        publish via trusted publishing (VSLAM-LAB / conda-recipes / conda-package.yml on vslamlab/vslamlab)
    pypose 0.7.3 (rev 70da60d, what DPVO uses): noarch python, Apache-2.0; host pytest-runner (setup_requires,
        builds have no PyPI), run pytorch >=2 / numpy / packaging (imported, not declared upstream)
    released 2026-10-08: tag pypose-v0.7.3 -> pypose-0.7.3-pyh4616a5c_0 (noarch)
    roma 1.5.2.1 (naver/roma tag v1.5.2.1 = b6f1d0a, what fontan had): noarch python, BSD-3-Clause, run pytorch/numpy;
        released 2026-10-08: tag roma-v1.5.2.1 (conda-recipes 32a4554). Used by MASt3R/DUSt3R
    Pangolin: conda-forge pangolin-opengl (0.9.x; linux/osx/win), NOT our own package: fontan pangolin
        (2024.07.03 master snapshot, linux only) gets replaced baseline by baseline as each is rebuilt

6) Package dpvo (package definition lives in the fork: github.com/VSLAM-LAB/dpvo pixi.toml + variants.yaml)
    fork changes (37a673e): pyproject.toml (metadata, vslamlab_dpvo_mono script); setup.py builds the 3 CUDA
        extensions + DPViewer (dpviewerx, pangolin-opengl) + DPRetrieval (dpretrieval, DBoW2 submodule compiled
        in) via CMake, CUDA archs from TORCH_CUDA_ARCH_LIST ("native" for dev), one build dir per environment
        (parallel CUDA variants shared build/temp -> stale CMake cache / mixed objects otherwise)
        DPViewer CMake: no GPU auto-detect, no forced _GLIBCXX_USE_CXX11_ABI=0, empty CUDA::nvToolsExt target
        (CUDA 12 has no nvToolsExt), find_package(OpenGL) before Pangolin
        host needs cuda-nvrtc-dev (Torch CMake config), libgl/libopengl-devel, libopencv (its run_exports pin
        the OpenCV dpretrieval links; without it the package pulled another OpenCV and failed to import)
    loop closure: --loop_closure 0 DPVO / 1 DPV-SLAM (proximity, default) / 2 DPV-SLAM++ (+ DBoW2 retrieval,
        DISK+LightGlue, Sim3 PGO); --orb_vocab; mode 2 fails loudly instead of silently disabling itself
        ORBvoc.txt.tar.gz on huggingface.co/vslamlab/dpvo_weights, fetched on first loop_closure: 2 run;
        DISK/LightGlue weights via torch.hub into Baselines/torch_home (TORCH_HOME)
        KITTI 00 ATE: mode 0 112.4 m, mode 1 104.7 m, mode 2 67.7 m (9 loops); KITTI 07: no loop accepted
        (2 of the 3 consecutive retrieval hits needed). Paper's DPV-SLAM++ KITTI 00 is much lower: not investigated
    CI note: import dpvo.dpvo (and vslamlab_dpvo_mono --help) creates a CUDA tensor -> CI test imports the rest
    released 2026-10-08 by CI: dpvo 1.0 (tag v1.0 -> 37a673e), both CUDA variants
    VSLAM-LAB (b00c677): dpvo = package from vslamlab channel; dpvo-dev forwards to the clone (like droidslam-dev);
        DPVO_baseline params loop_closure (default 1) + orb_vocab; exp_debug KITTI 07: dpvo (mode 2) 20.71 m,
        dpvo-dev 19.63 m
    open: viz/verbose split (verbose 1 also opens the Pangolin viewer); per-env build dir fix for lietorch and
        droidslam at their next release

7) Package mast3rslam (package definition lives in the fork: github.com/VSLAM-LAB/mast3rslam, fork of rmurai0610/MASt3R-SLAM)
    thirdparty/ is vendored (no submodules): mast3r + dust3r (+ CroCo curope CUDA ext), asmk (asmk.hamming C ext),
        in3d viewer + pyimgui (Cython bindings to Dear ImGui)
    old fontan package: multi-step pip build.sh, hard-coded -gencode sm_86 only, no curope (dust3r used the slow
        pytorch RoPE), stray __editable__ finders pointing at a deleted build dir
    fork changes (7b7bb50): setup.py builds all in one pip install, keeping the old package layout
        (site-packages/dust3r = the DUSt3R repo with croco next to it, so path_to_croco's relative path works):
        mast3r_slam_backends + curope (CUDA, TORCH_CUDA_ARCH_LIST), asmk.hamming, imgui.core/internal from pyimgui's
        pre-generated C++ (NO cython in host: pyimgui needs Cython<0.30 to regenerate), one build dir per environment;
        pyproject.toml metadata (CC-BY-NC-SA-4.0, non-commercial) + script; pkg_resources -> importlib.resources
    run deps: pytorch-gpu 2.7.*, lietorch >=1.0.1,<1.1, roma, numpy <2, faiss (asmk), viewer stack (in3d, moderngl,
        moderngl-window, pyglfw, pyglm) and pyrealsense2 stay required: the entry script / dataloader import them
    moderngl-window: conda-forge 3.1.1 (in3d was written for 2.x); imports fine, viewer (verbose: 1) NOT tested ->
        if it breaks, package 2.4.6 in conda-recipes
    released 2026-10-08 by CI: mast3rslam 1.0 (tag v1.0 -> 7b7bb50), both CUDA variants
    VSLAM-LAB (32d0418): mast3rslam = package from vslamlab channel; mast3rslam-dev forwards to the clone;
        exp_debug ETH table_3: mast3rslam 2.56 cm, mast3rslam-dev 2.56 cm, curope used (no slow-RoPE warning)
    open: verbose also drives the viewer (no_viz = not verbose), like DPVO

8) Package vggtslam (package definition lives in the fork: github.com/VSLAM-LAB/vggtslam, fork of MIT-SPARK/VGGT-SLAM;
   same commits as the earlier alejandrofontan/VGGT-SLAM-2-VSLAM-LAB)
    pure Python -> ONE noarch package (no CUDA variants, no variants.yaml); submodules third_party/vggt (VGGT_SPARK,
        CC BY-NC 4.0) and third_party/salad (GPL-3.0) packaged in by setup.py (vggt as a namespace package, untouched)
    license: BSD-2-Clause AND CC-BY-NC-4.0 AND GPL-3.0-only (all 3 files in dist-info; conda info/licenses flattens names)
    gtsam: conda-forge gtsam 4.3.0 has the SL(4) types and its python bindings pass VGGT-SLAM's calls (SL4,
        PriorFactorSL4, BetweenFactorSL4, LM) -> NO own gtsam package; fontan gtsam "1.0" (borglab develop snapshot,
        numpy 1, cephes patch) dropped; NumPy 2 now works (2.4.6)
    pins: pytorch-gpu 2.7.* (shared with the other baselines in the pixi cache; tested combination), opencv <5,
        open3d >=0.19,<0.20 (conda-forge open3d 0.20.0 fails to import with filament 1.77.2: undefined symbol),
        viser ==0.2.23 (upstream); environments pin python 3.11 (unpinned -> 3.14, pytorch 2.10)
    memory: VGGT-SLAM peaks at ~10.5 GB RSS (VGGT-1B loaded on CPU before moving to GPU) -> needs ~11 GB free RAM;
        the first two table_3 attempts died from swap because only ~10 GB was available (not an environment problem)
    released 2026-10-08 by CI: vggtslam 2.0 (tag v2.0 -> ca3913e), noarch
    VSLAM-LAB (89ba0c1): vggtslam = package from vslamlab channel + python 3.11; vggtslam-dev forwards to the clone;
        exp_debug ETH table_3: vggtslam 2.66 cm, vggtslam-dev 2.66 cm (33 keyframes, 3 submaps, 1 loop)

TODO (follow-up): align the PyTorch builds of the GPU baselines. All use pytorch 2.7.1, but three different
    cuda129 builds ended up in the cache (~2.1 GB each): mkl _302 (droidslam), mkl _304 (dpvo, mast3rslam),
    generic _203 (vggtslam). Steer them to one build in VSLAM-LAB's root pixi.toml (e.g. build = "cuda129_mkl*")
    and the forks' dev environments, then pixi update, so the cache holds one PyTorch per CUDA version.

9) MonoGS: NO package, only monogs-dev (fork: github.com/VSLAM-LAB/monogs, fork of muskie82/MonoGS)
    licence (LICENSE.md, Imperial College London): personal to the licensee, no transfer of copies to third parties
        -> MonoGS is not redistributed as a conda package; every user builds the fork (main) from source.
        To do: delete the old monogs-vslamlab package from the fontan channel
    fork changes: setup.py builds MonoGS + both CUDA submodules (simple-knn, diff-gaussian-rasterization) in one
        pip install, one build dir per environment; pyproject.toml metadata (LicenseRef-MonoGS) + 2 scripts;
        pixi.toml: [package] (pixi-build-python) only declares the deps, which [dev] installs into the environment;
        tasks install (pip -e, local GPU), execute-mono/execute-rgbd, test. No CI workflow (nothing published)
    deps: python 3.11, numpy 2, pytorch-gpu 2.7.* on linux-64-cuda129 (the GUI open3d only solves with cuda129),
        open3d 0.19.* build >=103 (BUILD_GUI=ON; gui/slam_gui.py imports open3d.visualization.gui), no evo
    VSLAM-LAB: monogs environment removed; monogs-dev forwards to Baselines/MonoGS-DEV (fetch-source clones main);
        configs/test_exp_monogs.yaml removed
    ATE is poor, but the environment is not the cause (exp_debug ETH table_3, RGB-D, 1 reset in both runs):
        python 3.11 + numpy 2.4.6: 13.6 cm (mono: 62.2 cm), exits normally
        python 3.10 + numpy 1.26.4: 15.0 cm, hung after tracking (killed after ~50 min)
