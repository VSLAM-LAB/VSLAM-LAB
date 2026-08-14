<p align="center">
<div align="center">
    <img src="docs/header.png" width="500"/>
</div>

<h3 align="center"> A Comprehensive Framework for Visual SLAM Baselines and Datasets</h3>

<p align="center">
    <a href="https://scholar.google.com/citations?user=SDtnGogAAAAJ&hl=en"><strong>Alejandro Fontan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=eq46ylAAAAAJ&hl=en"><strong>Tobias Fischer</strong></a>
    ·
    <a href="https://nmarticorena.github.io/"><strong>Nicolas Marticorena</strong></a>
</p>

 <p align="center">
     <a href="https://www.linkedin.com/in/somayeh-hussaini/?originalSubdomain=au"><strong>Somayeh Hussaini</strong></a>
    ·
     <a href="https://github.com/TedVanderfeen"><strong>Ted Vanderfeen </strong></a>
    ·
     <a href="https://scholar.google.com/citations?hl=es&user=s3eIy0YAAAAJ"><strong>Beverley Gorry </strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en"><strong>Javier Civera</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TDSmCKgAAAAJ&hl=en"><strong>Michael Milford</strong></a>
</p>

<br/>
<div align="left">

![Maintained? yes](https://img.shields.io/badge/Maintained%3F-yes-success) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](.github/CONTRIBUTING.md) ![Last commit](https://img.shields.io/github/last-commit/VSLAM-LAB/VSLAM-LAB) [![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/VSLAM-LAB/VSLAM-LAB/blob/main/LICENSE.txt) [![arXiv](https://img.shields.io/badge/arXiv-2410.23690-B31B1B.svg)](https://arxiv.org/abs/2504.04457)

</div>

## Introduction

**VSLAM-LAB** is designed to simplify the development, evaluation, and application of Visual SLAM (VSLAM) systems.
This framework enables users to compile and configure VSLAM systems, download and process datasets, and design, run, and
evaluate experiments — **all from a single command line**!

**Why Use VSLAM-LAB?**
- **Unified Framework:** Streamlines the management of VSLAM systems and datasets.
- **Ease of Use:** Run experiments with minimal configuration and single command executions.
- **Broad Compatibility:** Supports a wide range of VSLAM systems and datasets.
- **Reproducible Results:** Standardized methods for evaluating and analyzing results.

<!--
<div align="center">
    <img src="docs/diagram.svg" width="960"/>
</div>
-->

## Getting Started

To ensure all dependencies are installed in a reproducible manner, we use the package management tool [**pixi**](https://pixi.sh/latest/). If you haven't installed [**pixi**](https://pixi.sh/latest/) yet, please run the following command in your terminal:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
*After installation, restart your terminal or source your shell for the changes to take effect*. For more details, refer to the [**pixi documentation**](https://pixi.sh/latest/).

*If you already have pixi remember to update:* `pixi self-update`

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/VSLAM-LAB/VSLAM-LAB.git && cd VSLAM-LAB
```

## Quick Demo
You can now execute any baseline on any sequence from any dataset within VSLAM-LAB using the following command:
```bash
pixi run demo <baseline> <dataset> <sequence> <mode>
```
For a full list of available systems and datasets, see the [VSLAM-LAB Supported Baselines and Datasets](#vslam-lab-supported-baselines-and-datasets).
Example commands:
```bash
pixi run demo mast3rslam eth table_3 mono
pixi run demo droidslam rgbdtum rgbd_dataset_freiburg1_xyz rgbd
pixi run demo orbslam2 kitti 04 stereo
pixi run demo pycuvslam euroc MH_01_easy stereo-vi
```
*To change the paths where VSLAM-LAB-Benchmark or/and VSLAM-LAB-Evaluation data are stored (for example, to /media/${USER}/data), use the following commands:*
```bash
pixi run set-benchmark-path /media/${USER}/data
pixi run set-evaluation-path /media/${USER}/data
```

## Configure your own experiments
With **VSLAM-LAB**, you can easily design and configure experiments using a YAML file and run them with a single command.
To **run** the experiment demo, execute the following command:
```bash
pixi run vslamlab configs/exp_vslamlab.yaml (--overwrite)
```

Experiments in **VSLAM-LAB** are sequences of entries in a YAML file (see example **~/VSLAM-LAB/configs/exp_vslamlab.yaml**):
```yaml
exp_demo_droidslam:
  Config: config_vslamlab.yaml               # YAML file containing the sequences to be run
  NumRuns: 1                                 # Maximum number of executions per sequence
  Parameters: {verbose: 1, rgb_idx: [0,250]} # Vector with parameters that will be input to the baseline executable
  Module: droidslam                          # droidslam/monogs/orbslam2/mast3rslam/dpvo/...

exp_demo_orbslam2:
  Config: config_vslamlab.yaml
  NumRuns: 1
  Parameters: {verbose: 1, rgb_idx: [0,250]}
  Module: orbslam2

exp_demo_colmap:
  Config: config_vslamlab.yaml
  NumRuns: 1
  Parameters: {verbose: 1, rgb_idx: [0,250], matcher_type: exhaustive}
  Module: colmap
```
**Frame selection:** `Parameters` also accepts four optional flags to control which RGB frames of a sequence are fed to the baseline. They are independent and stack in this order — `rgb_idx` slices the frame range first, `rgb_step` then keeps 1 frame every *n* of what remains, `rgb_max` truncates that result to at most that many frames, and `rgb_vpr` finally downsamples to at most that many frames by visual dissimilarity (VPR) rather than a fixed stride, so visually redundant frames are dropped preferentially. If none are set, every frame of the sequence is used.

| Parameter | Effect |
|:----------|:-------|
| `rgb_idx: [start, end]` | Keep only frames `start..end` (inclusive) |
| `rgb_step: n` | Keep 1 frame every `n` frames of what remains |
| `rgb_max: n` | Truncate the result to at most `n` frames |
| `rgb_vpr: n` | Downsample the result to at most `n` frames, chosen by VPR dissimilarity - computes the sequence's VPR distance matrix via `pixi run vpr <dataset> <sequence>` first if it isn't already cached |

```yaml
exp_vslamlab:
  Config: config_vslamlab.yaml
  NumRuns: 1
  Parameters: {verbose: 1, rgb_idx: [0, 250], rgb_step: 3, rgb_max: 50, rgb_vpr: 30}
  Module: droidslam
```
**Config** files are YAML files containing the list of sequences to be executed in the experiment (see example **~/VSLAM-LAB/configs/config_vslamlab.yaml**):
```yaml
eth:
- table_3

rgbdtum:
- 'rgbd_dataset_freiburg1_xyz'
```
For a full list of available VSLAM systems and datasets, refer to the section [VSLAM-LAB Supported Baselines and Datasets](#vslam-lab-supported-baselines-and-datasets).

## VSLAM-LAB Pipeline Commands
In addition to running the full automated pipeline, **VSLAM-LAB** provides modular commands to interact directly with datasets and baselines. For a comprehensive list of all available commands consult [Wiki: Comand‐line Interface](https://github.com/VSLAM-LAB/VSLAM-LAB/wiki/Comand%E2%80%90line-Interface)

```bash
pixi run install-baseline <baseline>                     # Example: pixi run install-baseline droidslam
pixi run download-sequence <dataset> <sequence>          # Example: pixi run download-sequence eth table_3
pixi run run-exp <exp_yaml>                              # Example: pixi run run-exp configs/exp_vslamlab.yaml
pixi run evaluate-exp <exp_yaml>                         # Example: pixi run evaluate-exp configs/exp_vslamlab.yaml
pixi run compare-exp <exp_yaml>                          # Example: pixi run compare-exp configs/exp_vslamlab.yaml
```

## Add a new VSLAM Dataset

Expand the evaluation suite by integrating custom datasets. If you're using Claude Code, run the [`add-dataset`](.claude/skills/add-dataset/SKILL.md) skill (`/add-dataset <name> <source>`) to walk through the full workflow end to end. Otherwise, follow the instructions in [Wiki: Integrate a new VSLAM Dataset](https://github.com/VSLAM-LAB/VSLAM-LAB/wiki/Integrate-a-new-VSLAM-Dataset).

## Add a new VSLAM Baseline

Incorporate new algorithms into the framework. Follow the guide in [Wiki: Integrate a new VSLAM Baseline](https://github.com/VSLAM-LAB/VSLAM-LAB/wiki/Integrate-a-new-VSLAM-Baseline). Benchmark your method against state-of-the-art baselines across all supported datasets.

For a reference implementation, see the VGGT-SLAM integration in commit [259f7ae](https://github.com/VSLAM-LAB/VSLAM-LAB/commit/259f7aec88d4576880f3cc98983660f508af13a9).


## License
**VSLAM-LAB** is released under a **LICENSE.txt**. For a list of code dependencies which are not property of the authors of **VSLAM-LAB**, please check **docs/Dependencies.md**.


## Citation
If you're using **VSLAM-LAB** in your research, please cite:
```bibtex
@INPROCEEDINGS{fontan2025vslam,
  author={Fontan, Alejandro and Fischer, Tobias and Civera, Javier and Milford, Michael},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={VSLAM-LAB: A Comprehensive Framework for Visual SLAM Methods and Datasets},
  year={2025},
  volume={},
  number={},
  pages={9013-9020},
  keywords={Visualization;Simultaneous localization and mapping;Robot vision systems;Standardization;Benchmark testing;Programming;Reproducibility of results;Intelligent robots;Faces;Software development management},
  doi={10.1109/IROS60139.2025.11247218}
}
```

<!-- ## Acknowledgements

To [awesome-slam-datasets](https://github.com/youngguncho/awesome-slam-datasets) -->

# VSLAM-LAB Supported Baselines and Datasets
*For full per-dataset details — license, access requirements, and publication/BibTeX citation — see [`Datasets/extra-files/dataset_table.xlsx`](Datasets/extra-files/dataset_table.xlsx).*

| Baselines                                                                   | System |      Modes       |                                   License                                   |    Label     |  Conda Pkg     |  Camera Models     |
|:----------------------------------------------------------------------------|:------:|:------:|:----------------:|:---------------------------------------------------------------------------:|:------------:|:------------:|
| [**VGGT-SLAM**](https://github.com/MIT-SPARK/VGGT-SLAM) |  VSLAM   |  `mono`  |  [BSD-2](https://github.com/MIT-SPARK/VGGT-SLAM/blob/main/LICENSE)  |   `vggtslam`   | ✅ | `pinhole` |
| [**MASt3R-SLAM**](https://github.com/rmurai0610/MASt3R-SLAM)                | VSLAM  |       `mono`       |    [CC BY-NC-SA 4.0](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)    | `mast3rslam`  | ✅ | `radtan5` `unknown` |
| [**DPVO**](https://github.com/princeton-vl/DPVO)                            | VSLAM  |       `mono`       |    [License](https://github.com/princeton-vl/DPVO/blob/main/LICENSE)    | `dpvo`  | ✅ | `radtan5` |
| [**DROID-SLAM**](https://github.com/princeton-vl/DROID-SLAM)                | VSLAM  |`mono` `rgbd` `stereo`|    [BSD-3](https://github.com/princeton-vl/DROID-SLAM/blob/main/LICENSE)    | `droidslam`  | ✅ | `radtan5` |
| [**ORB-SLAM2**](https://github.com/alejandrofontan/ORB_SLAM2)               | VSLAM  |`mono` `rgbd` `stereo`| [GPLv3](https://github.com/raulmur/ORB_SLAM2/blob/master/LICENSE.txt)|  `orbslam2`  | ✅ | `radtan5`  |
| [**MonoGS**](https://github.com/muskie82/MonoGS)                            | VSLAM  | `mono` `rgbd` |     [License](https://github.com/muskie82/MonoGS?tab=License-1-ov-file)     |   `monogs`   | ✅ | `radtan5` |
| [**AnyFeature-VSLAM**](https://github.com/alejandrofontan/AnyFeature-VSLAM) | VSLAM  | `mono` | [GPLv3](https://github.com/alejandrofontan/VSLAM-LAB/blob/main/LICENSE.txt) | `anyfeature` | ✅ |  `radtan5` |
| **----------** | **-------** | **-------** | **----------** | **--------** | **---** | **----------** |
| [**PyCuVSLAM**](https://github.com/VSLAM-LAB/PyCuVSLAM/tree/main) | VSLAM  |`mono` `rgbd` `stereo(-vi)` | [NVIDIA](https://github.com/VSLAM-LAB/PyCuVSLAM/blob/main/LICENSE) |  `pycuvslam`  | ➖ | `radtan5` `equid4` |
| [**ORB-SLAM3**](https://github.com/UZ-SLAMLab/ORB_SLAM3)               | VSLAM  | `mono(-vi)` `rgbd(-vi)` `stereo(-vi)` |    [GPLv3](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/LICENSE)    |  `orbslam3`  | ✅ | `radtan5` `equid4`|
| [**OKVIS2**](https://github.com/ethz-mrl/okvis2)               | VSLAM  | `mono-vi` |    [BSD-3](https://github.com/ethz-mrl/okvis2/blob/main/LICENSE)    |  `okvis2`  | ✅ | `radtan5` `equid4` |
| **----------** | **-------** | **-------** | **----------** | **--------** | **---** | **----------** |
| [**COLMAP**](https://colmap.github.io/)                                     |  SfM   |       `mono`       |                [BSD](https://colmap.github.io/license.html)                 |   `colmap`   | ✅ | `radtan5` `equid4` `unknown` |
| [**VGGT**](https://vgg-t.github.io/) |  SfM   |  `mono`  |  [VGGT](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)  |   `vggt`   | ➖ | `pinhole` |

<table>
<thead>
<tr><th>Label</th><th>Features</th><th>Summary</th><th>Modes</th><th>Camera Models</th></tr>
</thead>
<tbody>
<tr><td colspan="5">🏠 Indoor datasets<hr></td></tr>
<tr><td><code>7scenes</code></td><td>📸🏠🤳</td><td><a href="https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/"><strong>Tracked RGB-D Kinect frames from seven indoor scenes, for camera...</strong></a></td><td><code>mono</code> <code>rgbd</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>aria-digital-twin</code></td><td>📸🏠🥽</td><td><a href="https://www.projectaria.com/datasets/adt/"><strong>Aria Digital Twin (ADT) - egocentric dataset captured with Meta's Project Aria...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code> <code>rgbd(-vi)</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>eth</code></td><td>📸🏠🤳</td><td><a href="https://www.eth3d.net/"><strong>ETH3D SLAM & Stereo Benchmarks</strong></a></td><td><code>mono</code> <code>rgbd</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>monotum</code></td><td>📸🏠🏞️🤳</td><td><a href="https://cvg.cit.tum.de/data/datasets/mono-dataset"><strong>Photometrically calibrated monocular visual odometry benchmark with 50...</strong></a></td><td><code>mono</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>msd</code></td><td>📸🏠🥽</td><td><a href="https://huggingface.co/datasets/collabora/monado-slam-datasets"><strong>Monado SLAM Dataset - Valve Index</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>equid4</code></td></tr>
<tr><td><code>openloris-d400</code></td><td>📸🏠🤳</td><td><a href="https://lifelong-robotic-vision.github.io/dataset/scene.html"><strong>OpenLORIS-Scene - real-world lifelong SLAM dataset from real robots in real...</strong></a></td><td><code>mono(-vi)</code> <code>rgbd(-vi)</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>openloris-t265</code></td><td>📸🏠🤳</td><td><a href="https://lifelong-robotic-vision.github.io/dataset/scene.html"><strong>OpenLORIS-Scene - real-world lifelong SLAM dataset from real robots in real...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>equid4</code></td></tr>
<tr><td><code>rgbdtum</code></td><td>📸🏠🤳</td><td><a href="https://cvg.cit.tum.de/data/datasets/rgbd-dataset"><strong>RGB-D SLAM benchmark with motion-capture ground-truth trajectories</strong></a></td><td><code>mono</code> <code>rgbd</code></td><td><code>radtan5</code></td></tr>
<tr><td><code>rvp-handheld</code></td><td>📸🏞️🤳</td><td><a href="https://rvp-group.net/slam-dataset.html"><strong>VBR - A Vision Benchmark in Rome (hand-held sequences) - hardware-synced stereo...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>radtan4</code></td></tr>
<tr><td><code>vitum</code></td><td>📸🏠🏞️🤳</td><td><a href="https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset"><strong>TUM Visual-Inertial Dataset - stereo fisheye and IMU sequences from a handheld...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>equid4</code></td></tr>
<tr><td colspan="5">💻 Synthetic datasets<hr></td></tr>
<tr><td><code>drunkards</code></td><td>💻🏠🤳</td><td><a href="https://davidrecasens.github.io/TheDrunkard'sOdometry/"><strong>The Drunkard's Dataset - synthetic RGB-D exploratory camera trajectories with...</strong></a></td><td><code>mono</code> <code>rgbd</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>nuim</code></td><td>💻🏠🤳</td><td><a href="https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"><strong>ICL-NUIM RGB-D benchmark with ground-truth trajectories and 3D models for...</strong></a></td><td><code>mono</code> <code>rgbd</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>replica</code></td><td>💻🏠🤳</td><td><a href="https://github.com/facebookresearch/Replica-Dataset"><strong>Photorealistic 3D indoor scene reconstructions (Replica), with RGB-D SLAM...</strong></a></td><td><code>mono</code> <code>rgbd</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>tartanair</code></td><td>💻🏞️🤳</td><td><a href="https://theairlab.org/tartanair-dataset/"><strong>Large-scale photorealistic simulation dataset with multimodal ground truth for...</strong></a></td><td><code>mono</code></td><td><code>pinhole</code></td></tr>
<tr><td colspan="5">🚗 Vehicle datasets<hr></td></tr>
<tr><td><code>kitti</code></td><td>📸🏞️🚗</td><td><a href="https://www.cvlibs.net/datasets/kitti/"><strong>KITTI Odometry Benchmark with stereo image sequences and ground-truth poses for...</strong></a></td><td><code>mono</code> <code>stereo</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>nsavp</code></td><td>📸🏞️🚗</td><td><a href="https://umautobots.github.io/nsavp"><strong>Novel Sensors for Autonomous Vehicle Perception - stereo thermal, event,...</strong></a></td><td><code>mono</code> <code>stereo</code></td><td><code>radtan4</code></td></tr>
<tr><td><code>rvp-car</code></td><td>📸🏞️🚗</td><td><a href="https://rvp-group.net/slam-dataset.html"><strong>VBR - A Vision Benchmark in Rome (car sequences) - hardware-synced stereo...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>radtan4</code></td></tr>
<tr><td colspan="5">🚁 UAV datasets<hr></td></tr>
<tr><td><code>euroc</code></td><td>📸🏞️🚁</td><td><a href="https://projects.asl.ethz.ch/datasets/euroc-mav/"><strong>EuRoC MAV visual-inertial dataset with stereo images, IMU, and ground-truth...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>radtan4</code></td></tr>
<tr><td colspan="5">🤖 Robot datasets<hr></td></tr>
<tr><td><code>rover-d435i</code></td><td>📸🏞️🤖</td><td><a href="https://iis-esslingen.github.io/rover/"><strong>ROVER: A Multiseason Dataset for Visual SLAM</strong></a></td><td><code>mono</code> <code>rgbd(-vi)</code></td><td><code>radtan4</code></td></tr>
<tr><td><code>rover-picam</code></td><td>📸🏞️🤖</td><td><a href="https://iis-esslingen.github.io/rover/"><strong>ROVER: A Multiseason Dataset for Visual SLAM</strong></a></td><td><code>mono</code></td><td><code>equid4</code></td></tr>
<tr><td><code>rover-t265</code></td><td>📸🏞️🤖</td><td><a href="https://iis-esslingen.github.io/rover/"><strong>ROVER: A Multiseason Dataset for Visual SLAM</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>equid4</code></td></tr>
<tr><td><code>ut-coda</code></td><td>📸🏞️🤖</td><td><a href="https://amrl.cs.utexas.edu/coda/"><strong>The UT Campus Object Dataset (CODa) - egocentric robot perception dataset...</strong></a></td><td><code>mono</code> <code>stereo</code></td><td><code>radtan5</code></td></tr>
<tr><td><code>vector</code></td><td>📸🏠🤳🤖</td><td><a href="https://star-datasets.github.io/vector/"><strong>VECtor - a versatile event-centric benchmark for multi-sensor SLAM, with a...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>radtan4</code></td></tr>
<tr><td colspan="5">🏗️ Construction Site datasets<hr></td></tr>
<tr><td><code>hilti2022</code></td><td>📸🏠🏞️🏗️🤳</td><td><a href="https://hilti-challenge.com/dataset-2022"><strong>Hilti-Oxford SLAM Challenge 2022 - multi-camera, IMU and lidar dataset for SLAM...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>equid4</code></td></tr>
<tr><td><code>hilti2026</code></td><td>📸🏠🏞️🏗️🤳</td><td><a href="https://github.com/Hilti-Research/hilti-trimble-slam-challenge-2026"><strong>Hilti x Trimble 360 Visual-Inertial SLAM Challenge 2026 - construction-site...</strong></a></td><td><code>mono(-vi)</code></td><td><code>equid4</code></td></tr>
<tr><td colspan="5">🪐 Planetary-analog Terrain datasets<hr></td></tr>
<tr><td><code>madmax</code></td><td>📸🏞️🪐🤳</td><td><a href="https://datasets.arches-projekt.de/morocco2018/"><strong>The MADMAX data set for visual-inertial rover navigation on Mars</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>s3li-etna</code></td><td>📸🏞️🪐🤳</td><td><a href="https://www.dlr.de/en/rm/research/publications-and-downloads/datasets/s3li-datasets"><strong>S3LI Etna - the DLR Planetary Stereo, Solid-State LiDAR, Inertial dataset,...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>s3li-vulcano</code></td><td>📸🏞️🪐🤳</td><td><a href="https://www.dlr.de/en/rm/research/publications-and-downloads/datasets/s3li-datasets"><strong>S3LI Vulcano - a multi-modal (RGB stereo, solid-state LiDAR, inertial) dataset...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>pinhole</code></td></tr>
<tr><td colspan="5">🌊 Underwater datasets<hr></td></tr>
<tr><td><code>ariel</code></td><td>📸🏠🏞️🌊🤖</td><td><a href="https://huggingface.co/datasets/ntnu-arl/underwater-datasets"><strong>Multi-Camera Underwater Visual-Inertial Dataset - a BlueROV2-based platform...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>equid4</code></td></tr>
<tr><td><code>caves</code></td><td>📸🏞️🌊</td><td><a href="https://cirs.udg.edu/caves-dataset/"><strong>Underwater Caves Sonar and Vision Data Set - a diver-guided AUV exploration of...</strong></a></td><td><code>mono</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>eiffel-tower</code></td><td>📸🏞️🌊🤖</td><td><a href="https://www.seanoe.org/data/00810/92226/"><strong>Deep-sea hydrothermal vent imagery from four ROV Victor6000 survey campaigns...</strong></a></td><td><code>mono</code></td><td><code>radtan4</code></td></tr>
<tr><td><code>malaysia</code></td><td>📸🏞️🌊🤳</td><td><a href="N/A"><strong>Handheld underwater coral-reef transect surveys (Malaysia, July 2026 campaign),...</strong></a></td><td><code>mono</code></td><td><code>unknown</code></td></tr>
<tr><td><code>pamir</code></td><td>📸🌊🤳</td><td><a href="https://huggingface.co/datasets/afrl-uw/Pamir_Visual-Inertial_Data"><strong>Multi-session visual-inertial SLAM dataset of the Pamir shipwreck (Barbados),...</strong></a></td><td><code>mono(-vi)</code></td><td><code>unknown</code></td></tr>
<tr><td><code>pamir-rig</code></td><td>📸🌊🤳</td><td><a href="https://huggingface.co/datasets/afrl-uw/Pamir_Visual-Inertial_Data"><strong>Multi-session visual-inertial SLAM dataset of the Pamir shipwreck (Barbados),...</strong></a></td><td><code>mono(-vi)</code> <code>stereo(-vi)</code></td><td><code>unknown</code></td></tr>
<tr><td><code>sesoko</code></td><td>📸🏞️🌊</td><td><a href="https://squidle.org/"><strong>AUV (TUNA-SAND) coral-reef mapping surveys off Sesoko Island, Okinawa, Japan...</strong></a></td><td><code>mono</code></td><td><code>unknown</code></td></tr>
<tr><td><code>soneva</code></td><td>📸🏞️🌊🤳</td><td><a href="https://huggingface.co/datasets/wildflow/soneva-corals"><strong>Coral reef time-series photogrammetry (Maldives), from Soneva Conservation and...</strong></a></td><td><code>mono</code></td><td><code>pinhole</code></td></tr>
<tr><td><code>sweetcorals</code></td><td>📸🏞️🌊🤳</td><td><a href="https://huggingface.co/datasets/wildflow/sweet-corals"><strong>Coral reef 3D photogrammetry (Indonesia), from Wildflow</strong></a></td><td><code>mono</code></td><td><code>pinhole</code> <code>unknown</code></td></tr>
<tr><td colspan="5">🫀 Intracorporeal datasets<hr></td></tr>
<tr><td><code>hamlyn</code></td><td>📸🫀🤳</td><td><a href="https://davidrecasens.github.io/EndoDepthAndMotion/"><strong>Hamlyn rectified stereo endoscopy dataset with depth ground truth, from...</strong></a></td><td><code>mono</code> <code>rgbd</code> <code>stereo</code></td><td><code>pinhole</code></td></tr>
</tbody>
</table>


| Tools                                                                                                                        | Features |   Label    |      Modes       |  Camera Models     |
|:--------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:----------:|:----------:|
| [**Stray Scanner App**](https://github.com/strayrobots/scanner) |  📸🤳 | `strayscanner` | `mono` `rgbd` | `pinhole` |
<!-- | [**Underwater caves sonar and vision data set**](https://cirs.udg.edu/caves-dataset/)  |  📸🏞️🌊 |   `caves`  | `mono` | `pinhole` | -->
<!-- | [**ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes**](https://scannetpp.mlsg.cit.tum.de/scannetpp/) |  📸🏠🤳 | `scannetplusplus` | `mono`| `pinhole` | -->
<!-- [**Monocular Visual Odometry Dataset**](https://cvg.cit.tum.de/data/datasets/mono-dataset) | 📸🏠🤳 |  `monotum`  | `mono` | `pinhole` | -->

Real / Synthetic : 📸 / 💻

Indoor / Outdoor / Underwater / Intracorporeal : 🏠 / 🏞️ /  🌊 / 🫀

Handheld / Headmounted / Vehicle / UAV  / Robot : 🤳 / 🥽 / 🚗 / 🚁 / 🤖

Section dividers only (not per-row features) — Construction Site / Planetary-analog Terrain : 🏗️ / 🪐

## VSLAM-LAB  Roadmap
### Baselines
- [ ] Extend `orbslam3` and `orbslam3-dev` to `rgbd-vi`
- [ ] Extend `okvis2` and `okvis2-dev` to `rgbd-vi` and `stereo-vi`

### Datasets
- [ ] Implement `monotum`

<!--
## VSLAM-LAB v1.0 Roadmap

### Core
- [ ] Build system set up (CMake + options for CUDA/CPU)
- [ ] Docker dev image (CUDA + ROS optional)
- [ ] Pre-commit hooks (clang-format, clang-tidy, black/isort if Python)
- [ ] Licensing & citation (LICENSE + CITATION.cff + BibTeX snippet)
- [ ] Example dataset download script (`scripts/get_data.sh`)

### Datasets
- [ ] KITTI extension to `stereo`
- [ ] ROVER extension to `stereo`, `mono-vi`, `stereo-vi`
- [ ] TartanAir extension to `stereo`
- [ ] EuRoC extension to `stereo-vi`
- [ ] monotum re-implement `mono`
- [ ] 7scenes re-implement `mono`, `rgbd`
- [ ] drunkards re-implement `mono`, `rgbd`
- [ ] hamlyn re-implement mono `mono`
- [ ] caves re-implement `mono`
- [ ] hilti2022 re-implement `mono`
- [ ] scannetplusplus re-implement `mono`
- [ ] ariel re-implement `mono`
- [ ] lamar implement `mono`
- [ ] squidle implement `mono`
- [ ] openloris re-implement `mono`
- [ ] madmax implement `mono`, `rgbd`, `stereo`, `mono-vi`, `stereo-vi`
- [ ] sweetcorals implement `mono`
- [ ] reefslam implement `mono`
- [ ] ...

### Baselines
- [ ] AnyFeature VSLAM implement `mono`, `rgbd`, `stereo`
- [ ] DSO VSLAM implement `mono`
- [ ] MonoGS re-implement `mono`, `rgbd`
- [ ] VGGT implement SfM
- [ ] ORBSLAM3 implement `mono`, `rgbd`, `stereo`, `stereo-vi`, `rgbd-vi`
- [ ] OKVIS2 implement `mono`, `stereo-vi`
- [ ] pyCuVSLAM implement `mono`, `rgbd`, `stereo`, `mono-vi`, `stereo-vi`

### Metrics
- [ ] Include RPE
- [ ] Link metrics with modes

### Tooling
- [ ] Ablation tools
- [ ] ROS

### Docs
- [ ] README quickstart (build, run, datasets)
- [ ] Config reference (YAML/TOML)
- [ ] Architecture diagram
- [ ] Contributing guide

### Demos
- [ ] Example video/gif of live run

### Project Management
- [ ] Define statuses: Backlog → In Progress → Review → Done
- [ ] Convert key items above to sub-issues
-->
