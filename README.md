<p align="center">
<div align="center">
    <img src="docs/vslamlab_header.png" width="450"/>
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
git clone https://github.com/alejandrofontan/VSLAM-LAB.git && cd VSLAM-LAB
```

## Quick Demo
You can now execute any baseline on any sequence from any dataset within VSLAM-LAB using the following command:
```
pixi run demo <baseline> <dataset> <sequence>
```
For a full list of available systems and datasets, see the [VSLAM-LAB Supported Baselines and Datasets](#vslam-lab-supported-baselines-and-datasets).
Example commands:
```
pixi run demo mast3rslam eth table_3
pixi run demo droidslam euroc MH_01_easy
pixi run demo orbslam2 rgbdtum rgbd_dataset_freiburg1_xyz
```
*To change the paths where VSLAM-LAB-Benchmark or/and VSLAM-LAB-Evaluation data are stored (for example, to /media/${USER}/data), use the following commands:*
```
pixi run set-benchmark-path /media/${USER}/data
pixi run set-evaluation-path /media/${USER}/data
```
## VSLAM-LAB Info Functions
```bash
pixi run baseline-info <baseline>
pixi run print-baselines
pixi run print-datasets
```

## Configure your own experiments
With **VSLAM-LAB**, you can easily design and configure experiments using a YAML file and run them with a single command.
To **run** the experiment demo, execute the following command:
```
pixi run vslamlab configs/exp_vslamlab.yaml (--overwrite)
```

Experiments in **VSLAM-LAB** are sequences of entries in a YAML file (see example **~/VSLAM-LAB/configs/exp_vslamlab.yaml**):
```
exp_vslamlab:
  Config: config_vslamlab.yaml  # YAML file containing the sequences to be run 
  NumRuns: 1                    # Maximum number of executions per sequence
  Parameters: {verbose: 1}      # Vector with parameters that will be input to the baseline executable 
  Module: droidslam             # droidslam/monogs/orbslam2/mast3rslam/dpvo/...                    
```
**Config** files are YAML files containing the list of sequences to be executed in the experiment (see example **~/VSLAM-LAB/configs/config_vslamlab.yaml**):
```
rgbdtum:
  - 'rgbd_dataset_freiburg1_xyz'
hamlyn:
  - 'rectified01'
7scenes:
  - 'chess_seq-01'
eth:
  - 'table_3'
euroc:
  - 'MH_01_easy'
monotum:
  - 'sequence_01'
```
For a full list of available VSLAM systems and datasets, refer to the section [VSLAM-LAB Supported Baselines and Datasets](#vslam-lab-supported-baselines-and-datasets).

## VSLAM-LAB Pipeline Functions
Instead of running the full VSLAM-LAB pipeline, you can interact with datasets and baselines using the commands below:

```bash
pixi run validate-experiment-yaml <exp_yaml>             # Example: pixi run validate-experiment-yaml configs/exp_vslamlab.yaml
pixi run overwrite-exp <exp_yaml>                        # Example: pixi run overwrite-exp configs/exp_vslamlab.yaml
pixi run update-experiment-csv-logs <exp_yaml>           # Example: pixi run update-experiment-csv-logs configs/exp_vslamlab.yaml

pixi run check-experiment-resources <exp_yaml>           # Example: pixi run check-experiment-resources configs/exp_vslamlab.yaml
pixi run get-experiment-resources <exp_yaml>             # Example: pixi run get-experiment-resources configs/exp_vslamlab.yaml

pixi run check-experiment-state <exp_yaml>               # Example: pixi run check-experiment-state configs/exp_vslamlab.yaml

pixi run install-baseline <baseline>                     # Example: pixi run install-baseline droidslam
pixi run install-baselines <baseline1> <baseline2> ...   # Example: pixi run install-baselines droidslam orbslam2

pixi run download-sequence <dataset> <sequence>          # Example: pixi run download-sequence eth table_3
pixi run download-sequences <dataset1> <sequence1> <dataset2> <sequence2> ... \
                                                         # Example: pixi run download-sequences eth table_3 rgbdtum rgbd_dataset_freiburg1_xyz
pixi run download-dataset <dataset>                      # Example: pixi run download-dataset eth
pixi run download-datasets <dataset1> <dataset2>         # Example: pixi run download-datasets eth rgbdtum

pixi run run-exp <exp_yaml>                              # Example: pixi run run-exp configs/exp_vslamlab.yaml
pixi run evaluate-exp <exp_yaml>                         # Example: pixi run evaluate-exp configs/exp_vslamlab.yaml
pixi run compare-exp <exp_yaml>                          # Example: pixi run compare-exp configs/exp_vslamlab.yaml

```

## Add a new Dataset

Datasets in **VSLAM-LAB** are stored in a folder named **VSLAM-LAB-Benchmark**, which is created by default in the same parent directory as **VSLAM-LAB**.

1. To add a new dataset, structure your dataset as follows:
```
~/VSLAM-LAB-Benchmark
└── YOUR_DATASET
    └── sequence_01
        ├── rgb_0
            └── img_01
            └── img_02
            └── ...
        ├── calibration.yaml
        ├── rgb.csv
        └── groundtruth.csv
    └── sequence_02
        ├── ...
    └── ...   
```

2. Derive a new class **dataset_{your_dataset}.py** for your dataset from  **~/VSLAM-LAB/Datasets/Dataset_vslamlab.py**, and create a corresponding YAML configuration file named **dataset_{your_dataset}.yaml**.
	
3. Include the call for your dataset in function *def get_dataset(...)* in **~/VSLAM-LAB/Datasets/get_dataset.py**
```
 from Datasets.dataset_{your_dataset} import {YOUR_DATASET}_dataset
    ...
 def get_dataset(dataset_name, benchmark_path)
    ...
    switcher = {
        "rgbdtum": lambda: RGBDTUM_dataset(benchmark_path),
        ...
        "dataset_{your_dataset}": lambda: {YOUR_DATASET}_dataset(benchmark_path),
    }
```

## Add a new Baseline

Baselines  in **VSLAM-LAB** are stored in a folder named **VSLAM-LAB-Benchmark**, which is created by default in the same parent directory as **VSLAM-LAB**

1. Start modifying the pixi.toml (more information in ) (check ...):
```
[environments]
vggtslam-dev = { features = ["vggtslam-dev", "cuda126", "py11", "vggt"], solve-group = "vggt" }

...

# ########################################################################################################################
# ########################################################################################################################
# vggtslam-dev
[feature.vggtslam-dev]
channels = ["https://prefix.dev/conda-forge"]

platforms = ["linux-64"]

[feature.vggtslam-dev.tasks]
git-clone = "./git-clone.sh VSLAM-LAB VGGT-SLAM-VSLAM-LAB VGGT-SLAM-DEV"
install = {cmd = './setup.sh', cwd = 'Baselines/VGGT-SLAM-DEV', depends-on = ['git-clone']}
execute-mono = { cmd = 'python3 main.py --image_folder office_loop --max_loops 1 --vis_map', cwd = "Baselines/VGGT-SLAM-DEV" }

[feature.vggtslam-dev.dependencies]
compilers = "*"
cmake = "*"
make = "*"
...
```

2. Derive a new class **baseline_{your_baseline}.py** from  **~/VSLAM-LAB/Baselines/BaselineVSLAMLab.py**. (check ...):

3. Include the call for your baseline in function *def get_baseline(...)* in **~/VSLAM-LAB/Baselines/get_baseline.py** (check ...):
```
 from Baselines.baseline_files.baseline_{your_baseline} import {YOUR_BASELINE}_dataset
    ...
 def get_baseline_switcher():
    ...
    switcher = {
        "droidslam": lambda: DROIDSLAM_baseline(),
        ...
        "{your_baseline}": lambda: {YOUR_BASELINE}_baseline()
    }
```


## License
**VSLAM-LAB** is released under a **LICENSE.txt**. For a list of code dependencies which are not property of the authors of **VSLAM-LAB**, please check **docs/Dependencies.md**.


## Citation
If you're using **VSLAM-LAB** in your research, please cite. If you're specifically using VSLAM systems or datasets that have been included, please cite those as well. We provide a [spreadsheet](https://docs.google.com/spreadsheets/d/1V8_TLqlccipJ6x_TXkgLsw9zWszHU9M-0mGgDT92TEs/edit?usp=drive_link) with citation for each dataset and VSLAM system for your convenience.
```bibtex
@article{fontan2025vslam,
  title={VSLAM-LAB: A Comprehensive Framework for Visual SLAM Methods and Datasets},
  author={Fontan, Alejandro and Fischer, Tobias and Civera, Javier and Milford, Michael},
  journal={arXiv preprint arXiv:2504.04457},
  year={2025}
}
```

<!-- ## Acknowledgements

To [awesome-slam-datasets](https://github.com/youngguncho/awesome-slam-datasets) -->

# VSLAM-LAB Supported Baselines and Datasets
We provide a [spreadsheet](https://docs.google.com/spreadsheets/d/1V8_TLqlccipJ6x_TXkgLsw9zWszHU9M-0mGgDT92TEs/edit?usp=drive_link) with more detailed information for each baseline and dataset.

| Baselines                                                                   | System |     Sensors      |                                   License                                   |    Label     |  Conda Pkg     |  Camera Models     |  
|:----------------------------------------------------------------------------|:------:|:------:|:----------------:|:---------------------------------------------------------------------------:|:------------:|:------------:|
| [**MASt3R-SLAM**](https://github.com/rmurai0610/MASt3R-SLAM)                | VSLAM  |       `mono`       |    [CC BY-NC-SA 4.0](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)    | `mast3rslam`  | ✅ | `radtan5` `unknown` |
| [**DPVO**](https://github.com/princeton-vl/DPVO)                            | VSLAM  |       `mono`       |    [License](https://github.com/princeton-vl/DPVO/blob/main/LICENSE)    | `dpvo`  | ✅ | `radtan5` |
| [**DROID-SLAM**](https://github.com/princeton-vl/DROID-SLAM)                | VSLAM  |`mono` `rgbd` `stereo`|    [BSD-3](https://github.com/princeton-vl/DROID-SLAM/blob/main/LICENSE)    | `droidslam`  | ✅ | `radtan5` |
| [**ORB-SLAM2**](https://github.com/alejandrofontan/ORB_SLAM2)               | VSLAM  |`mono` `rgbd` `stereo`| [GPLv3](https://github.com/raulmur/ORB_SLAM2/blob/master/LICENSE.txt)|  `orbslam2`  | ✅ | `radtan5`  |
| [**MonoGS**](https://github.com/muskie82/MonoGS)                            | VSLAM  | `mono` `rgbd` |     [License](https://github.com/muskie82/MonoGS?tab=License-1-ov-file)     |   `monogs`   | ✅ | `radtan5` |
| [**AnyFeature-VSLAM**](https://github.com/alejandrofontan/AnyFeature-VSLAM) | VSLAM  | `mono` | [GPLv3](https://github.com/alejandrofontan/VSLAM-LAB/blob/main/LICENSE.txt) | `anyfeature` | ✅ |  `radtan5` |
| **----------** | **-------** | **-------** | **----------** | **--------** | **---** | **----------** |
| [**PyCuVSLAM**](https://github.com/VSLAM-LAB/PyCuVSLAM/tree/main) | VSLAM  |`mono` `rgbd` `stereo` `stereo-vi`| [NVIDIA](https://github.com/VSLAM-LAB/PyCuVSLAM/blob/main/LICENSE) |  `pycuvslam`  | ➖ | `radtan5` `equid4` |
| [**ORB-SLAM3**](https://github.com/UZ-SLAMLab/ORB_SLAM3)               | VSLAM  | `mono` `rgbd` `stereo` `mono-vi` `stereo-vi` |    [GPLv3](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/LICENSE)    |  `orbslam3`  | ✅ | `radtan5` `equid4`|
| [**OKVIS2**](https://github.com/ethz-mrl/okvis2)               | VSLAM  | `mono-vi` |    [BSD-3](https://github.com/ethz-mrl/okvis2/blob/main/LICENSE)    |  `okvis2`  | ✅ | `radtan5` `equid4` |
| **----------** | **-------** | **-------** | **----------** | **--------** | **---** | **----------** |
| [**GLOMAP**](https://lpanaf.github.io/eccv24_glomap/)                       |  SfM   |       `mono`       |         [BSD-3](https://github.com/colmap/glomap/blob/main/LICENSE)         |   `glomap`   | ✅ | `radtan5` `equid4` `unknown` |
| [**COLMAP**](https://colmap.github.io/)                                     |  SfM   |       `mono`       |                [BSD](https://colmap.github.io/license.html)                 |   `colmap`   | ✅ | `radtan5` `equid4` `unknown` |
| [**VGGT**](https://colmap.github.io/) |  SfM   |  `mono`  |  [VGGT](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)  |   `vggt`   | ➖ | `pinhole` |


| Datasets                                                                                                                        | Features |   Label    |     Sensors      |  Camera Models     |        
|:--------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:----------:|:----------:|
| [**ETH3D SLAM Benchmarks**](https://www.eth3d.net/slam_datasets)                                                                |  📸🏠🤳 |   `eth`    |`mono` `rgbd`| `pinhole` |
| [**RGB-D SLAM Dataset and Benchmark**](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)                                       |  📸🏠🤳 |  `rgbdtum`  |`mono` `rgbd`| `radtan5` |
| [**The KITTI Vision Benchmark Suite**](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)                                 |  📸🏞️🚗 |   `kitti`   |`mono` `stereo` | `pinhole` |
| [**The EuRoC MAV Dataset**](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)                       |  📸🏞️🚁 |   `euroc`   | `mono` `stereo`  `mono-vi` `stereo-vi` | `radtan4` |
| [**The Replica Dataset**](https://github.com/facebookresearch/Replica-Dataset) - [**iMAP**](https://edgarsucar.github.io/iMAP/) |  💻🏠🤳 |  `replica`  | `mono` `rgbd`  | `pinhole` |
| [**TartanAir: A Dataset to Push the Limits of Visual SLAM**](https://theairlab.org/tartanair-dataset/)                          |  💻🏞️🤳 | `tartanair` | `mono`  | `pinhole` |
| [**ICL-NUIM RGB-D Benchmark Dataset**](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)                                    |  💻🏠🤳 |   `nuim`    | `mono` `rgbd`  | `pinhole` | 
| [**Monado SLAM Dataset - Valve Index**](https://huggingface.co/datasets/collabora/monado-slam-datasets)                         |  📸🏠🥽 | `msd` | `mono` `stereo` `mono-vi` `stereo-vi` | `equid4` |
| [**ROVER: A Multiseason Dataset for Visual SLAM**](https://iis-esslingen.github.io/rover/)   | 📸🏞️🚗 | `rover_picam/d435i/t265` |`mono` `rgbd` `stereo` `mono-vi` `stereo-vi` | `radtan5` `equid4` |
| [**The UT Campus Object Dataset**](https://amrl.cs.utexas.edu/coda/) | 📸🏞️🤖 |  `ut_coda`  |`mono` `stereo`| `radtan5` |
| [**Sesoko campaign**](https://www.southampton.ac.uk/smmi/news/2017/06/20-southampton-tokyo-collaboration.page) | 📸🏞️🌊 |    `sesoko`    |`mono` | `pinhole` |

Real / Synthetic : 📸 / 💻 

Indoor / Outdoor : 🏠 / 🏞️ 

Handheld / Headmounted / Vehicle / UAV  / Robot / AUV :🤳 / 🥽 / 🚗 / 🚁 / 🤖 / 🌊


<!--| [**Monocular Visual Odometry Dataset**](https://cvg.cit.tum.de/data/datasets/mono-dataset)                                      |   real    |  handheld  |  `monotum`  | ⛔  | `Pinhole` |
| [**RGB-D Dataset 7-Scenes**](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)                          |   real    |  handheld  |  `7scenes`  | ⛔  | `Pinhole` |
| [**The Drunkard's Dataset**](https://davidrecasens.github.io/TheDrunkard%27sOdometry)                                           | synthetic |  handheld  | `drunkards` | ⛔  | `Pinhole` |
| [**Hamlyn Rectified Dataset**](https://davidrecasens.github.io/EndoDepthAndMotion/)                                             |   real    |  handheld  |  `hamlyn`   | ⛔  | `Pinhole` |
| [**Underwater caves sonar and vision data set**](https://cirs.udg.edu/caves-dataset/)                                           |   real    | underwater |   `caves`  | ⛔  | `Pinhole` |
| [**HILTI-OXFORD 2022**](http://hilti-challenge.com/dataset-2022.html)   |   real    | handheld |  `hilti2022`  | ⛔  | `Pinhole` |
-->

## VSLAM-LAB  Roadmap
### Baselines
- [ ] Extend `orbslam3` and `orbslam3-dev` to `rgbd-vi`
- [ ] Extend `okvis2` and `okvis2-dev` to `rgbd-vi` and `stereo-vi`

### Datasets
- [ ] Implement `monotum`
- [ ] Implement `7scenes`
- [ ] Implement `drunkards`
- [ ] Implement `hamlyn`
- [ ] Implement `caves`
- [ ] Implement `hilti2022`
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
- [ ] Link metrics with modalities

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