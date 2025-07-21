<p align="center">

<h1 align="center">
  <img src="docs/logo.ico" alt="VSLAM-LAB Logo" width="30" 
       style="vertical-align: middle; position: relative; top: -10px; margin-right: 10px;">
  <span style="font-size: 2em; font-weight: bold;">VSLAM-LAB</span>
</h1>

<!-- <h1 align="center">
  <span style="font-size: 2em; font-weight: bold;">VSLAM-LAB</span>
  <img src="docs/logo.png" alt="VSLAM-LAB Logo" width="30" 
       style="vertical-align: middle; position: relative; top: -10px; margin-left: 10px;">
</h1> -->

<!-- <h1 align="center">
  <img src="docs/logo.png" alt="VSLAM-LAB Logo" width="42" 
       style="vertical-align: middle; position: relative; top: -10px; margin-right: 10px;">
  <span style="font-size: 2em; font-weight: bold;">VSLAM-LAB</span>
</p> -->

  <h3 align="center"> A Comprehensive Framework for Visual SLAM Baselines and Datasets</h3> 
  </h1>
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
    <a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en"><strong>Javier Civera</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TDSmCKgAAAAJ&hl=en"><strong>Michael Milford</strong></a>
  </p>

**VSLAM-LAB** is designed to simplify the development, evaluation, and application of Visual SLAM (VSLAM) systems. 
This framework enables users to compile and configure VSLAM systems, download and process datasets, and design, run, and
evaluate experiments — **all from a single command line**!

**Why Use VSLAM-LAB?**
- **Unified Framework:** Streamlines the management of VSLAM systems and datasets.
- **Ease of Use:** Run experiments with minimal configuration and single command executions.
- **Broad Compatibility:** Supports a wide range of VSLAM systems and datasets.
- **Reproducible Results:** Standardized methods for evaluating and analyzing results.

## Getting Started

To ensure all dependencies are installed in a reproducible manner, we use the package management tool [**pixi**](https://pixi.sh/latest/). If you haven't installed [**pixi**](https://pixi.sh/latest/) yet, please run the following command in your terminal:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
*After installation, restart your terminal or source your shell for the changes to take effect*. For more details, refer to the [**pixi documentation**](https://pixi.sh/latest/).

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
pixi run demo dpvo 7scenes chess_seq-01
pixi run demo monogs hamlyn rectified01
```
*To change the paths where VSLAM-LAB-Benchmark or/and VSLAM-LAB-Evaluation data are stored (for example, to /media/${USER}/data), use the following commands:*
```
pixi run set-benchmark-path /media/${USER}/data
pixi run set-evaluation-path /media/${USER}/data
```
## Configure your own experiments
With **VSLAM-LAB**, you can easily design and configure experiments using a YAML file and run them with a single command.
To **run** the experiment demo, execute the following command:
```
ARGUMENT="--exp_yaml exp_mono.yaml" pixi run vslamlab
```

Experiments in **VSLAM-LAB** are sequences of entries in a YAML file (see example **~/VSLAM-LAB/configs/exp_mono.yaml**):
```
exp_vslamlab:
  Config: config_mono.yaml     # YAML file containing the sequences to be run 
  NumRuns: 1                   # Maximum number of executions per sequence
  Parameters: {verbose: 1}     # Vector with parameters that will be input to the baseline executable 
  Module: droidslam            # droidslam/monogs/orbslam2/mast3rslam/dpvo/...                    
```
**Config** files are YAML files containing the list of sequences to be executed in the experiment (see example **~/VSLAM-LAB/configs/config_mono.yaml**):
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

## Add a new dataset

Datasets in **VSLAM-LAB** are stored in a folder named **VSLAM-LAB-Benchmark**, which is created by default in the same parent directory as **VSLAM-LAB**. If you want to modify the location of your datasets, change the variable **VSLAMLAB_BENCHMARK** in **~/VSLAM-LAB/utilities.py**.

1. To add a new dataset, structure your dataset as follows:
```
~/VSLAM-LAB-Benchmark
└── YOUR_DATASET
    └── sequence_01
        ├── rgb
            └── img_01
            └── img_02
            └── ...
        ├── calibration.yaml
        ├── rgb.txt
        └── groundtruth.txt
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
pixi.toml
```
kill_all 

header

dependencies

tasks

  git-clone

  execute
```

Baselines/get_baseline.py
```
# ADD your imports here
from Baselines.baseline_droidslam import DROIDSLAM_baseline
from Baselines.baseline_droidslam import DROIDSLAM_baseline_dev
...

def get_baseline(baseline_name):
    baseline_name = baseline_name.lower()
    switcher = {
        # ADD your baselines here
        "droidslam": lambda: DROIDSLAM_baseline(),
        "droidslam-dev": lambda: DROIDSLAM_baseline_dev(),
        ...
    }

    return switcher.get(baseline_name, lambda: "Invalid case")()
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

## Acknowledgements

To [awesome-slam-datasets](https://github.com/youngguncho/awesome-slam-datasets)

# VSLAM-LAB Supported Baselines and Datasets
We provide a [spreadsheet](https://docs.google.com/spreadsheets/d/1V8_TLqlccipJ6x_TXkgLsw9zWszHU9M-0mGgDT92TEs/edit?usp=drive_link) with more detailed information for each baseline and dataset.

| Baselines                                                                   | System |     Sensors      |                                   License                                   |    Label     |
|:----------------------------------------------------------------------------|:------:|:----------------:|:---------------------------------------------------------------------------:|:------------:|
| [**MASt3R-SLAM**](https://github.com/rmurai0610/MASt3R-SLAM)                | VSLAM  |       mono       |    [CC BY-NC-SA 4.0](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)    | `mast3rslam`  |
| [**DROID-SLAM**](https://github.com/princeton-vl/DROID-SLAM)                | VSLAM  |       mono       |    [BSD-3](https://github.com/princeton-vl/DROID-SLAM/blob/main/LICENSE)    | `droidslam`  |
| [**DPVO**](https://github.com/princeton-vl/DPVO)                            | VSLAM  |       mono       |    [License](https://github.com/princeton-vl/DPVO/blob/main/LICENSE)    | `dpvo`  |
| [**ORB-SLAM2**](https://github.com/alejandrofontan/ORB_SLAM2)               | VSLAM  | mono/RGBD/Stereo |    [GPLv3](https://github.com/raulmur/ORB_SLAM2/blob/master/LICENSE.txt)    |  `orbslam2`  | 
| [**MonoGS**](https://github.com/muskie82/MonoGS)                            | VSLAM  | mono/RGBD/Stereo |     [License](https://github.com/muskie82/MonoGS?tab=License-1-ov-file)     |   `monogs`   | 
| [**GLOMAP**](https://lpanaf.github.io/eccv24_glomap/)                       |  SfM   |       mono       |         [BSD-3](https://github.com/colmap/glomap/blob/main/LICENSE)         |   `glomap`   |
| [**DUST3R**](https://dust3r.europe.naverlabs.com)                           |  SfM   |       mono       |    [CC BY-NC-SA 4.0](https://github.com/naver/dust3r/blob/main/LICENSE)     |   `dust3r`   | 
| [**COLMAP**](https://colmap.github.io/)                                     |  SfM   |       mono       |                [BSD](https://colmap.github.io/license.html)                 |   `colmap`   | 
| [**GenSfM**](https://github.com/Ivonne320/GenSfM)                                     |  SfM   |       mono       |                [BSD](https://github.com/Ivonne320/GenSfM/blob/main/COPYING.txt)                 |   `gensfm`   | 
| [**DSO**](https://github.com/alejandrofontan/dso)                           |   VO   |       mono       |       [GPLv3](https://github.com/JakobEngel/dso/blob/master/LICENSE)        |    `dso`     |
| [**AnyFeature-VSLAM**](https://github.com/alejandrofontan/AnyFeature-VSLAM) | VSLAM  |       mono       | [GPLv3](https://github.com/alejandrofontan/VSLAM-LAB/blob/main/LICENSE.txt) | `anyfeature` |
| [**evo**](https://github.com/MichaelGrupp/evo)                       |  Eval  |        -         |    [GPLv3](https://github.com/MichaelGrupp/evo/blob/master/LICENSE)    |    `evo`     |



| Datasets                                                                                                                        |   Data    |    Mode    |    Label    |
|:--------------------------------------------------------------------------------------------------------------------------------|:---------:|:----------:|:-----------:|
| [**ETH3D SLAM Benchmarks**](https://www.eth3d.net/slam_datasets)                                                                |   real    |  handheld  |    `eth`    |
| [**RGB-D SLAM Dataset and Benchmark**](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)                                       |   real    |  handheld  |  `rgbdtum`  |
| [**ICL-NUIM RGB-D Benchmark Dataset**](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)                                    | synthetic |  handheld  |   `nuim`    | 
| [**Monocular Visual Odometry Dataset**](https://cvg.cit.tum.de/data/datasets/mono-dataset)                                      |   real    |  handheld  |  `monotum`  |
| [**The KITTI Vision Benchmark Suite**](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)                                 |   real    |  vehicle   |   `kitti`   |
| [**RGB-D Dataset 7-Scenes**](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)                          |   real    |  handheld  |  `7scenes`  |
| [**The EuRoC MAV Dataset**](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)                       |   real    |    UAV     |   `euroc`   |
| [**TartanAir: A Dataset to Push the Limits of Visual SLAM**](https://theairlab.org/tartanair-dataset/)                          | synthetic |  handheld  | `tartanair` |
| [**The Drunkard's Dataset**](https://davidrecasens.github.io/TheDrunkard%27sOdometry)                                           | synthetic |  handheld  | `drunkards` |
| [**The Replica Dataset**](https://github.com/facebookresearch/Replica-Dataset) - [**iMAP**](https://edgarsucar.github.io/iMAP/) | synthetic |  handheld  |  `replica`  |
| [**Hamlyn Rectified Dataset**](https://davidrecasens.github.io/EndoDepthAndMotion/)                                             |   real    |  handheld  |  `hamlyn`   |
| [**Underwater caves sonar and vision data set**](https://cirs.udg.edu/caves-dataset/)                                           |   real    | underwater |   `caves`   |
| [**HILTI-OXFORD 2022**](http://hilti-challenge.com/dataset-2022.html)   |   real    | handheld |  `hilti2022`  |
