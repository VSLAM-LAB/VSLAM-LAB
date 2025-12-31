# Run methods

import time
import csv
import shutil
from typing import Any
from pathlib import Path

from Baselines.BaselineVSLAMLab_utilities import log_run_sequence_time
from path_constants import RGB_BASE_FOLDER, VSLAMLAB_EVALUATION
from Run import ablations
from Run.downsample_rgb_frames import downsample_rgb_frames, get_rows

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

#@ray.remote(num_gpus=1)  
def run_sequence(exp_it, exp, baseline, dataset, sequence_name, ablation=False):
    # Check baseline is installed
    baseline.check_installation()

    run_time_start = time.time()

    # Create experiment folder
    exp_folder = exp.folder / dataset.dataset_folder / sequence_name
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Select images
    create_rgb_exp_csv(exp, dataset, sequence_name, baseline.default_parameters)

    # Sava data for evaluation
    get_sequence_data_for_evaluation(exp, dataset, sequence_name)

    # Build execution command
    exec_command = baseline.build_execute_command(exp_it, exp, dataset, sequence_name)

    # Prepare Ablation
    if ablation:
        exec_command = ablations.prepare_ablation(exp_it, exp, baseline, dataset, sequence_name, exec_command)

    # Execute experiment
    print(f"\n{SCRIPT_LABEL}Running (it {exp_it + 1}/{exp.num_runs}) {baseline.label} in {dataset.dataset_color}{sequence_name}\033[0m of {dataset.dataset_label} ...")
    results = baseline.execute(exec_command, exp_it, exp_folder)

    # Finish Ablation
    if ablation:
        ablations.finish_ablation(exp_it, baseline, dataset, sequence_name)

    # Log iteration duration
    duration_time = time.time() - run_time_start
    log_run_sequence_time(exp_folder, exp_it, duration_time)

    results['duration_time'] = duration_time
    return results

def create_rgb_exp_csv(exp, dataset, sequence_name, default_parameters = ""):
    sequence_path = dataset.dataset_path / sequence_name
    exp_folder = exp.folder / dataset.dataset_folder / sequence_name

    if 'rgb_csv' in exp.parameters:
        rgb_csv = sequence_path / exp.parameters['rgb_csv']
    else:
        rgb_csv = sequence_path / f"{RGB_BASE_FOLDER}.csv"

    rgb_exp_csv = exp_folder / f"{RGB_BASE_FOLDER}_exp.csv"

    if rgb_exp_csv.exists():
        rgb_exp_csv.unlink()
    shutil.copy(rgb_csv, rgb_exp_csv)

    rgb_idx = 'rgb_idx' in exp.parameters
    max_rgb = 'max_rgb' in exp.parameters or 'max_rgb' in default_parameters and not rgb_idx
       
    if max_rgb or rgb_idx:
        if max_rgb:
            max_rgb_num = exp.parameters['max_rgb'] if 'max_rgb' in exp.parameters else default_parameters['max_rgb']
            min_fps = dataset.rgb_hz / 10
            _, _, downsampled_rows = downsample_rgb_frames(rgb_csv, max_rgb_num, min_fps, True)

        if rgb_idx:
            downsampled_rows = get_rows(
                list(range(exp.parameters['rgb_idx'][0], exp.parameters['rgb_idx'][1] + 1)), rgb_csv)
        
        with open(rgb_exp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(downsampled_rows[0].keys()))
            writer.writeheader()
            writer.writerows(downsampled_rows)

def get_sequence_data_for_evaluation(exp: Any, dataset: Any, sequence_name: str) -> None:
    sequence_path = dataset.dataset_path /  sequence_name
    exp_folder = Path(exp.folder) / Path(dataset.dataset_folder) / sequence_name
    groundtruth_csv = sequence_path / 'groundtruth.csv'
    groundtruth_csv_dst = exp_folder / 'groundtruth.csv' 
    if not groundtruth_csv_dst.exists():
        shutil.copy(groundtruth_csv, groundtruth_csv_dst)

    rgb_folder = sequence_path / "rgb_0"
    first_image = next(rgb_folder.iterdir())
    thumbnails_folder =  VSLAMLAB_EVALUATION / "thumbnails"
    rgb_thumbnail = thumbnails_folder/ f"rgb_thumbnail_{dataset.dataset_name}_{sequence_name}{first_image.suffix}"
    thumbnails_folder.mkdir(parents=True, exist_ok=True)
    if not rgb_thumbnail.exists():
        shutil.copy(first_image, rgb_thumbnail)