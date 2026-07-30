# Run methods

import time
import shutil
import pandas as pd
from typing import Any
from pathlib import Path

from Baselines.BaselineVSLAMLAB_utilities import log_run_sequence_time
from path_constants import RGB_BASE_FOLDER, VSLAMLAB_EVALUATION
from Run import ablations
from utilities import print_msg, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

def get_rows(rows_idx, rgb_csv):
    df = pd.read_csv(Path(rgb_csv))

    idx = [int(i) for i in rows_idx if isinstance(i, (int,)) or str(i).lstrip("-").isdigit()]
    idx = [i for i in idx if 0 <= i < len(df)]

    return df.iloc[idx].to_dict(orient="records")

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

def create_rgb_exp_csv(exp: Any, dataset: Any, sequence_name: str, default_parameters: dict | None = None) -> None:
    """Build the experiment's rgb csv, applying rgb_idx/rgb_step/rgb_max filtering if requested."""
    sequence_path = dataset.sequence_path(sequence_name)
    exp_folder = exp.folder / dataset.dataset_folder / sequence_name

    # Seed rgb_exp_csv with a full copy of the source csv; if no downsampling/slicing is requested below, this copy is the final result
    if 'rgb_csv' in exp.parameters:
        rgb_csv = sequence_path / exp.parameters['rgb_csv']
    else:
        rgb_csv = dataset.rgb_csv_path(sequence_name)

    rgb_exp_csv = exp_folder / f"{RGB_BASE_FOLDER}_exp.csv"

    if rgb_exp_csv.exists():
        rgb_exp_csv.unlink()
    shutil.copy(rgb_csv, rgb_exp_csv)

    # rgb_idx, rgb_step, rgb_max are independent and stackable: idx slices the frame range first, step then keeps 1 in every n of what's left, and max truncates the remainder to a frame count
    has_default = isinstance(default_parameters, dict)
    has_rgb_idx = 'rgb_idx' in exp.parameters or (has_default and 'rgb_idx' in default_parameters)
    has_rgb_step = 'rgb_step' in exp.parameters or (has_default and 'rgb_step' in default_parameters)
    has_rgb_max = 'rgb_max' in exp.parameters or (has_default and 'rgb_max' in default_parameters)

    if has_rgb_idx or has_rgb_step or has_rgb_max:
        filter_info = []

        if has_rgb_idx:
            rgb_idx = exp.parameters['rgb_idx'] if 'rgb_idx' in exp.parameters else default_parameters['rgb_idx']
            total_frames = len(pd.read_csv(rgb_csv))
            start, end = rgb_idx[0], rgb_idx[1]
            if 0 <= start <= end < total_frames:
                rows = get_rows(list(range(start, end + 1)), rgb_csv)
            else:
                print_msg(SCRIPT_LABEL, f"rgb_idx {rgb_idx} invalid for {sequence_name} (valid range 0-{total_frames - 1}); using the whole sequence instead", flag="error", verb='NONE')
                rows = pd.read_csv(rgb_csv).to_dict(orient="records")
                has_rgb_idx = False
        else:
            rows = pd.read_csv(rgb_csv).to_dict(orient="records")
        filter_info.append(f"rgb_idx={[start, end] if has_rgb_idx else 'all'} -> {len(rows)} frames")

        if has_rgb_step:
            rgb_step = exp.parameters['rgb_step'] if 'rgb_step' in exp.parameters else default_parameters['rgb_step']
            rows = rows[::rgb_step]
            filter_info.append(f"rgb_step={rgb_step} -> {len(rows)} frames")

        if has_rgb_max:
            max_rgb_num = exp.parameters['rgb_max'] if 'rgb_max' in exp.parameters else default_parameters['rgb_max']
            rows = rows[:max_rgb_num]
            filter_info.append(f"rgb_max={max_rgb_num} -> {len(rows)} frames")

        header = list(rows[0].keys())
        write_csv_rows(rgb_exp_csv, header, [list(row.values()) for row in rows])
        print_msg(SCRIPT_LABEL, f"RGB frame filtering for {sequence_name}: {' | '.join(filter_info)}", verb='LOW')

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