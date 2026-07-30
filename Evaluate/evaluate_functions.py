import subprocess
import os, shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from Evaluate.evo_functions import evo_metric, evo_get_accuracy
from path_constants import VSLAM_LAB_EVALUATION_FOLDER, TRAJECTORY_FILE_NAME, GROUNTRUTH_FILE
from utilities import print_msg, ws, format_msg, read_csv

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def evaluate_sequence(exp, dataset, sequence_name, overwrite=False):
    sequence_name = str(sequence_name)

    # Enable evo to save trajectories in zip format
    # TODO : Remove this
    command =  "pixi run -e vslamlab evo_config set save_traj_in_zip true"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    METRIC = 'ate'
    
    # Define input paths
    trajectories_path = exp.folder / dataset.dataset_folder / sequence_name
    groundtruth_csv = exp.folder / dataset.dataset_folder / sequence_name /  GROUNTRUTH_FILE

    # Define output paths
    evaluation_folder = exp.folder / dataset.dataset_folder / sequence_name / VSLAM_LAB_EVALUATION_FOLDER
    accuracy_csv = evaluation_folder / f'{METRIC}.csv'

    # Load experiments log
    exp_log = read_csv(exp.log_csv)
    if overwrite:
        if evaluation_folder.exists():
            shutil.rmtree(evaluation_folder)
        exp_log.loc[(exp_log["dataset_name"] == dataset.dataset_name) & (exp_log["sequence_name"] == sequence_name), "EVALUATION"] = "none"

    evaluation_folder.mkdir(parents=True, exist_ok=True)

    # Find runs to evaluate
    runs_to_evaluate = []
    for _, row in exp_log.iterrows():
        if row["SUCCESS"] and (row["EVALUATION"] == 'none') and (row["dataset_name"] == dataset.dataset_name) and (row["sequence_name"] == sequence_name):
            exp_it = str(row["exp_it"]).zfill(5) 
            runs_to_evaluate.append(exp_it)

    print_msg(
        SCRIPT_LABEL,
        f"Evaluating '{str(evaluation_folder).replace(sequence_name, f'{dataset.dataset_color}{sequence_name}\033[0m')}'"
    )
    if len(runs_to_evaluate) == 0:
        exp_log.to_csv(exp.log_csv, index=False)
        return
    
    # Evaluate runs
    zip_files = []
    for exp_it in tqdm(runs_to_evaluate):
        trajectory_file = trajectories_path / f"{exp_it}_{TRAJECTORY_FILE_NAME}.csv"
        success = evo_metric(METRIC, groundtruth_csv, trajectory_file, evaluation_folder, 1e9 / dataset.rgb_hz)
        if success[0]:
            zip_files.append(evaluation_folder / f"{exp_it}_{TRAJECTORY_FILE_NAME}.zip")
        else:
            exp_log.loc[(exp_log["exp_it"] == int(exp_it)) & (exp_log["dataset_name"] == dataset.dataset_name) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
            tqdm.write(format_msg(ws(8), f"{success[1]}", "error"))
    if len(zip_files) == 0:
        exp_log.to_csv(exp.log_csv, index=False)
        return   
    
    # Retrieve accuracies
    evo_get_accuracy(zip_files, accuracy_csv)

    # Final Checks
    if not os.path.exists(accuracy_csv):
        exp_log.to_csv(exp.log_csv, index=False)
        return
    
    accuracy = pd.read_csv(accuracy_csv)
    for evaluated_run in runs_to_evaluate:
        if exp_log.loc[(exp_log["exp_it"] == int(exp_it)) & (exp_log["dataset_name"] == dataset.dataset_name) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"].any() == 'failed':
            continue
        trajectory_name_txt = f"{evaluated_run}_{TRAJECTORY_FILE_NAME}.txt"
        exists = (accuracy["traj_name"] == trajectory_name_txt).any()
        if exists:
            run_mask = (exp_log["exp_it"] == int(evaluated_run)) & (exp_log["dataset_name"] == dataset.dataset_name) & (exp_log["sequence_name"] == sequence_name)
            exp_log.loc[run_mask, "EVALUATION"] = METRIC

            # Find number of frames in the sequence
            rgb_exp_csv = trajectories_path / f"rgb_exp.csv"
            with open(rgb_exp_csv, "r") as file:
                num_frames = sum(1 for _ in file)-1
            accuracy.loc[accuracy["traj_name"] == trajectory_name_txt,"num_frames"] = num_frames
            exp_log.loc[run_mask, "num_frames"] = num_frames

            # Find number of tracked frames
            trajectory_file_txt = evaluation_folder / trajectory_name_txt
            if not trajectory_file_txt.exists():
                exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["dataset_name"] == dataset.dataset_name) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
                continue
            with open(trajectory_file_txt, "r") as file:
                num_tracked_frames = sum(1 for _ in file)
            accuracy.loc[accuracy["traj_name"] == trajectory_name_txt,"num_tracked_frames"] = num_tracked_frames    
            exp_log.loc[run_mask, "num_tracked_frames"] = num_tracked_frames

            # Find number of evaluated frames
            trajectory_file_tum = trajectories_path / VSLAM_LAB_EVALUATION_FOLDER / trajectory_name_txt.replace(".txt", ".tum")
            if not trajectory_file_tum.exists():
                exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["dataset_name"] == dataset.dataset_name) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
                continue
            with open(trajectory_file_tum, "r") as file:
                num_evaluated_frames = sum(1 for _ in file) - 1
            accuracy.loc[accuracy["traj_name"] == trajectory_name_txt,"num_evaluated_frames"] = num_evaluated_frames   
            exp_log.loc[run_mask, "num_evaluated_frames"] = num_evaluated_frames 
        else:
            exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["dataset_name"] == dataset.dataset_name) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'

    exp_log.to_csv(exp.log_csv, index=False)
    accuracy.to_csv(accuracy_csv, index=False)
