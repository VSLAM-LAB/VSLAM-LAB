import subprocess
import os, shutil
import pandas as pd
from tqdm import tqdm

from Evaluate.evo_functions import evo_metric, evo_get_accuracy
from path_constants import VSLAM_LAB_EVALUATION_FOLDER, TRAJECTORY_FILE_NAME
from utilities import print_msg, ws, format_msg

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def evaluate_sequence(exp, dataset, sequence_name, overwrite=False):
    command =  "pixi run -e vslamlab evo_config set save_traj_in_zip true"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    METRIC = 'ate'
    
    trajectories_path = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
    evaluation_folder = os.path.join(trajectories_path, VSLAM_LAB_EVALUATION_FOLDER)
    accuracy_csv = os.path.join(evaluation_folder, f'{METRIC}.csv')

    # Load experiments log
    exp_log = pd.read_csv(exp.log_csv)
    if overwrite:
        if os.path.exists(evaluation_folder):
            shutil.rmtree(evaluation_folder)        
        exp_log.loc[exp_log["sequence_name"] == sequence_name, "EVALUATION"] = "none"

    os.makedirs(evaluation_folder, exist_ok=True)

    # Find runs to evaluate
    runs_to_evaluate = []
    for _, row in exp_log.iterrows():
        if row["SUCCESS"] and (row["EVALUATION"] == 'none') and (row["sequence_name"] == sequence_name):
            exp_it = str(row["exp_it"]).zfill(5) 
            runs_to_evaluate.append(exp_it)

    print_msg(SCRIPT_LABEL, f"Evaluating '{evaluation_folder.replace(sequence_name, f"{dataset.dataset_color}{sequence_name}\033[0m")}'")
    if len(runs_to_evaluate) == 0:
        exp_log.to_csv(exp.log_csv, index=False)
        return
    
    # Evaluate runs
    zip_files = []
    for exp_it in tqdm(runs_to_evaluate):
        trajectory_file = os.path.join(trajectories_path, f"{exp_it}_{TRAJECTORY_FILE_NAME}.txt")
        success = evo_metric('ate', groundtruth_txt, trajectory_file, evaluation_folder, 1.0 / dataset.rgb_hz)
        if success[0]:
            zip_files.append(os.path.join(evaluation_folder, f"{exp_it}_{TRAJECTORY_FILE_NAME}.zip"))
        else:
            exp_log.loc[(exp_log["exp_it"] == int(exp_it)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
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
        if exp_log.loc[(exp_log["exp_it"] == int(exp_it)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"].any() == 'failed':
            continue
        trajectory_file = f"{evaluated_run}_{TRAJECTORY_FILE_NAME}.txt"
        exists = (accuracy["traj_name"] == trajectory_file).any()
        if exists:
            run_mask = (exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name)
            exp_log.loc[run_mask, "EVALUATION"] = METRIC

            # Find number of frames in the sequence
            rgb_exp_txt = os.path.join(trajectories_path, f"rgb_exp.txt")
            with open(rgb_exp_txt, "r") as file:
                num_frames = sum(1 for _ in file)
            accuracy.loc[accuracy["traj_name"] == trajectory_file,"num_frames"] = num_frames
            exp_log.loc[run_mask, "num_frames"] = num_frames

            # Find number of tracked frames
            trajectory_file_txt = os.path.join(trajectories_path, trajectory_file)
            if not os.path.exists(trajectory_file_txt):
                exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
                continue
            with open(trajectory_file_txt, "r") as file:
                num_tracked_frames = sum(1 for _ in file)
            accuracy.loc[accuracy["traj_name"] == trajectory_file,"num_tracked_frames"] = num_tracked_frames    
            exp_log.loc[run_mask, "num_tracked_frames"] = num_tracked_frames

            # Find number of evaluated frames
            trajectory_file_tum = os.path.join(trajectories_path,VSLAM_LAB_EVALUATION_FOLDER, trajectory_file.replace(".txt", ".tum"))
            if not os.path.exists(trajectory_file_tum):
                exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'
                continue
            with open(trajectory_file_tum, "r") as file:
                num_evaluated_frames = sum(1 for _ in file) - 1
            accuracy.loc[accuracy["traj_name"] == trajectory_file,"num_evaluated_frames"] = num_evaluated_frames   
            exp_log.loc[run_mask, "num_evaluated_frames"] = num_evaluated_frames 
        else:
            exp_log.loc[(exp_log["exp_it"] == int(evaluated_run)) & (exp_log["sequence_name"] == sequence_name),"EVALUATION"] = 'failed'

    exp_log.to_csv(exp.log_csv, index=False)
    accuracy.to_csv(accuracy_csv, index=False)

