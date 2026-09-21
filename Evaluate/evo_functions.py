import sys
import os
import shutil

sys.path.append(os.getcwd())
from tqdm import tqdm

import subprocess
import zipfile
import pandas as pd
import numpy as np
from utilities import find_files_with_string, read_trajectory_csv, save_trajectory_csv, read_trajectory_txt
from path_constants import ABLATION_PARAMETERS_CSV, TRAJECTORY_FILE_NAME

def evo_metric(metric, groundtruth_csv, trajectory_csv, evaluation_folder, max_time_difference=0.1):
    # Paths
    traj_file_name = os.path.basename(trajectory_csv).replace(".csv", "")
    traj_zip = os.path.join(evaluation_folder, f"{traj_file_name}.zip")
    traj_tum = os.path.join(evaluation_folder, f"{traj_file_name}.tum")
    gt_tum = traj_tum.replace(TRAJECTORY_FILE_NAME, "gt")
    traj_txt = os.path.join(evaluation_folder, f"{traj_file_name}.txt")
    gt_txt = os.path.join(evaluation_folder, f"groundtruth.txt")

    # Read trajectory.csv
    traj_df = read_trajectory_csv(trajectory_csv)
    if traj_df is None:
        return [False, f"Trajectory .csv is empty: {trajectory_csv}"]

    # Sort trajectory by timestamp
    trajectory_sorted = traj_df.sort_values(by=traj_df.columns[0])

    if not trajectory_sorted.equals(traj_df):
        save_trajectory_csv(trajectory_csv, trajectory_sorted)

    trajectory_sorted.to_csv(traj_txt, header=False, index=False, sep=' ', lineterminator='\n')

    # Read groundtruth.csv
    gt_df = read_trajectory_csv(groundtruth_csv)
    gt_df.to_csv(gt_txt, header=False, index=False, sep=' ', lineterminator='\n')

    # Evaluate
    if metric == 'ate':
        command = (f"evo_ape tum {gt_txt} {traj_txt} -va -as "
                   f"--t_max_diff {max_time_difference} --save_results {traj_zip}")
    if metric == 'rpe':
        command = f"evo_rpe tum {gt_txt} {traj_txt} --all_pairs --delta 5 -va -as --save_results {traj_zip}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, _ = process.communicate()

    if not os.path.exists(traj_zip):
        return [False, f"Zip file not created: {traj_zip}"]

    # Write aligned trajectory
    with zipfile.ZipFile(traj_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(traj_txt + '.tum'):
                with zip_ref.open(file_name) as source_file:
                    aligned_trajectory_file = os.path.join(evaluation_folder,
                        os.path.basename(file_name).replace(".txt", ""))
                    with open(aligned_trajectory_file, 'wb') as target_file:
                        target_file.write(source_file.read())
                break

    aligned_trajectory = read_trajectory_txt(aligned_trajectory_file)
    if aligned_trajectory is None:
        return [False, f"Aligned trajectory file is empty: {aligned_trajectory_file}"]
    aligned_trajectory.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    aligned_trajectory = aligned_trajectory.sort_values(by='ts')
    save_trajectory_csv(aligned_trajectory_file, aligned_trajectory, header=True)

    # Write aligned gt
    with zipfile.ZipFile(traj_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(gt_txt + '.tum'):
                with zip_ref.open(file_name) as source_file:
                    with open(gt_tum, 'wb') as target_file:
                        target_file.write(source_file.read())
                break

    aligned_gt = read_trajectory_txt(gt_tum)
    if aligned_gt is None:
        return [False, f"Aligned gt file is empty: {gt_tum}"]
    aligned_gt.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    aligned_gt = aligned_gt.sort_values(by='ts')
    save_trajectory_csv(gt_tum, aligned_gt, header=True)

    return [True, "Success"]

def evo_get_accuracy(zip_files, accuracy_csv):
    ZIP_CHUNK_SIZE = 500
    zip_files.sort()
    zip_files_chunks = [zip_files[i:i + ZIP_CHUNK_SIZE] for i in range(0, len(zip_files), ZIP_CHUNK_SIZE)]
    zip_files_chunks = [' '.join(str(file) for file in chunk) for chunk in zip_files_chunks]

    for zip_file_chunk in zip_files_chunks:
        if os.path.exists(accuracy_csv):
            existing_data = pd.read_csv(accuracy_csv)
            os.remove(accuracy_csv)
        else:
            existing_data = None

        command = (f"pixi run -e vslamlab evo_res {zip_file_chunk} --save_table {accuracy_csv}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        _, _ = process.communicate()

        if os.path.exists(accuracy_csv):
            new_data = pd.read_csv(accuracy_csv)
            new_data.columns.values[0] = "traj_name"
            new_columns = ['num_frames', 'num_tracked_frames', 'num_evaluated_frames']
            for col in new_columns:
                new_data[col] = 0

            if existing_data is not None:
                new_data = pd.concat([existing_data, new_data], ignore_index=True)
            new_data.to_csv(accuracy_csv, index=False)
        else:
            if existing_data is not None:
                existing_data.to_csv(accuracy_csv, index=False)

    for zip_file in zip_files:
      os.remove(zip_file)