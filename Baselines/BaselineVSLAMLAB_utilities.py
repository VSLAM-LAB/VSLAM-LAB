import csv
import os

SCRIPT_LABEL = "[baseline_utilities.py] "

def initialize_log_run_sequence_time(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['experiment_id', 'runtime'])  # Write the header


def append_to_log_run_sequence_time(csv_path, experiment_id, run_time):
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([experiment_id, run_time])


def log_run_sequence_time(exp_folder, experiment_id, run_time):
    csv_path = os.path.join(exp_folder, 'log_run_sequence_time.csv')
    initialize_log_run_sequence_time(csv_path)
    append_to_log_run_sequence_time(csv_path, experiment_id, run_time)


def append_ablation_parameters_to_csv(file_path, parameter_dict):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = parameter_dict.keys()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(parameter_dict)
