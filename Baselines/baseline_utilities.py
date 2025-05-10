import csv
import os

SCRIPT_LABEL = "[baseline_utilities.py] "

# ADD your imports here
from Baselines.baseline_droidslam import DROIDSLAM_baseline
from Baselines.baseline_droidslam import DROIDSLAM_baseline_dev
from Baselines.baseline_mast3rslam import MAST3RSLAM_baseline
from Baselines.baseline_mast3rslam import MAST3RSLAM_baseline_dev
from Baselines.baseline_orbslam2 import ORBSLAM2_baseline
from Baselines.baseline_orbslam2 import ORBSLAM2_baseline_dev

from Baselines.baseline_dpvo import DPVO_baseline
from Baselines.baseline_dpvo import DPVO_baseline_dev
from Baselines.baseline_monogs import MONOGS_baseline
from Baselines.baseline_monogs import MONOGS_baseline_dev

from Baselines.baseline_anyfeature import ANYFEATURE_baseline
from Baselines.baseline_dso import DSO_baseline
from Baselines.baseline_dust3r import DUST3R_baseline
from Baselines.baseline_colmap import COLMAP_baseline
from Baselines.baseline_glomap import GLOMAP_baseline
from Baselines.baseline_depthpro import DEPTHPRO_baseline

from Baselines.baseline_orbslam3 import ORBSLAM3_baseline

def get_baseline(baseline_name):
    baseline_name = baseline_name.lower()
    switcher = {
        # ADD your baselines here
        "droidslam": lambda: DROIDSLAM_baseline(),
        "droidslam-dev": lambda: DROIDSLAM_baseline_dev(),
        "mast3rslam": lambda: MAST3RSLAM_baseline(),
        "mast3rslam-dev": lambda: MAST3RSLAM_baseline_dev(),
        "orbslam2": lambda: ORBSLAM2_baseline(),
        "orbslam2-dev": lambda: ORBSLAM2_baseline_dev(),
        "dpvo": lambda: DPVO_baseline(),
        "dpvo-dev": lambda: DPVO_baseline_dev(),
        "monogs": lambda: MONOGS_baseline(),
        "monogs-dev": lambda: MONOGS_baseline_dev(),

        "anyfeature": lambda: ANYFEATURE_baseline(),
        "dso": lambda: DSO_baseline(),
        "dust3r": lambda: DUST3R_baseline(),
        "colmap": lambda: COLMAP_baseline(),
        "glomap": lambda: GLOMAP_baseline(),
        "depthpro": lambda: DEPTHPRO_baseline(),

        "orbslam3": lambda: ORBSLAM3_baseline()     
    }

    return switcher.get(baseline_name, lambda: "Invalid case")()


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
