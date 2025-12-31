import os
import cv2
import shutil
import numpy as np
import yaml
import inspect
import time
import pandas as pd

from path_constants import RGB_BASE_FOLDER
from path_constants import ABLATION_PARAMETERS_CSV

from utilities import ws
from Baselines.BaselineVSLAMLab_utilities import append_ablation_parameters_to_csv
from Datasets.DatasetVSLAMLab_utilities import load_rgb_csv

SCRIPT_LABEL = f"\033[35m[{os.path.basename(__file__)}]\033[0m "


def modify_yaml_parameter(settings_ablation_yaml, section_name, parameter_name, new_value):
    with open(settings_ablation_yaml, 'r') as file:
        data = yaml.safe_load(file)
    if section_name in data and parameter_name in data[section_name]:
        data[section_name][parameter_name] = new_value
    else:
        print(f"    Parameter '{parameter_name}' or section '{section_name}' not found in the YAML file.")
    with open(settings_ablation_yaml, 'w') as file:
         yaml.safe_dump(data, file)

def prepare_ablation(exp_it, exp, baseline, dataset, sequence_name, exec_command):
    print(f"\n{SCRIPT_LABEL}Sequence {dataset.dataset_color}{sequence_name}\033[0m preparing ablation: {exp.ablation_csv}")
    #print(f"\n{SCRIPT_LABEL}Running (it {exp_it + 1}/{exp.num_runs}) {baseline.label} in {dataset.dataset_color}{sequence_name}\033[0m of {dataset.dataset_label} ...")

    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    settings_yaml = baseline.settings_yaml
    ablation_parameters_csv = pd.read_csv(exp.ablation_csv)

    # Create new _settings_ablation.yaml file
    settings_ablation_yaml = os.path.join(exp_folder, os.path.basename(settings_yaml).replace('_settings', '_settings_ablation'))
    if os.path.exists(settings_ablation_yaml):
        os.remove(settings_ablation_yaml)
    shutil.copy(settings_yaml, settings_ablation_yaml)
    exec_command = exec_command.replace(settings_yaml, settings_ablation_yaml)

    # Update _settings_ablation.yaml file
    row = ablation_parameters_csv.iloc[exp_it]
    for parameter, value in row.items():
        if parameter == 'exp_it':
            continue
        if parameter == 'image_noise':
            print(f"{ws(8)}image_noise: std: {value}")
            add_noise_to_images_start(exp_it, exp, dataset, sequence_name, value)
            continue
        section_name, parameter_name = parameter.split('.', 1)
        modify_yaml_parameter(settings_ablation_yaml, section_name, parameter_name, value)
        print(f"{ws(8)}{section_name}: {parameter_name}: {value}")

    return exec_command


def finish_ablation(exp_it, baseline, dataset, sequence_name):
    print(f"{ws(8)}Sequence '{sequence_name}' finishing ablation ...")
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    #add_noise_to_images_finish(sequence_path, exp_it)



def add_noise_to_images_start(exp_it, exp, dataset, sequence_name, image_noise):
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    rgb_exp_txt = os.path.join(exp_folder, f"{RGB_BASE_FOLDER}_exp.txt")

    std_noise = image_noise
    
    if std_noise == 0.0:
        return 

    # Create rgb_path_ablation folder to store new images
    rgb_path_ablation = os.path.join(sequence_path, f"{RGB_BASE_FOLDER}_ablation")
    if os.path.exists(rgb_path_ablation):
        shutil.rmtree(rgb_path_ablation)
        #return
    os.makedirs(rgb_path_ablation, exist_ok=True)

    def add_gaussian_noise(image_, mean=0, std_dev=25):
        np.random.seed(int(time.time() * 1000) % (2**32))
        noise = np.random.normal(mean, std_dev, image_.shape).astype(np.float32)
        noisy_image_ = image_ + noise
        noisy_image_ = np.clip(noisy_image_, 0, 255).astype(np.uint8)
        return noisy_image_

    with open(rgb_exp_txt, 'r') as file:
        lines = file.readlines()

    rgb_paths, *_ = load_rgb_csv(rgb_exp_txt)
    with open(rgb_exp_txt, 'w') as file:
        for i, line in enumerate(lines):
            rgb_path = rgb_paths[i]
            file.write(line.replace(f"{RGB_BASE_FOLDER}/", f"{RGB_BASE_FOLDER}_ablation/"))

            rgb_file = os.path.join(sequence_path, rgb_path)
            image = cv2.imread(rgb_file)
            noisy_image = add_gaussian_noise(image, mean=0, std_dev=std_noise)

            rgb_ablation_path = rgb_path.replace(f"{RGB_BASE_FOLDER}/", f"{RGB_BASE_FOLDER}_ablation/")
            rgb_file_ablation = os.path.join(sequence_path, rgb_ablation_path)
            cv2.imwrite(os.path.join(sequence_path, rgb_file_ablation), noisy_image)
    return


def add_noise_to_images_finish(sequence_path, exp_it):
    #if not (exp_it % 100 == 0):
    #    return

    # Remove rgb folder
    rgb_path_ablation = os.path.join(sequence_path, f"{RGB_BASE_FOLDER}_ablation")
    if os.path.exists(rgb_path_ablation):
        shutil.rmtree(rgb_path_ablation)



