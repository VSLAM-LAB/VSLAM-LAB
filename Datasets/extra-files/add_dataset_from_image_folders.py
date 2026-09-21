import os
import sys
import shutil
import argparse
from PIL import Image
import yaml

sys.path.append(os.getcwd())

from path_constants import VSLAM_LAB_DIR
from path_constants import VSLAMLAB_BENCHMARK
from utilities import check_sequence_integrity
from utilities import ws

# This script integrates an EXISTING and COMPLIANT dataset into VSLAM-LAB by creating a new dataset class: dataset_{
# dataset_name}.py, generating a corresponding YAML configuration: dataset_{dataset_name}.yaml, and updating imports
# in: dataset_utilities.py.

def add_code_below_line(file_path, search_line, code_to_add):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if code_to_add in ''.join(lines):
        return

    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line)
            if search_line in line:
                file.write(code_to_add + '\n')


def replace_string_in_file(file_path, search_string, replace_string):

    with open(file_path, 'r') as file:
        file_data = file.read()

    new_file_data = file_data.replace(search_string, replace_string)

    with open(file_path, 'w') as file:
        file.write(new_file_data)


def main():
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--dataset_folder', type=str, required=True, help="Folder with the sequences")
    parser.add_argument('--dataset_name', type=str, required=False, help="Name of the dataset")
    parser.add_argument('--fps', type=float, default=10.0, help="fps")
    parser.add_argument('--resolution_scale', type=float, default=1.0, help=" [w,h] = [f x w0 , f x h0]")

    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    if args.dataset_name is None:
        dataset_name = str(dataset_folder).lower()
        DATASET_NAME = str(dataset_folder).upper()
    else:
        dataset_name = str(args.dataset_name).lower()
        DATASET_NAME = str(args.dataset_name).upper()

    # Verify the existence of the dataset folder
    dataset_path = os.path.join(VSLAMLAB_BENCHMARK, DATASET_NAME)
    if not os.path.exists(dataset_folder):
        print(f"Dataset \'{dataset_folder}\' does not exist!")
        exit(0)

    # Derive dataset class
    dataset_imagefolder_py = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extra-files', 'dataset_imagefolder.py')
    dataset_imagefolder_yaml = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extra-files', 'dataset_imagefolder.yaml')
    dataset_new_py = os.path.join(VSLAM_LAB_DIR, 'Datasets', f'dataset_{dataset_name}.py')
    dataset_new_yaml = os.path.join(VSLAM_LAB_DIR, 'Datasets', f'dataset_{dataset_name}.yaml')

    if not os.path.exists(dataset_new_py):
        shutil.copy(dataset_imagefolder_py, dataset_new_py)
    if not os.path.exists(dataset_new_yaml):
        shutil.copy(dataset_imagefolder_yaml, dataset_new_yaml)

    #
    with open(dataset_new_yaml, 'r') as file:
        data = yaml.safe_load(file)

    data['dataset_name'] = dataset_name
    data['resolution_scale'] = args.resolution_scale
    data['rgb_hz'] = args.fps
    data['dataset_folder_raw'] = dataset_folder
    data['sequence_names'] = []

    all_items = os.listdir(dataset_folder)
    sequences = [item for item in all_items if os.path.isdir(os.path.join(dataset_folder, item))]
    sequences.sort()
    for sequence in sequences:
        data['sequence_names'].append(sequence)

    with open(dataset_new_yaml, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    #
    replace_string_in_file(dataset_new_py, 'imagefolder', dataset_name)
    replace_string_in_file(dataset_new_py, 'IMAGEFOLDER', DATASET_NAME)

    #
    dataset_utilities_py = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'dataset_utilities.py')
    add_code_below_line(dataset_utilities_py, '# ADD your imports here',
                        f'from Datasets.dataset_{dataset_name} import {DATASET_NAME}_dataset')
    add_code_below_line(dataset_utilities_py, '# ADD your datasets here',
                        f'        "{dataset_name}": lambda: {DATASET_NAME}_dataset(),')
    #
    config_yaml = {}
    config_yaml[dataset_name] = data['sequence_names']
    with open(os.path.join(VSLAM_LAB_DIR, 'configs', f'config_{dataset_name}.yaml'), 'w') as yaml_file:
        yaml.dump(config_yaml, yaml_file, default_flow_style=False)

    #
    gitignore_file = os.path.join(VSLAM_LAB_DIR, '.gitignore')
    with open(gitignore_file, 'r') as file:
        existing_lines = file.read().splitlines()

    lines_to_add = [
        '',
        f'{os.path.join('Datasets', f'dataset_{dataset_name}.py')}',
        f'{os.path.join('Datasets', f'dataset_{dataset_name}.yaml')}',
        f'{os.path.join('configs', f'config_{dataset_name}.yaml')}'
    ]
    with open(gitignore_file, 'a') as file:
        for line in lines_to_add:
            if line not in existing_lines:
                file.write(line + '\n')


if __name__ == "__main__":
    main()
