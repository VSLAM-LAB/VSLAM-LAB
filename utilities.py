import os, sys, yaml, re
import urllib.request
import zipfile
import py7zr
import tarfile
import subprocess 
from PIL import Image
from colorama import Fore, Style
import pandas as pd
from pathlib import Path

from path_constants import VSLAM_LAB_DIR, VSLAMLAB_VERBOSITY, VerbosityManager

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def check_parameter_for_relative_path(parameter_value):
    if "VSLAM-LAB" in parameter_value:
        if ":" in parameter_value:
            return re.sub(r'(?<=:)[^:]*VSLAM-LAB', VSLAM_LAB_DIR, str(parameter_value))
        return re.sub(r'^.*VSLAM-LAB', VSLAM_LAB_DIR, str(parameter_value))
    return parameter_value

def filter_inputs(args):
    if  not args.run and not args.evaluate and not args.compare:
        args.run = True
        args.evaluate = True
        args.compare = True

def ws(n):
    white_spaces = ""
    for i in range(0, n):
        white_spaces = white_spaces + " "
    return white_spaces


def find_files_with_string(folder_path, matching_string):
    matching_files = []
    for file_name in os.listdir(folder_path):
        if matching_string in file_name:
            file_path = os.path.join(folder_path, file_name)
            matching_files.append(file_path)
    matching_files.sort()
    return matching_files


def check_yaml_file_integrity(yaml_file):
    if not os.path.exists(yaml_file):  # Check if file exists
        print(f"Error: The file '{yaml_file}' does not exist.")
        sys.exit(1)
    if not yaml_file.lower().endswith(('.yaml', '.yml')):  # Check if the file is a yaml file
        print(f"Error: The file '{yaml_file}' is not a yaml file.")
        sys.exit(1)
    try:  # Check the integrity of the yaml file
        with open(yaml_file, 'r') as file:
            yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading the file '{yaml_file}': {e}")
        sys.exit(1)


def find_common_sequences(experiments):
    num_experiments = len(experiments)
    exp_tmp = {}
    for [_, exp] in experiments.items():
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                if not (dataset_name in exp_tmp):
                    exp_tmp[dataset_name] = {}
                for sequence_name in sequence_names:
                    if sequence_name in exp_tmp[dataset_name]:
                        exp_tmp[dataset_name][sequence_name] += 1
                    else:
                        exp_tmp[dataset_name][sequence_name] = 1

    dataset_sequences = {}
    for [dataset_name, sequence_names] in exp_tmp.items():
        for [sequence_name, num_sequences] in sequence_names.items():
            if num_experiments == num_sequences:
                if dataset_name not in dataset_sequences:
                    dataset_sequences[dataset_name] = []
                dataset_sequences[dataset_name].append(sequence_name)
    return dataset_sequences


# Functions to download files

# Downloads the given URL to a file in the given directory. Returns the
# path to the downloaded file.
# Taken from https://www.eth3d.net/slam_datasets/download_eth3d_slam_datasets.py.
def downloadFile(url, dest_dir_path):
    file_name = url.split('/')[-1]
    dest_file_path = os.path.join(dest_dir_path, file_name)

    url_object = urllib.request.urlopen(url)

    with open(dest_file_path, 'wb') as outfile:
        meta = url_object.info()
        if sys.version_info[0] == 2:
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta["Content-Length"])
        print("    Downloading: %s (size [bytes]: %s)" % (url, file_size))

        file_size_downloaded = 0
        block_size = 8192
        while True:
            buffer = url_object.read(block_size)
            if not buffer:
                break

            file_size_downloaded += len(buffer)
            outfile.write(buffer)

            sys.stdout.write("        %d / %d  (%3f%%)\r" % (
                file_size_downloaded, file_size, file_size_downloaded * 100. / file_size))
            sys.stdout.flush()

    return dest_file_path


def decompressFile(filepath, extract_to=None):
    """
    Decompress a .zip, .tar.gz, .tar, or .7z file and return the extraction directory.
    """
    if not extract_to:
        extract_to = os.path.dirname(filepath)

    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            for file in zip_ref.namelist():
                try:
                    zip_ref.extract(file, extract_to)
                except zipfile.BadZipFile:
                    print(f"Skipping corrupted file: {file}")
    elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz') or filepath.endswith('.tar'):
        mode = 'r:gz' if filepath.endswith('.gz') else 'r'
        with tarfile.open(filepath, mode) as tar_ref:
            tar_ref.extractall(extract_to)
    elif filepath.endswith('.7z'):
        with py7zr.SevenZipFile(filepath, mode='r') as z:
            z.extractall(path=extract_to)
    else:
        print("Unsupported file format. Please provide a .zip, .tar.gz, .tar, or .7z file.")
        return None

    return extract_to


def replace_string_in_files(directory, old_string, new_string):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h') or file.endswith('.cpp'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = content.replace(old_string, new_string)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)


def is_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            return True
    except Exception:
        return False


def list_image_files_in_folder(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and is_image_file(file_path):
            image_files.append(filename)
    return image_files

def deactivate_env(vslamlab_env):
    file_path = os.path.join(VSLAM_LAB_DIR, 'pixi.toml')
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Split content by lines
    lines = file_content.splitlines()

    # Flags to track if we are within the baseline section
    inside_baseline = False
    inside_environments = False

    # Iterate through lines and comment the baseline section
    for i in range(len(lines)):
        line = lines[i].strip()

        if f"# {vslamlab_env} begin" == line:
            inside_baseline = True
            continue
        elif f"# {vslamlab_env} end" == line:
            inside_baseline = False
            continue

        if f"# environments begin" == line:
            inside_environments = True
            continue
        elif f"# environments end" == line:
            inside_environments = False
            continue

        # If inside the baseline block, comment the line
        if inside_baseline:
            if not ("#" in lines[i]):
                lines[i] = "# " + lines[i]
        if inside_environments:
            if f'features = ["{vslamlab_env}"]' in lines[i]:
                if not ("#" in lines[i]):
                    lines[i] = "# " + lines[i]

    new_file_content = "\n".join(lines)
    with open(file_path, 'w') as file:
        file.write(new_file_content)

    subprocess.run("pixi clean && pixi update", shell=True)


def activate_env(vslamlab_env):
    file_path = os.path.join(VSLAM_LAB_DIR, 'pixi.toml')
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Split content by lines
    lines = file_content.splitlines()

    # Flags to track if we are within the baseline section
    inside_baseline = False
    inside_environments = False

    # Iterate through lines and comment the baseline section
    for i in range(len(lines)):
        line = lines[i].strip()

        if f"# {vslamlab_env} begin" == line:
            inside_baseline = True
            continue
        elif f"# {vslamlab_env} end" == line:
            inside_baseline = False
            continue

        if f"# environments begin" == line:
            inside_environments = True
            continue
        elif f"# environments end" == line:
            inside_environments = False
            continue

        # If inside the baseline block, comment the line
        if inside_baseline:
            lines[i] = lines[i].replace("# ", '')
        if inside_environments:
            if f'features = ["{vslamlab_env}"]' in lines[i]:
                lines[i] = lines[i].replace("# ", '')

    new_file_content = "\n".join(lines)
    with open(file_path, 'w') as file:
        file.write(new_file_content)

    subprocess.run("pixi clean && pixi update", shell=True)


def show_time(time_s):
    if time_s < 60:
        return f"{time_s:.2f} seconds"
    if time_s < 3600:
        return f"{(time_s / 60):.2f} minutes"
    return f"{(time_s / 3600):.2f} hours"

def format_msg(script_label, msg, flag="info"):
    if flag == "info":
        return f"{script_label}{msg}"
    elif flag == "warning":
        return f"{script_label}{Fore.YELLOW} {msg} {Style.RESET_ALL}"
    elif flag == "error":
        return f"{script_label}{Fore.RED} {msg} {Style.RESET_ALL}"

def print_msg(script_label, msg, flag="info", verb='NONE'):
    if VerbosityManager[verb] <= VerbosityManager[VSLAMLAB_VERBOSITY]:
        print(format_msg(script_label, msg, flag))
    
def read_trajectory_csv(csv_file):
    try:
        trajectory = pd.read_csv(Path(csv_file))
        if trajectory.empty:
            trajectory = None
    except (pd.errors.EmptyDataError, FileNotFoundError):
        trajectory = None
    return trajectory

def read_trajectory_txt(txt_file):
    try:
        trajectory = pd.read_csv(Path(txt_file), header = None, sep = ' ', lineterminator = '\n')
        if trajectory.empty:
            trajectory = None
    except (pd.errors.EmptyDataError, FileNotFoundError):
        trajectory = None
    return trajectory

def save_trajectory_csv(trajectory_csv, trajectory, header=None, index=False, sep=' ', lineterminator='\n'):
    trajectory.to_csv(trajectory_csv, header=header, index=index, sep=sep, lineterminator=lineterminator)

def read_csv(csv_file):
    if not os.path.exists(csv_file):
        return pd.DataFrame()
    try:
        csv_data = pd.read_csv(csv_file)
        if csv_data.empty:
            return pd.DataFrame()
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()
    return csv_data

if __name__ == "__main__":

    if len(sys.argv) > 2:
        function_name = sys.argv[1]

        if function_name == "deactivate_env":
            deactivate_env(sys.argv[2])
        if function_name == "activate_env":
            activate_env(sys.argv[2])


