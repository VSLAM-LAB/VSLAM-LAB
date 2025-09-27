import os, yaml
import signal
import subprocess
import psutil
import threading
import time
import queue
import pynvml
from huggingface_hub import hf_hub_download
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from utilities import ws, print_msg
from path_constants import VSLAMLAB_BASELINES, TRAJECTORY_FILE_NAME


SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class BaselineVSLAMLab:

    def __init__(self, baseline_name, baseline_folder, default_parameters=''):
        self.baseline_name = baseline_name
        self.baseline_folder = baseline_folder
        self.baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        self.label = f"\033[96m{baseline_name}\033[0m"
        self.settings_yaml = os.path.join(self.baseline_path, f'vslamlab_{baseline_name}_settings.yaml')
        self.default_parameters = default_parameters
        self.color = 'black'
        self.name_label = baseline_folder

    def get_default_parameters(self):
        return self.default_parameters

    def is_cloned(self):
        if os.path.isdir(os.path.join(self.baseline_path, '.git')):
            return True
        return False
    
    def git_clone(self):
        if self.is_cloned():
            return

        log_file_path = os.path.join(VSLAMLAB_BASELINES, f'git_clone_{self.baseline_name}.txt')
        git_clone_command = f"pixi run --frozen -e {self.baseline_name} git-clone"
        with open(log_file_path, 'w') as log_file:
            print(f"\n{SCRIPT_LABEL}git clone {self.label}\033[0m : {self.baseline_path}")
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(git_clone_command, shell=True, stdout=log_file, stderr=log_file)

    def is_installed(self):
        return False

    def install(self):
        if self.is_installed()[0]:
            return

        log_file_path = os.path.join(self.baseline_path, f'install_{self.baseline_name}.txt')
        install_command = f"pixi run --frozen -e {self.baseline_name} install -v"
        with open(log_file_path, 'w') as log_file:
            print(f"\n{SCRIPT_LABEL}Installing {self.label}\033[0m : {self.baseline_path}")
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(install_command, shell=True, stdout=log_file, stderr=log_file)

    def check_installation(self):
        self.git_clone()
        self.install()

    def info_print(self):
        print(f'Name: {self.label}')
        is_installed, install_msg = self.is_installed()

        if is_installed:
            print_msg(f"{ws(0)}", f"Installed:\033[92m {install_msg}\033[0m", verb='LOW')
        else:    
            print_msg(f"{ws(0)}", f"Installed:\033[93m {install_msg}\033[0m", verb='LOW')
  
        is_cloned = self.is_cloned()
        print(f"Path:\033[92m {self.baseline_path}\033[0m" if is_cloned else f"Path:\033[93m {self.baseline_path} (missing)\033[0m")
        print(f'Modalities: {self.modes}')
        print(f'Default parameters: {self.get_default_parameters()}')

    def download_vslamlab_settings(self): # Download vslamlab_{baseline_name}_settings.yaml
        if not os.path.isfile(self.settings_yaml):
            settings_yaml = os.path.basename(self.settings_yaml)
            print_msg(SCRIPT_LABEL, f"Downloading {self.settings_yaml} ...",'info')
            _ = hf_hub_download(repo_id=f'vslamlab/{self.baseline_name}', filename=settings_yaml, repo_type='model', local_dir=self.baseline_path)
        return os.path.isfile(self.settings_yaml)
    
    def kill_process(self, process):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  
        try:
            process.wait(timeout=5) 
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL) 
        print_msg(SCRIPT_LABEL, "Process killed.",'error')

    def monitor_memory(self, process, interval, comment_queue, success_flag, memory_stats):
        MAX_SWAP_PERC = 0.80
        MAX_RAM_PERC= 0.95

        nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        device_handle = nvmlDeviceGetHandleByIndex(0) 

        swap_0, swap_max = psutil.swap_memory().used / (1024**3), psutil.swap_memory().total / (1024**3)
        ram_0, ram_max = psutil.virtual_memory().used / (1024**3), psutil.virtual_memory().total / (1024**3)
        gpu_0 = nvmlDeviceGetMemoryInfo(device_handle).used / (1024**3)

        ram_inc, swap_inc, gpu_inc = 0, 0, 0
        ram_inc_max, swap_inc_max, gpu_inc_max = 0, 0, 0
        while process.poll() is None: 
            try:
                swap, ram =  psutil.swap_memory(), psutil.virtual_memory()
                swap_used, ram_used = swap.used / (1024**3), ram.used / (1024**3)
                gpu_mem_info = nvmlDeviceGetMemoryInfo(device_handle)
                gpu_used = gpu_mem_info.used / (1024**3)
                #gpu_total = gpu_mem_info.total / (1024**3)
                #gpu_inc = gpu_used - gpu_0
                #gpu_perc = gpu_used / gpu_total

                ram_inc = ram_used - ram_0
                swap_inc = swap_used - swap_0
                gpu_inc = gpu_used - gpu_0

                ram_inc_max = max(ram_inc_max, ram_inc)
                swap_inc_max = max(swap_inc_max, swap_inc)
                gpu_inc_max = max(gpu_inc_max, gpu_inc)

                swap_perc, ram_perc = swap_used / swap_max, ram_used / ram_max
                if ram_perc > MAX_RAM_PERC:
                    print_msg(SCRIPT_LABEL, f"Memory threshold exceeded  {ram_used:0.1f} GB / {ram_max:0.1f} GB > {100 * MAX_RAM_PERC:0.2f} %",'error')
                    success_flag[0] = False
                    self.kill_process(process)
                    comment_queue.put(f"Memory threshold exceeded  {ram_used:0.1f} GB / {ram_max:0.1f} GB > {100 * MAX_RAM_PERC:0.2f} %. Process killed.")  
                    break

                if swap_perc > MAX_SWAP_PERC:
                    print_msg(SCRIPT_LABEL, f"Filling swap memory  {swap_used:0.1f} GB / {swap_max:0.1f} GB > {100 * MAX_SWAP_PERC:0.2f}",'error')
                    success_flag[0] = False
                    self.kill_process(process)
                    comment_queue.put(f"Filling swap memory  {swap_used:0.1f} GB / {swap_max:0.1f} GB > {100 * MAX_SWAP_PERC:0.2f} %. Process killed.")  
                    break
                    #print(f"\n{SCRIPT_LABEL} {Fore.RED} Cleaning swap memory... {Style.RESET_ALL}")
                    #subprocess.run("pixi run --frozen -e default clean_swap", shell=True)

                time.sleep(interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                break  

        memory_stats['ram'] = ram_inc_max
        memory_stats['swap'] = swap_inc_max
        memory_stats['gpu'] = gpu_inc_max


    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*1000000):
        log_file_path = os.path.join(exp_folder, "system_output_" + str(exp_it).zfill(5) + ".txt")
        comments = ""
        comment_queue = queue.Queue()
        success_flag = [True] 
        memory_stats = {}
        with open(log_file_path, 'w') as log_file:
            print(f"{ws(8)}log file: {log_file_path}")
            process = subprocess.Popen(command, shell=True, stdout=log_file, stderr=log_file, text=True, preexec_fn=os.setsid)
            memory_thread = threading.Thread(target=self.monitor_memory, args=(process, 10, comment_queue, success_flag, memory_stats))
            memory_thread.start()

            try:
                _, _ = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                print_msg(SCRIPT_LABEL, f"Process took too long > {timeout_seconds} seconds",'error')
                comments = f"Process took too long > {timeout_seconds} seconds. Process killed."
                success_flag[0] = False
                self.kill_process(process)
            
            memory_thread.join()
            while not comment_queue.empty():
                comments += comment_queue.get() + "\n"

        if not os.path.exists(os.path.join(exp_folder, str(exp_it).zfill(5) + f"_{TRAJECTORY_FILE_NAME}.csv" )):
            success_flag[0] = False

        return {
            "success": success_flag[0],
            "comments": comments,
            "ram": memory_stats.get('ram', 'N/A'),
            "swap": memory_stats.get('swap', 'N/A'),
            "gpu": memory_stats.get('gpu', 'N/A')
        }

    def build_execute_command_cpp(self, exp_it, exp, dataset, sequence_name):
        sequence_path = os.path.join(dataset.dataset_path, sequence_name)
        exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        rgb_exp_csv = os.path.join(exp_folder, 'rgb_exp.csv')

        vslamlab_command = [f"sequence_path:{sequence_path}",
                            f"calibration_yaml:{calibration_yaml}",
                            f"rgb_csv:{rgb_exp_csv}",
                            f"exp_folder:{exp_folder}",
                            f"exp_id:{exp_it}",
                            f"settings_yaml:{self.settings_yaml}"]

        for parameter_name, parameter_value in self.default_parameters.items():
            if parameter_name in exp.parameters:
                vslamlab_command += [f"{str(parameter_name)}:{str(exp.parameters[parameter_name])}"]
            else:
                vslamlab_command += [f"{str(parameter_name)}:{str(parameter_value)}"]

        if "mode:mono" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-mono " + ' '.join(vslamlab_command)

        if "mode:rgbd" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-rgbd " + ' '.join(vslamlab_command)

        if "mode:stereo" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-stereo " + ' '.join(vslamlab_command)

        if "mode:mono-vi" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-mono_vi " + ' '.join(vslamlab_command)

        return vslamlab_command

    def build_execute_command_python(self, exp_it, exp, dataset, sequence_name):
        sequence_path = os.path.join(dataset.dataset_path, sequence_name)
        exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        rgb_exp_csv = os.path.join(exp_folder, 'rgb_exp.csv')

        vslamlab_command = [f"--sequence_path {sequence_path}",
                            f"--calibration_yaml {calibration_yaml}",
                            f"--rgb_csv {rgb_exp_csv}",
                            f"--exp_folder {exp_folder}",
                            f"--exp_it {exp_it}",
                            f"--settings_yaml {self.settings_yaml}"]

        for parameter_name, parameter_value in self.default_parameters.items():
            if parameter_name in exp.parameters:
                vslamlab_command += [f"--{str(parameter_name)} {str(exp.parameters[parameter_name])}"]
            else:
                vslamlab_command += [f"--{str(parameter_name)} {str(parameter_value)}"]

        if "--mode mono" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-mono " + ' '.join(vslamlab_command)
        if "--mode rgbd" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-rgbd " + ' '.join(vslamlab_command)
        if "--mode stereo" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-stereo " + ' '.join(vslamlab_command)    
        if "--mode mono-vi" in vslamlab_command:
            vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute-mono_vi " + ' '.join(vslamlab_command)

        return vslamlab_command

    def modify_yaml_parameter(self, settings_ablation_yaml, section_name, parameter_name, new_value):
        with open(settings_ablation_yaml, 'r') as file:
            data = yaml.safe_load(file)

        if section_name in data and parameter_name in data[section_name]:
            data[section_name][parameter_name] = new_value
        else:
            print(f"    Parameter '{parameter_name}' or section '{section_name}' not found in the YAML file.")

        with open(settings_ablation_yaml, 'w') as file:
            yaml.safe_dump(data, file)
