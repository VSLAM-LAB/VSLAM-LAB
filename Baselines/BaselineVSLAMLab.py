
"""
Module: VSLAM-LAB - Baselines - BaselineVSLAMLab.py
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2024-07-12
- Updated: 2025-12-30
- License: GPLv3 License

BaselineVSLAMLab: A class to handle Visual SLAM baseline-related operations.

"""

import os
from pathlib import Path

import signal
import subprocess
import psutil
import threading
import time
import queue
import pynvml
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download

from utilities import ws, print_msg
from path_constants import VSLAMLAB_BASELINES, TRAJECTORY_FILE_NAME

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class BaselineVSLAMLab(ABC):
    """Base baseline class for VSLAM-LAB."""

    # ---- Abstract hooks that concrete baselines must implement ---- 
    @abstractmethod  
    def __init__(self, baseline_name: str, baseline_folder: str, default_parameters='') -> None:  
        # Basic fields
        self.baseline_name: str = baseline_name
        self.baseline_folder: str = baseline_folder
        self.label: str = f"\033[96m{baseline_name}\033[0m"
        self.color: str = 'black'
        self.name_label: str = baseline_folder

        # Paths
        self.baseline_path: Path = VSLAMLAB_BASELINES / baseline_folder
        self.settings_yaml: Path = self.baseline_path / f'vslamlab_{baseline_name}_settings.yaml'

        # Defaults parameters
        self.default_parameters = default_parameters

    @abstractmethod  
    def build_execute_command(self, exp_it, exp, dataset, sequence_name) -> str: ...

    @abstractmethod  
    def is_installed(self) -> bool: ...

    def is_cloned(self) -> bool:
        return (self.baseline_path / '.git').is_dir()
    
    def git_clone(self) -> None:
        if self.is_cloned():
            return

        log_file_path = VSLAMLAB_BASELINES / f'git_clone_{self.baseline_name}.txt'
        git_clone_command = f"pixi run --frozen -e {self.baseline_name} git-clone"
        with open(log_file_path, 'w') as log_file:
            print(f"\n{SCRIPT_LABEL}git clone {self.label}\033[0m : {self.baseline_path}")
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(git_clone_command, shell=True, stdout=log_file, stderr=log_file)

    ####################################################################################################################
    # Auxiliary methods    
    def install(self) -> None:
        if self.is_installed()[0]:
            return

        log_file_path = self.baseline_path / f'install_{self.baseline_name}.txt'
        install_command = f"pixi run --frozen -e {self.baseline_name} install -v"
        with open(log_file_path, 'w') as log_file:
            print(f"\n{SCRIPT_LABEL}Installing {self.label}\033[0m : {self.baseline_path}")
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(install_command, shell=True, stdout=log_file, stderr=log_file)

    def check_installation(self) -> None:
        self.git_clone()
        self.install()

    def info_print(self) -> None:
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

    def download_vslamlab_settings(self) -> bool: # Download vslamlab_{baseline_name}_settings.yaml
        if not self.settings_yaml.is_file():
            settings_yaml = self.settings_yaml.name
            print_msg(SCRIPT_LABEL, f"Downloading {self.settings_yaml} ...",'info')
            _ = hf_hub_download(repo_id=f'vslamlab/{self.baseline_name}', filename=settings_yaml, repo_type='model', local_dir=self.baseline_path)
        return self.settings_yaml.is_file()
    
    def build_execute_command_cpp(self, exp_it, exp, dataset, sequence_name):
        sequence_path = dataset.dataset_path / sequence_name
        exp_folder = Path(exp.folder) / dataset.dataset_folder / sequence_name
        calibration_yaml = sequence_path / 'calibration.yaml'
        rgb_exp_csv = exp_folder / 'rgb_exp.csv'

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

        mode_str = next((s for s in vslamlab_command if s.startswith("mode:")), None).replace("mode:", '')
        return f"pixi run --frozen -e {self.baseline_name} execute-{mode_str} " + ' '.join(vslamlab_command)

    def build_execute_command_python(self, exp_it, exp, dataset, sequence_name):
        sequence_path = dataset.dataset_path / sequence_name
        exp_folder = Path(exp.folder) / dataset.dataset_folder / sequence_name
        calibration_yaml = sequence_path / 'calibration.yaml'
        rgb_exp_csv = exp_folder / 'rgb_exp.csv'

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

        mode_str = next((s for s in vslamlab_command if s.startswith("--mode ")), None).replace("--mode ", '')
        return f"pixi run --frozen -e {self.baseline_name} execute-{mode_str} " + ' '.join(vslamlab_command)

    ####################################################################################################################
    # Execute methods
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

        # Initialize NVML safely
        gpu_handle = None
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass # No NVIDIA GPU or driver issue

        # Initial snapshots
        swap_0 = psutil.swap_memory().used / (1024**3)
        swap_max = psutil.swap_memory().total / (1024**3)
        ram_0 = psutil.virtual_memory().used / (1024**3)
        ram_max = psutil.virtual_memory().total / (1024**3)

        gpu_0 = 0
        if gpu_handle:
            try:
                gpu_0 = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024**3)
            except Exception:
                pass

        ram_inc_max, swap_inc_max, gpu_inc_max = 0, 0, 0      
        while process.poll() is None: 
            try:
                # 1. Check System Safety (Global)
                ram = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                ram_used = ram.used / (1024**3)
                swap_used = swap.used / (1024**3)
                
                ram_perc = ram_used / ram_max if ram_max > 0 else 0.0
                swap_perc = swap_used / swap_max if swap_max > 0 else 0.0

                if ram_perc > MAX_RAM_PERC:
                    msg = f"RAM threshold exceeded: {ram_used:.1f}/{ram_max:.1f} GB (> {MAX_RAM_PERC:.0%})"
                    print_msg(SCRIPT_LABEL, msg, 'error')
                    success_flag[0] = False
                    comment_queue.put(msg + ". Process killed.")
                    self.kill_process(process)
                    break
                
                if swap_perc > MAX_SWAP_PERC:
                    msg = f"Swap threshold exceeded: {swap_used:.1f}/{swap_max:.1f} GB (> {MAX_SWAP_PERC:.0%})"
                    print_msg(SCRIPT_LABEL, msg, 'error')
                    success_flag[0] = False
                    comment_queue.put(msg + ". Process killed.")
                    self.kill_process(process)
                    break

                # 2. Track Usage Stats (Incremental)
                ram_inc_max = max(ram_inc_max, ram_used - ram_0)
                swap_inc_max = max(swap_inc_max, swap_used - swap_0)

                if gpu_handle:
                    try:
                        gpu_used = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024**3)
                        gpu_inc_max = max(gpu_inc_max, gpu_used - gpu_0)
                    except Exception:
                        pass # GPU stats failed, ignore

                time.sleep(interval)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                break

        # Shutdown NVML if it was initialized
        if gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        memory_stats['ram'] = ram_inc_max
        memory_stats['swap'] = swap_inc_max
        memory_stats['gpu'] = gpu_inc_max

    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*1000000):
        log_file_path = exp_folder / ("system_output_" + str(exp_it).zfill(5) + ".txt")
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

        if not (exp_folder / (str(exp_it).zfill(5) + f"_{TRAJECTORY_FILE_NAME}.csv")).exists():
            success_flag[0] = False

        return {
            "success": success_flag[0],
            "comments": comments,
            "ram": memory_stats.get('ram', 'N/A'),
            "swap": memory_stats.get('swap', 'N/A'),
            "gpu": memory_stats.get('gpu', 'N/A')
        }

    ####################################################################################################################
    # Utils
    def get_default_parameters(self) -> dict:
        return self.default_parameters