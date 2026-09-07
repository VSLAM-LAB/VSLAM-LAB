
"""
Module: VSLAM-LAB - Baselines - BaselineVSLAMLAB.py
- Author: Alejandro Fontan Villacampa
- Version: 2.1
- Created: 2024-07-12
- Updated: 2026-09-06
- License: GPLv3 License

BaselineVSLAMLAB: A class to handle Visual SLAM baseline-related operations.

"""

import os
import platform
from pathlib import Path

import shlex
import signal
import subprocess
import sys
import psutil
import threading
import time
import queue
import pynvml
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from huggingface_hub import hf_hub_download

from utilities import ws, print_msg
from path_constants import VSLAMLAB_BASELINES, TRAJECTORY_FILE_NAME, CALIBRATION_EXP_YAML, RGB_EXP_CSV, EXP_FRAMEWORK_PARAMETERS, VSLAMLAB_VERBOSITY, VerbosityManager

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

if TYPE_CHECKING:  # annotations only: vslamlab_utilities imports this package, so a runtime import would be circular
    from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
    from vslamlab_utilities import Experiment


class BaselineVSLAMLAB(ABC):
    """Base baseline class for VSLAM-LAB."""

    # ---- Abstract hooks that concrete baselines must implement ----
    @abstractmethod
    def __init__(self, baseline_name: str, baseline_folder: str, default_parameters: dict | None = None) -> None:
        # Basic fields
        self.baseline_name: str = baseline_name
        self.baseline_folder: str = baseline_folder
        self.label: str = f"\033[96m{baseline_name}\033[0m"
        self.color: str = 'black'
        self.name_label: str = baseline_folder

        # Paths
        self.baseline_path: Path = VSLAMLAB_BASELINES / baseline_folder
        self.settings_yaml: Path = self.baseline_path / f'vslamlab_{baseline_name}_settings.yaml'

        # Default parameters (overridable per experiment via its Parameters: block)
        self.default_parameters: dict = default_parameters or {}

        # Supported modes and entry-point argument style (set by each concrete baseline subclass, see _COMMAND_STYLES)
        self.modes: list[str] = []
        self.command_style: str = ''

        # Set once ensure_pixi_env has run in this process
        self._pixi_env_ready: bool = False

    ####################################################################################################################
    # Ensure Installed
    def ensure_installed(self) -> None:
        self.fetch_source()
        self.install()

    def ensure_pixi_env(self) -> None:
        if self._pixi_env_ready:
            return
        print()
        print_msg(SCRIPT_LABEL, f"pixi initializing {self.label} environment")
        subprocess.run(["pixi", "install", "-e", self.baseline_name])
        self._pixi_env_ready = True

    def has_source(self) -> bool:
        return (self.baseline_path / '.git').exists()

    def fetch_source(self) -> None:
        if self.has_source():
            return

        self.ensure_pixi_env()

        print()
        print_msg(SCRIPT_LABEL, f"fetch source {self.label} : {self.baseline_path}")
        subprocess.run(["pixi", "run", "--frozen", "-e", self.baseline_name, "fetch-source"])

        if not self.has_source():
            print_msg(SCRIPT_LABEL, f"fetch source of {self.baseline_name} failed (see output above)", flag="error", verb='NONE')
            sys.exit(1)

    def is_installed(self) -> tuple[bool, str]:
        return (True, 'is installed') if self.has_source() else (False, 'not installed (conda package available)')

    def install(self) -> None:
        is_installed, _ = self.is_installed()
        if is_installed:
            return

        self.ensure_pixi_env()

        log_file_path = self.baseline_path / f'install_{self.baseline_name}.txt'
        print()
        print_msg(SCRIPT_LABEL, f"install {self.label} : {self.baseline_path}")
        print_msg(ws(6), f"log file: {log_file_path}")
        with open(log_file_path, 'w') as log_file:
            subprocess.run(["pixi", "run", "--frozen", "-e", self.baseline_name, "install", "-v"], stdout=log_file, stderr=log_file)

        is_installed, msg = self.is_installed()
        if not is_installed:
            print_msg(SCRIPT_LABEL, f"install of {self.baseline_name} failed ({msg}), see {log_file_path}", flag="error", verb='NONE')
            sys.exit(1)

    ####################################################################################################################
    # Build Execute Command
    # Argument style per baseline entry-point type: (token format, name of the experiment-iteration key)
    _COMMAND_STYLES: dict[str, tuple[str, str]] = {
        'cpp': ('{key}:{value}', 'exp_id'),
        'python': ('--{key} {value}', 'exp_it'),
    }

    def resolve_parameters(self, exp: 'Experiment') -> dict:
        """Baseline parameters for this run: defaults, overridden by the experiment's Parameters: block.
        Subclasses override to derive one parameter from another (e.g. allfeature's feature_yaml from feature)."""
        parameters = {name: exp.parameters.get(name, value) for name, value in self.default_parameters.items()}

        # Keys neither this baseline nor the run pipeline knows are ignored: warn (not exit), since one
        # experiment yaml is often shared across baselines with different parameter sets
        unknown = [key for key in exp.parameters if key not in self.default_parameters and key not in EXP_FRAMEWORK_PARAMETERS]
        if unknown:
            print()
            print_msg(SCRIPT_LABEL, f"{self.baseline_name} ignores unknown parameter(s) {unknown} (known: {list(self.default_parameters)})", flag="warning", verb='LOW')

        return parameters

    def build_execute_command(self, exp_it: int, exp: 'Experiment', dataset: 'DatasetVSLAMLAB', sequence_name: str) -> str:
        if self.command_style not in self._COMMAND_STYLES:
            print_msg(SCRIPT_LABEL, f"{self.baseline_name} has command_style '{self.command_style}', expected one of {list(self._COMMAND_STYLES)}", flag="error", verb='NONE')
            sys.exit(1)
        arg_format, exp_it_key = self._COMMAND_STYLES[self.command_style]

        exp_folder = exp.folder / dataset.dataset_folder / sequence_name
        arguments = {'sequence_path': dataset.sequence_path(sequence_name),
                     'calibration_yaml': exp_folder / CALIBRATION_EXP_YAML,  # per-experiment copy written by Run.run_functions.create_calibration_exp_yaml
                     'rgb_csv': exp_folder / RGB_EXP_CSV,
                     'exp_folder': exp_folder,
                     exp_it_key: exp_it,
                     'settings_yaml': self.settings_yaml}
        arguments.update(self.resolve_parameters(exp))

        mode = arguments.get('mode')  # selects the pixi task execute-<mode>
        if mode not in self.modes:
            print_msg(SCRIPT_LABEL, f"{self.baseline_name} does not support mode '{mode}', expected one of {self.modes}", flag="error", verb='NONE')
            sys.exit(1)

        # Values are shell-quoted (no-op for plain values) since execute() runs the command with shell=True
        tokens = [arg_format.format(key=key, value=shlex.quote(str(value))) for key, value in arguments.items()]
        return f"pixi run --frozen -e {self.baseline_name} execute-{mode} " + ' '.join(tokens)

    ####################################################################################################################
    # Execute methods
    def kill_process(self, process):
        if platform.system() == 'Windows':
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        else:
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

                if sys.platform == "linux" and swap_perc > MAX_SWAP_PERC:
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
            _popen_kwargs = {} if platform.system() == 'Windows' else {'preexec_fn': os.setsid}
            if VerbosityManager[VSLAMLAB_VERBOSITY] <= VerbosityManager['LOW']:
                process = subprocess.Popen(command, shell=True, stdout=log_file, stderr=log_file, text=True, **_popen_kwargs)
            else:
                process = subprocess.Popen(command, shell=True, **_popen_kwargs)

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
    # Auxiliary methods
    def info_print(self) -> None:
        print(f'Name: {self.label}')
        is_installed, install_msg = self.is_installed()

        if is_installed:
            print_msg(f"{ws(0)}", f"Installed:\033[92m {install_msg}\033[0m", verb='LOW')
        else:
            print_msg(f"{ws(0)}", f"Installed:\033[93m {install_msg}\033[0m", verb='LOW')

        has_source = self.has_source()
        print(f"Path:\033[92m {self.baseline_path}\033[0m" if has_source else f"Path:\033[93m {self.baseline_path} (missing)\033[0m")
        print(f'Modalities: {self.modes}')
        print(f'Default parameters: {self.get_default_parameters()}')

    def download_vslamlab_settings(self) -> bool: # Download vslamlab_{baseline_name}_settings.yaml
        if not self.settings_yaml.is_file():
            settings_yaml = self.settings_yaml.name
            print_msg(SCRIPT_LABEL, f"Downloading {self.settings_yaml} ...",'info')
            _ = hf_hub_download(repo_id=f'vslamlab/{self.baseline_name}', filename=settings_yaml, repo_type='model', local_dir=self.baseline_path)
        return self.settings_yaml.is_file()

    ####################################################################################################################
    # Utils
    def get_default_parameters(self) -> dict:
        return self.default_parameters