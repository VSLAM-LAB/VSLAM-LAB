import os
import sys
from enum import Enum
from pathlib import Path

HUGGINGFACE_TOKEN = None

VSLAM_LAB_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
VSLAM_LAB_PATH = Path(os.path.dirname(VSLAM_LAB_DIR))

VSLAMLAB_BENCHMARK = VSLAM_LAB_PATH / "VSLAM-LAB-Benchmark"
VSLAMLAB_EVALUATION = VSLAM_LAB_PATH / 'VSLAM-LAB-Evaluation'
VSLAMLAB_BASELINES = VSLAM_LAB_DIR / 'Baselines'
VSLAMLAB_VIDEOS = VSLAMLAB_BENCHMARK / 'VIDEOS'

COMPARISONS_YAML_DEFAULT = VSLAM_LAB_DIR / 'configs' / 'comp_complete.yaml'
EXP_YAML_DEFAULT = 'exp_debug.yaml'
CONFIG_DEFAULT = 'config_debug.yaml'

VSLAM_LAB_EVALUATION_FOLDER = 'vslamlab_evaluation'
RGB_BASE_FOLDER = 'rgb'
GROUNTRUTH_FILE = 'groundtruth.csv'
CALIBRATION_EXP_YAML = 'calibration_exp.yaml'  # per-experiment copy of a sequence's calibration.yaml (Run/run_functions.py)

ABLATION_PARAMETERS_CSV = 'log_ablation_parameters.csv'

TRAJECTORY_FILE_NAME = 'KeyFrameTrajectory'
SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class Retention(str, Enum):
    MINIMAL="minimal"; STANDARD="standard"; FULL="full"
BENCHMARK_RETENTION = Retention.STANDARD

VSLAMLAB_VERBOSITY = 'LOW'

VerbosityManager = {
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "NONE": 0
}

def set_VSLAMLAB_path(new_path: str, file_path: str, target_line_start: str) -> None:
    new_line = f"{target_line_start} Path(\"{Path(new_path).expanduser().resolve()}\")"

    with open(file_path, 'r') as file:
        lines = file.readlines()

    replaced = False
    with open(file_path, 'w') as file:
        for line in lines:
            if not replaced and line.startswith(target_line_start):
                file.write(new_line + '\n')
                replaced = True
            else:
                file.write(line)

    if replaced:
        print(f"{SCRIPT_LABEL}Set {new_line}")
    else:
        print(f"{SCRIPT_LABEL}\033[91m[ERROR]\033[0m No line starting with '{target_line_start}' found in {file_path}")
        sys.exit(1)

if __name__ == "__main__":

    if len(sys.argv) > 2:
        function_name = sys.argv[1]
        if function_name == 'set_VSLAMLAB_BENCHMARK_path':
            set_VSLAMLAB_path(os.path.join(sys.argv[2], 'VSLAM-LAB-Benchmark'), __file__, "VSLAMLAB_BENCHMARK =")
            set_VSLAMLAB_path(os.path.join(sys.argv[2], 'VSLAM-LAB-Benchmark', 'VIDEOS'), __file__, "VSLAMLAB_VIDEOS =")
        if function_name == 'set_VSLAMLAB_EVALUATION_path':
            set_VSLAMLAB_path(os.path.join(sys.argv[2], 'VSLAM-LAB-Evaluation'), __file__, "VSLAMLAB_EVALUATION =")
