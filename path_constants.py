import os
import sys

VSLAM_LAB_DIR = os.path.dirname(os.path.abspath(__file__))
VSLAM_LAB_PATH = os.path.dirname(VSLAM_LAB_DIR)

VSLAMLAB_BENCHMARK = "/home/vmehta/datasets/vislam-lab/VSLAM-LAB-Benchmark"
VSLAMLAB_EVALUATION = "/home/vmehta/datasets/vislam-lab/VSLAM-LAB-Evaluation"
VSLAMLAB_BASELINES = os.path.join(VSLAM_LAB_DIR, 'Baselines')
VSLAMLAB_VIDEOS = "/home/vmehta/datasets/vislam-lab/VSLAM-LAB-Benchmark/VIDEOS"

COMPARISONS_YAML_DEFAULT = os.path.join(VSLAM_LAB_DIR, 'configs', 'comp_complete.yaml')
EXP_YAML_DEFAULT = 'exp_debug.yaml'
CONFIG_DEFAULT = 'config_debug.yaml'

VSLAM_LAB_EVALUATION_FOLDER = 'vslamlab_evaluation'
RGB_BASE_FOLDER = 'rgb'
GROUNTRUTH_FILE = 'groundtruth.csv'

ABLATION_PARAMETERS_CSV = 'log_ablation_parameters.csv'

TRAJECTORY_FILE_NAME = 'KeyFrameTrajectory'
SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

VSLAMLAB_BENCHMARK_WEIGHT = 'light'
VSLAMLAB_VERBOSITY = 'LOW'

VerbosityManager = {
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "NONE": 0
}

def set_VSLAMLAB_path(new_path, file_path, target_line_start):
    new_line = f"{target_line_start} \"{new_path}\""
    print(f"{SCRIPT_LABEL}Set {new_line}")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip().startswith(target_line_start):
                file.write(new_line + '\n')
            else:
                file.write(line)

if __name__ == "__main__":

    if len(sys.argv) > 2:
        function_name = sys.argv[1]
        if function_name == 'set_VSLAMLAB_BENCHMARK_path':
            set_VSLAMLAB_path(os.path.join(sys.argv[2], 'VSLAM-LAB-Benchmark'), __file__, "VSLAMLAB_BENCHMARK =")
            set_VSLAMLAB_path(os.path.join(sys.argv[2], 'VSLAM-LAB-Benchmark', 'VIDEOS'), __file__, "VSLAMLAB_VIDEOS =")
        if function_name == 'set_VSLAMLAB_EVALUATION_path':
            set_VSLAMLAB_path(os.path.join(sys.argv[2], 'VSLAM-LAB-Evaluation'), __file__, "VSLAMLAB_EVALUATION =")
