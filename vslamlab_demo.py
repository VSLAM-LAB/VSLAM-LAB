import os, sys, yaml, shutil

from vslamlab_utilities import check_experiments
from Run.run_functions import run_sequence
from Datasets.get_dataset import get_dataset
from Baselines.get_baseline import get_baseline
from path_constants import  VSLAMLAB_BENCHMARK, VSLAM_LAB_DIR, VSLAMLAB_VIDEOS
from vslamlab_utilities import add_video

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def main():
    exp_demo = 'demo'
    exp_yaml = os.path.join(VSLAM_LAB_DIR, 'configs', 'exp_demo.yaml')
    config_yaml =  os.path.join(VSLAM_LAB_DIR, 'configs', 'config_demo.yaml') 

    baseline_name = sys.argv[1]
    dataset_name = sys.argv[2]
    sequence_name = sys.argv[3]

    mode = sys.argv[4] if len(sys.argv) > 4 else 'mono'
    
    if dataset_name == 'videos':
        sequence_name = add_video(sequence_name)

    baseline = get_baseline(baseline_name)
    dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)

    # Write experiment yaml
    exp_data = {}
    exp_data[exp_demo] = {}
    exp_data[exp_demo]['Config'] = config_yaml
    exp_data[exp_demo]['NumRuns'] = 1
    exp_data[exp_demo]['Parameters'] = baseline.get_default_parameters()
    exp_data[exp_demo]['Module'] = baseline_name

    exp_data[exp_demo]['Parameters']['mode'] = mode
    yaml.safe_dump(exp_data, open(exp_yaml, 'w'))

    # Write config yaml
    exp_seq = {
    dataset_name: [
        sequence_name
    ]}   
    yaml.safe_dump(exp_seq, open(config_yaml, 'w'))

    # Create experiment
    experiments = check_experiments(exp_yaml, overwrite=True)

    # Run experiment
    dataset.download_sequence(sequence_name)
    run_sequence(0, experiments[exp_demo], baseline, dataset, sequence_name)

if __name__ == "__main__":
    main()
