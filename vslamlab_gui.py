import os, sys

from vslamlab_utilities import baseline_info, print_datasets, print_baselines, validate_experiment_yaml, overwrite_exp
from vslamlab_utilities import update_experiment_csv_logs, check_experiment_resources, get_experiment_resources, check_experiment_state
from vslamlab_utilities import install_baseline, install_baselines, download_sequence, download_sequences, download_dataset, download_datasets
from vslamlab_utilities import run_exp, evaluate_exp, compare_exp, write_demo_yaml_fles
from path_constants import VSLAM_LAB_DIR

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

if __name__ == "__main__":
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        
        # VSLAM-LAB info functions   
        if function_name == "baseline_info":
            baseline_name = sys.argv[2]
            baseline_info(baseline_name)

        if function_name == "print_baselines":
            print_baselines()

        if function_name == "print_datasets":
            print_datasets()    

        # VSLAM-LAB pipeline functions
        if function_name == "validate_experiment_yaml":
            exp_yaml = sys.argv[2]
            validate_experiment_yaml(exp_yaml)

        if function_name == "overwrite_exp":
            exp_yaml = sys.argv[2]
            overwrite_exp(exp_yaml)

        if function_name == "update_experiment_csv_logs":
            exp_yaml = sys.argv[2]
            update_experiment_csv_logs(exp_yaml)

        if function_name == "check_experiment_resources":
            exp_yaml = sys.argv[2]
            check_experiment_resources(exp_yaml)

        if function_name == "get_experiment_resources":
            exp_yaml = sys.argv[2]
            get_experiment_resources(exp_yaml)

        if function_name == "check_experiment_state":
            exp_yaml = sys.argv[2]
            check_experiment_state(exp_yaml)   

        if function_name == "install_baseline":
            baseline_name = sys.argv[2]
            install_baseline(baseline_name)  

        if function_name == "install_baselines":
            baseline_names = sys.argv[2:]
            install_baselines(baseline_names)  

        if function_name == "download_sequence":
            dataset_name = sys.argv[2]
            sequence_name = sys.argv[3]
            download_sequence(dataset_name, sequence_name)  

        if function_name == "download_sequences":
            dataset_sequence_names = sys.argv[2:]
            download_sequences(dataset_sequence_names)  

        if function_name == "download_dataset":
            dataset_name = sys.argv[2]
            download_dataset(dataset_name)  

        if function_name == "download_datasets":
            dataset_names = sys.argv[2:]
            download_datasets(dataset_names)  

        if function_name == "run_exp":
            exp_yaml = sys.argv[2]
            validate_experiment_yaml(exp_yaml)
            overwrite = False
            if len(sys.argv) >= 4:
                if sys.argv[3] == '--overwrite':
                    overwrite = True
                    overwrite_exp(exp_yaml)
            update_experiment_csv_logs(exp_yaml)
            get_experiment_resources(exp_yaml)
            check_experiment_state(exp_yaml)   
            run_exp(exp_yaml) 

        if function_name == "evaluate_exp":
            exp_yaml = sys.argv[2]
            overwrite = False
            if len(sys.argv) >= 4:
                if sys.argv[3] == '--overwrite':
                    overwrite = True
            evaluate_exp(exp_yaml, overwrite)   

        if function_name == "compare_exp":
            exp_yaml = sys.argv[2]
            compare_exp(exp_yaml) 

        # VSLAM-LAB main pipeline    
        if function_name == "vslamlab":
            exp_yaml = sys.argv[2]
            validate_experiment_yaml(exp_yaml)
            overwrite = False
            if len(sys.argv) >= 4:
                if sys.argv[3] == '--overwrite':
                    overwrite = True
                    overwrite_exp(exp_yaml)
            update_experiment_csv_logs(exp_yaml)
            get_experiment_resources(exp_yaml)
            check_experiment_state(exp_yaml)   
            run_exp(exp_yaml) 
            evaluate_exp(exp_yaml, overwrite)     
            compare_exp(exp_yaml) 

        # VSLAM-LAB demo  
        if function_name == "demo":
            baseline_name = sys.argv[2]
            dataset_name = sys.argv[3]
            sequence_name = sys.argv[4]
            exp_yaml = VSLAM_LAB_DIR / 'configs' / 'exp_demo.yaml'
            config_yaml = VSLAM_LAB_DIR / 'configs' / 'config_demo.yaml'
            if len(sys.argv) > 5:
                mode = sys.argv[5]
                write_demo_yaml_fles(baseline_name, dataset_name, sequence_name, mode)
            else:
                write_demo_yaml_fles(baseline_name, dataset_name, sequence_name)
            validate_experiment_yaml(exp_yaml)
            overwrite_exp(exp_yaml)
            update_experiment_csv_logs(exp_yaml)
            get_experiment_resources(exp_yaml)
            check_experiment_state(exp_yaml)   
            run_exp(exp_yaml) 