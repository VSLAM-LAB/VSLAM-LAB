import sys, os, yaml, shutil, csv, time
import pandas as pd
from typing import Any
from pathlib import Path
from inputimeout import inputimeout, TimeoutOccurred
 

from utilities import ws, load_yaml_file, print_msg, show_time, read_csv
from Datasets.get_dataset import list_available_datasets, get_dataset
from Baselines.get_baseline import list_available_baselines, get_baseline
from Run.run_functions import run_sequence
from Evaluate.evaluate_functions import evaluate_sequence
from Evaluate import compare_functions
from path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION, VSLAM_LAB_DIR, CONFIG_DEFAULT, VSLAMLAB_VIDEOS, COMPARISONS_YAML_DEFAULT

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

##################################################################################################################################################
# write_demo_yaml_fles
##################################################################################################################################################
##################################################################################################################################################
def write_demo_yaml_fles(baseline_name: str, dataset_name: str, sequence_name: str, mode :str = None) -> None:
    # Write experiment yaml
    exp_demo = 'demo'
    exp_yaml = VSLAM_LAB_DIR / 'configs' / 'exp_demo.yaml'
    config_yaml =  VSLAM_LAB_DIR / 'configs' / 'config_demo.yaml'

    baseline = get_baseline(baseline_name)

    exp_data = {}
    exp_data[exp_demo] = {}
    exp_data[exp_demo]['Config'] = str(config_yaml)
    exp_data[exp_demo]['NumRuns'] = 1
    exp_data[exp_demo]['Parameters'] = baseline.get_default_parameters()
    exp_data[exp_demo]['Module'] = baseline_name
    if mode:
        exp_data[exp_demo]['Parameters']['mode'] = mode
    
    # Write experiment yaml
    with open(exp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(exp_data, f)

    # Write config yaml
    exp_seq = { dataset_name: [sequence_name] }   
    with open(config_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(exp_seq, f)

##################################################################################################################################################
# Info commands
##################################################################################################################################################
##################################################################################################################################################

def baseline_info(baseline_name: str) -> None:

    baseline_list = list_available_baselines()
    if baseline_name not in baseline_list:
        print_msg(f"\n{SCRIPT_LABEL}", f"'{baseline_name}' baseline doesn't exist.", "error")
        print_baselines()
        exit(0)

    baseline = get_baseline(baseline_name)    
    baseline.info_print()

def print_baselines() -> None:
    baseline_list = list_available_baselines()
    print_msg(f"\n{SCRIPT_LABEL}", f"Accessible baselines in VSLAM-LAB:", "info")
    for baseline in baseline_list:
        print(f" - {baseline}")
    print("For detailed information about a baseline, use 'pixi run baseline-info <baseline_name>'")

def print_datasets() -> None:
    dataset_list = list_available_datasets()
    print(f"\n{SCRIPT_LABEL}Accessible datasets in VSLAM-LAB:")
    for dataset in dataset_list:
        print(f" - {dataset}")

def add_video(video_path):
    abs_path = os.path.abspath(video_path)
    video_name_ext = os.path.basename(abs_path)
    sequence_name = os.path.splitext(video_name_ext)[0]
    dataset_videos_yaml = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'dataset_videos.yaml')
    with open(dataset_videos_yaml, 'r') as f:
        data = yaml.safe_load(f)
    if 'sequence_names' not in data or data['sequence_names'] is None:
        data['sequence_names'] = []
    if sequence_name not in data['sequence_names']:
        data['sequence_names'].append(sequence_name)
    with open(dataset_videos_yaml, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
    if not os.path.exists(os.path.join(VSLAMLAB_VIDEOS, video_name_ext)):
        shutil.copy2(abs_path, os.path.join(VSLAMLAB_VIDEOS, video_name_ext))
    
    return sequence_name

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

class Experiment:
    def __init__(self, name, settings):            
        self.name = name
        self.folder = os.path.join(VSLAMLAB_EVALUATION, self.name)
        self.num_runs = settings.get('NumRuns', 1)
        self.module = settings.get('Module', "default")
        self.parameters = settings['Parameters']

        self.log_csv = os.path.join(self.folder, 'vslamlab_exp_log.csv')
        self.config_yaml = os.path.join(VSLAM_LAB_DIR, 'configs', settings.get('Config', CONFIG_DEFAULT))
        self.ablation_csv = settings.get('Ablation', None)

##################################################################################################################################################
# compare_exp
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
def compare_exp(exp_yaml: str | Path) -> None:
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)

    experiments = {}
    for exp_name, settings in exp_data.items():
        experiments[exp_name] = Experiment(exp_name, settings)

    comparison_path = os.path.join(VSLAMLAB_EVALUATION, f"comp_{str(os.path.basename(exp_yaml)).replace('.yaml', '')}")
    print_msg(f"\n{SCRIPT_LABEL}", f"Comparing (in {comparison_path}) ...")
    if os.path.exists(comparison_path):
        shutil.rmtree(comparison_path)
    os.makedirs(comparison_path)
    os.makedirs(os.path.join(comparison_path, 'figures'))
    compare_functions.full_comparison(experiments, VSLAMLAB_BENCHMARK, COMPARISONS_YAML_DEFAULT, comparison_path)

##################################################################################################################################################
# eval_exp
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
def evaluate_exp(exp_yaml: str | Path, overwrite: bool = False) -> None:
    
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)
    experiments = {}
    first_evaluation_found = True
    for exp_name, settings in exp_data.items():
        experiments[exp_name] = Experiment(exp_name, settings)

    for [_, exp] in experiments.items():
        exp_log = read_csv(exp.log_csv)
        if(not exp_log['EVALUATION'].str.contains('none').any()) and not overwrite:
            continue
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    if first_evaluation_found:
                        print_msg(f"\n{SCRIPT_LABEL}", f"Evaluating (in {VSLAMLAB_EVALUATION}) ...")
                        first_evaluation_found = False
                    evaluate_sequence(exp, dataset, sequence_name, overwrite)

##################################################################################################################################################
# run_exp
##################################################################################################################################################
##################################################################################################################################################
def run_exp(exp_yaml: str | Path) -> None:
    start_time = time.time()

    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)

    experiments = {}
    for exp_name, settings in exp_data.items():
        experiments[exp_name] = Experiment(exp_name, settings)

    all_experiments_completed = {exp_name: False for exp_name in experiments}
    completed_runs = {}
    not_completed_runs = {}
    num_executed_runs = 0
    duration_time_total = 0
    while not all(all_experiments_completed.values()):

        remaining_iterations = 0
        for [exp_name, exp] in experiments.items():
            exp_log = read_csv(exp.log_csv)
            completed_runs[exp_name] = (exp_log["STATUS"] == "completed").sum()  
            not_completed_runs[exp_name] = (exp_log["STATUS"] != "completed").sum() 
            remaining_iterations += not_completed_runs[exp_name]
            
            if not_completed_runs[exp_name] == 0:
                all_experiments_completed[exp_name] = True
                continue
            
            first_not_finished_experiment = exp_log[exp_log["STATUS"] != "completed"].index.min()
            row = exp_log.loc[first_not_finished_experiment]
            baseline = get_baseline(row['method_name'])
            dataset = get_dataset(row['dataset_name'], VSLAMLAB_BENCHMARK)    

            if num_executed_runs == 0:
                print(f"\n{SCRIPT_LABEL}Running experiments (in {exp_yaml}) ...")
            results = run_sequence(row['exp_it'], exp, baseline, dataset, row['sequence_name'])

            duration_time = results['duration_time']
            duration_time_total += duration_time
            num_executed_runs += 1
            remaining_iterations -= 1

            exp_log["STATUS"] = exp_log["STATUS"].astype(str)
            exp_log["SUCCESS"] = exp_log["SUCCESS"].astype(str)
            exp_log["COMMENTS"] = exp_log["COMMENTS"].astype(str)
            exp_log.loc[first_not_finished_experiment, "STATUS"] = "completed"
            exp_log.loc[first_not_finished_experiment, "SUCCESS"] = results['success']
            exp_log.loc[first_not_finished_experiment, "COMMENTS"] = results['comments']
            exp_log.loc[first_not_finished_experiment, "TIME"] = duration_time
            exp_log.loc[first_not_finished_experiment, "RAM"] = results['ram']
            exp_log.loc[first_not_finished_experiment, "SWAP"] = results['swap']
            exp_log.loc[first_not_finished_experiment, "GPU"] = results['gpu']
            exp_log.to_csv(exp.log_csv, index=False)
                
            all_experiments_completed[exp_name] = exp_log['STATUS'].eq("completed").all()
        
        if(duration_time_total > 1):
            print(f"\n{SCRIPT_LABEL}: Experiment report: {exp_yaml}")
            print(f"{ws(4)}\033[93mNumber of executed iterations: {num_executed_runs} / {num_executed_runs + remaining_iterations} \033[0m")
            print(f"{ws(4)}\033[93mAverage time per iteration: {show_time(duration_time_total / num_executed_runs)}\033[0m")
            print(f"{ws(4)}\033[93mTotal time consumed: {show_time(duration_time_total)}\033[0m")
            print(f"{ws(4)}\033[93mRemaining time until completion: {show_time(remaining_iterations * duration_time_total / num_executed_runs)}\033[0m")

    if num_executed_runs > 0:
        run_time = (time.time() - start_time)
        print(f"\033[93m[Experiment runtime: {show_time(run_time)}]\033[0m")

##################################################################################################################################################
# download_sequence
##################################################################################################################################################
##################################################################################################################################################

def download_sequence(dataset_name: str, sequence_name: str) -> None:
    dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
    dataset.download_sequence(sequence_name)

def download_sequences(dataset_sequence_name: list[str]) -> None:
    for i in range(0, len(dataset_sequence_name), 2):
        dataset_name = dataset_sequence_name[i]
        sequence_name = dataset_sequence_name[i + 1]
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        dataset.download_sequence(sequence_name)

def download_dataset(dataset_name: str) -> None:
    dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
    for sequence_name in dataset.get_sequence_names():
        dataset.download_sequence(sequence_name)

def download_datasets(dataset_names: list[str]) -> None:
    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        for sequence_name in dataset.get_sequence_names():
            dataset.download_sequence(sequence_name)

##################################################################################################################################################
# install_baseline
##################################################################################################################################################
##################################################################################################################################################

def install_baseline(baseline_name: list[str]) -> None:
    baseline = get_baseline(baseline_name)
    is_baseline_installed, _ = baseline.is_installed()
    if not is_baseline_installed:
        baseline.git_clone()
        baseline.install()

def install_baselines(baselines_to_install: str) -> None:
    for baseline_name in baselines_to_install:
        install_baseline(baseline_name)

##################################################################################################################################################
# check_experiment_state
##################################################################################################################################################
##################################################################################################################################################


def check_experiment_state(exp_yaml: str | Path) -> None:
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment running state: {exp_yaml}", verb='LOW')

    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)

    total_num_runs = 0
    executed_num_runs = 0
    
    for exp_name, settings in exp_data.items():
        exp_folder = VSLAMLAB_EVALUATION / exp_name
        exp_log_csv = exp_folder / "vslamlab_exp_log.csv"

        total_num_runs_exp = settings.get('NumRuns')
        total_num_runs += total_num_runs_exp

        executed_num_runs_exp = 0
        if exp_folder.exists() & exp_log_csv.exists():
            exp_log = pd.read_csv(exp_log_csv)
            executed_num_runs_exp += (exp_log["STATUS"] == "completed").sum()  
            executed_num_runs += executed_num_runs_exp
        
        if executed_num_runs_exp == total_num_runs_exp:
            print(f"{ws(4)}- {exp_name}: \033[92m{executed_num_runs_exp} / {total_num_runs_exp} ({100 * executed_num_runs_exp/total_num_runs_exp} %)\033[0m")
        else:
            print(f"{ws(4)}- {exp_name}: \033[93m{executed_num_runs_exp} / {total_num_runs_exp} ({100 * executed_num_runs_exp/total_num_runs_exp} %)\033[0m")


##################################################################################################################################################
# check_experiment_resources
##################################################################################################################################################
##################################################################################################################################################
def check_experiment_baselines_installed(exp_data: Any, exp_yaml: str | Path) -> tuple[int, int, list[str]]:
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment baselines: {exp_yaml}", verb='LOW')
    baselines: dict[str, str] = {}
    for exp_name, settings in exp_data.items():
        baselines[settings.get('Module')] = exp_name

    num_baselines_to_install = 0
    baselines_to_install = []
    for baseline_name, exp_name in baselines.items():
        baseline = get_baseline(baseline_name)
        is_baseline_installed, install_msg = baseline.is_installed()
        if is_baseline_installed:
            print_msg(f"{ws(4)}", f"- {baseline.label}:\033[92m {install_msg}\033[0m", verb='LOW')
        else:    
            print_msg(f"{ws(4)}", f"- {baseline.label}:\033[93m {install_msg}\033[0m", verb='LOW')
            num_baselines_to_install += 1
            baselines_to_install.append(baseline_name)

    num_automatic_install = num_baselines_to_install
    return num_baselines_to_install, num_automatic_install, baselines_to_install

def check_experiment_sequences_available(exp_data: Any, exp_yaml: str | Path) -> tuple[int, int, list[str]]:
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment sequences: {exp_yaml}", verb='LOW')
    
    configs: set[str] = set()
    for _, settings in exp_data.items():
        configs.add(settings.get('Config'))

    sequences : dict[str, str] = {}
    for config_yaml in configs:
        config_file = os.path.join(VSLAM_LAB_DIR, 'configs', config_yaml)
        with open(config_file, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names: 
                    sequences[sequence_name] = dataset_name
    
    # Check sequence availability
    sequences_to_download = {}
    num_total_sequences = len(sequences)
    num_available_sequences = 0
    for sequence_name, dataset_name in sequences.items():
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        if dataset_name not in sequences_to_download:
            sequences_to_download[dataset_name] = []
        if dataset.check_sequence_availability(sequence_name, False) == "available":
            print_msg(f"{ws(4)}", f"- {dataset.dataset_label} {dataset.dataset_color}{sequence_name}:\033[92m available\033[0m", verb='MEDIUM')
            num_available_sequences += 1
        else:
            print_msg(f"{ws(4)}", f"- {dataset.dataset_label} {dataset.dataset_color}{sequence_name}:\033[93m not available\033[0m \033[92m (automatic download)\033[0m", verb='MEDIUM')
            sequences_to_download[dataset_name].append(sequence_name)
    print_msg(f"{ws(4)}", f"- Sequences available: \033[92m{num_available_sequences}\033[0m / {num_total_sequences}", verb='LOW')
    if num_available_sequences < num_total_sequences:
        print_msg(f"{ws(4)}", f"- Sequences to download: \033[93m{num_total_sequences - num_available_sequences}\033[0m / {num_total_sequences}", verb='LOW')

    # Check download issues
    num_download_issues = 0
    num_automatic_solution = 0
    first_download_issue_found = False
    for dataset_name, sequence_names in sequences_to_download.items():
        if sequence_names == []:
            continue
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        issues_seq = dataset.get_download_issues(sequence_names)
        if issues_seq:
            if not first_download_issue_found:
                print_msg(f"\n{ws(4)}", "LIST OF DOWNLOAD ISSUES:",'warning')
            first_download_issue_found = True
        for issue_seq in issues_seq:
            print_msg(f"{ws(4)}", f"- [{dataset_name}][{issue_seq['name']}]: {issue_seq['description']}",'warning')
            print(f"{ws(8)}[{issue_seq['mode']}]: {issue_seq['solution']}")
            num_download_issues += 1
            if issue_seq['mode'] == '\033[92mautomatic download\033[0m':
                num_automatic_solution += 1

    return num_download_issues, num_automatic_solution, sequences_to_download

def check_experiment_resources(exp_yaml: str | Path) -> tuple[list[str], list[str]]:
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)
   
    num_baselines_to_install, num_automatic_install, baselines_to_install = check_experiment_baselines_installed(exp_data, exp_yaml)
    num_download_issues, num_automatic_download, sequences_to_download = check_experiment_sequences_available(exp_data, exp_yaml)
    
    if num_baselines_to_install > 0 or num_download_issues > 0:
        print_msg(f"\n{SCRIPT_LABEL}",f"Your experiments have {num_baselines_to_install} install issues and {num_download_issues} download issues:",'warning')
        if(num_baselines_to_install - num_automatic_install) > 0:
            print_msg(f"{ws(4)}",f"- {num_baselines_to_install - num_automatic_install} baselines need to be installed manually.",'warning')
            print_msg(f"{ws(4)}", f"Some issues are  not automatically fixable. Please, fix them manually and run the experiment again.",'error')
            exit(1)
        if num_download_issues - num_automatic_download > 0:
            print_msg(f"{ws(4)}", f"Some issues are  not automatically fixable. Please, fix them manually and run the experiment again.",'error')
            exit(1)
        
        print(f"{ws(4)}All issues are \033[92mautomatically\033[0m fixable.")

    return baselines_to_install, sequences_to_download, num_download_issues

def get_experiment_resources(exp_yaml: str | Path) -> None:
    baselines_to_install, sequences_to_download, num_download_issues = check_experiment_resources(exp_yaml)
    if num_download_issues > 0:
        #print_msg(f"\n{SCRIPT_LABEL}",f"Your experiments have {num_baselines_to_install} install issues and {num_download_issues} download issues:",'warning')
        message = (
            f"{SCRIPT_LABEL}"
            f"There {'is' if num_download_issues == 1 else 'are'} "
            f"{num_download_issues} download/install issue"
            f"{'' if num_download_issues == 1 else 's'} "
            "\033[92mautomatically\033[0m fixable. "
            "Would you like to continue solving them (Y/n): "
        )
        #message = (f"{SCRIPT_LABEL}There is {num_download_issues} download/install issues \033[92mautomatically\033[0m fixable. Would you like to continue solving them (Y/n): ")
        try:
            user_input = inputimeout(prompt=message, timeout=60*10).strip()
        except TimeoutOccurred:
            user_input = 'Y'
            print(f"{ws(4)}No input detected. Defaulting to 'Y'.")
        if user_input == 'n':
            exit() 

    install_baselines(baselines_to_install)

    first_time = True
    for dataset_name, sequence_names in sequences_to_download.items():
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        for sequence_name in sequence_names:
            if first_time:
                print(f"\n{SCRIPT_LABEL}Downloading (to {VSLAMLAB_BENCHMARK}) ...")
                first_time = False
            dataset.download_sequence(sequence_name)

##################################################################################################################################################
# update_experiment_csv_log
##################################################################################################################################################
##################################################################################################################################################
def update_experiment_csv_log(exp_name: str, settings: Any) -> bool:
    updated = False

    exp_folder = VSLAMLAB_EVALUATION / exp_name
    exp_log_csv = exp_folder / "vslamlab_exp_log.csv"
    exp_log = pd.read_csv(exp_log_csv)

    baseline_name = settings.get('Module')
    all_match = exp_log["method_name"].eq(baseline_name).all()
    if not all_match:
        print_msg(f"{SCRIPT_LABEL}", f"The original method cannot be changed ({(exp_log["method_name"][0])} != {baseline_name}). Only new sequences or more runs can be added to the experiment.",'error')
        exit(1)

    config_yaml = settings.get('Config')
    config_file = os.path.join(VSLAM_LAB_DIR, 'configs', config_yaml)
    num_runs = settings.get('NumRuns')
    with open(config_file, 'r') as file:
        config_file_data = yaml.safe_load(file)
        for dataset_name, sequence_names in config_file_data.items():
            for sequence_name in sequence_names: 
                for iRun in range(0, num_runs):
                        subset = exp_log[
                            (exp_log["dataset_name"] == dataset_name) &
                            (exp_log["sequence_name"] == sequence_name) &
                            (exp_log["exp_it"] == iRun)]

                        if subset.empty:
                            updated = True
                            new_row = {
                                "method_name": baseline_name,
                                "dataset_name": dataset_name,
                                "sequence_name": sequence_name,
                                "exp_it": iRun,
                                "STATUS": "",
                                "SUCCESS": "",
                                "TIME": "0.0",
                                "RAM": "0.0",
                                "SWAP": "0.0",
                                "GPU": "0.0",
                                "COMMENTS": "",
                                "EVALUATION": "none",
                                "num_frames": "0",
                                "num_tracked_frames": "0",
                                "num_evaluated_frames": "0",
                            }
                            exp_log = pd.concat(
                                [exp_log, pd.DataFrame([new_row])],
                                ignore_index=True,
                            )
    if updated:
        exp_log.to_csv(exp_log_csv, index=False)
    return updated
         
def create_experiment_csv_log(exp_name: str, settings: Any) -> None:
    exp_folder = VSLAMLAB_EVALUATION / exp_name
    exp_log_csv = exp_folder / "vslamlab_exp_log.csv"
    if not exp_folder.exists():
        exp_folder.mkdir(parents=True, exist_ok=True)

    if exp_log_csv.exists(): 
        return

    log_headers = ["method_name", "dataset_name", "sequence_name", "exp_it", "STATUS", "SUCCESS", "TIME", "RAM", "SWAP", "GPU", "COMMENTS", "EVALUATION"]

    with open(exp_log_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_headers)
        config_yaml = os.path.join(VSLAM_LAB_DIR, "configs", settings.get("Config"))
        num_runs = settings.get("NumRuns")
        baseline_name = settings.get("Module")
        with open(config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for i in range(num_runs):
                for dataset_name, sequence_names in config_file_data.items():
                    for sequence_name in sequence_names:
                        exp_it = str(i).zfill(5)  
                        writer.writerow([baseline_name, dataset_name, sequence_name, f"{exp_it}", "", "",0.0, 0.0, 0.0, 0.0, "", "none"])

def update_experiment_csv_logs(exp_yaml: str | Path) -> None:
    
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)

    num_updates = 0
    for exp_name, settings in exp_data.items():
        exp_folder = VSLAMLAB_EVALUATION / exp_name
        exp_log_csv = exp_folder / "vslamlab_exp_log.csv"
        if not exp_folder.exists():
            exp_folder.mkdir(parents=True, exist_ok=True)

        if not exp_log_csv.exists(): 
            if num_updates == 0:
                print_msg(f"\n{SCRIPT_LABEL}", f"Update experiment csv logs: {exp_yaml}", verb='LOW')
            print(f"{ws(4)}- \033[92mCreate new\033[0m: {exp_log_csv}")
            create_experiment_csv_log(exp_name, settings)
            num_updates += 1
        elif update_experiment_csv_log(exp_name, settings):
            if num_updates == 0:
                print_msg(f"\n{SCRIPT_LABEL}", f"Update experiment csv logs: {exp_yaml}", verb='LOW')
            print(f"{ws(4)}- \033[92mUpdate\033[0m: {exp_log_csv}")
            num_updates += 1

    if num_updates == 0:
        print_msg(f"\n{SCRIPT_LABEL}", f"Update experiment csv logs: {exp_yaml} : \033[92mEverything up-to-date\033[0m", verb='LOW')

##################################################################################################################################################
# overwrite_exp
##################################################################################################################################################
##################################################################################################################################################
def overwrite_exp(exp_yaml: str | Path) -> None:
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)
    print_msg(f"\n{SCRIPT_LABEL}", f"Overwrite experiment: '{exp_yaml}'", "warning")
    for exp_name, _ in exp_data.items():
        exp_folder = VSLAMLAB_EVALUATION / exp_name
        exp_folder.mkdir(parents=True, exist_ok=True)
        print_msg(ws(4), f"- Delete:'{exp_folder}'", "warning")

        for item in exp_folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

##################################################################################################################################################
# validate_experiment_yaml
##################################################################################################################################################
##################################################################################################################################################

###################### Check experiment syntax ######################
def check_experiment_baseline_names(exp_data: Any, exp_yaml: str | Path) -> None:
    errors: list[str] = []
    for exp_name, settings in exp_data.items():
        baseline_name = settings.get("Module")
        baseline = get_baseline(baseline_name)
        if baseline == "Invalid case":
            errors.append(
                f"[Error] Module: '{baseline_name}' baseline in '{exp_name}' doesn't exist."
            )

    if not errors:
        return

    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment baseline names (in '{exp_yaml}'):", "info")
    for error in errors:
        print_msg(ws(4), error, "error")
    print_baselines()
    sys.exit(1)

def check_experiment_sequence_names(exp_data: Any, exp_yaml: str | Path) -> None:
    errors: list[str] = []
    configs: set[str] = set()
    for _, settings in exp_data.items():
        config_yaml = settings.get("Config")
        configs.add(config_yaml)
    
    dataset_list = set(list_available_datasets())
    
    for config_yaml in configs:
        config_file = VSLAM_LAB_DIR / 'configs' / config_yaml
        config_file_data = load_yaml_file(config_file)
        
        for dataset_name, sequence_names in config_file_data.items():
            if dataset_name not in dataset_list:
                errors.append(f"[Error] Dataset '{dataset_name}' doesn't exist (in config '{config_file}').")
                continue

            dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)

            for sequence_name in sequence_names:
                if not dataset.contains_sequence(sequence_name):
                    errors.append(
                        f"[Error] Sequence '{sequence_name}' in dataset '{dataset_name}' doesn't exist (in config '{config_file}')."
                    )

    if not errors:
        return
    
    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment dataset and sequence names (in '{exp_yaml}'):", "info")
    for error in errors:
        print_msg(ws(4), error, "error")

    print_datasets()
    sys.exit(1)    

###################### Check experiment conflicts ######################
def check_experiment_baselines_conflicts(exp_data:  Any, exp_yaml: str | Path,) -> str:

    errors: list[str] = []
    modes: list[str] = []

    for exp_name, settings in exp_data.items():
        baseline_name = settings.get("Module")
        baseline = get_baseline(baseline_name)
    
        mode = (settings.get("Parameters", {}).get("mode") or baseline.default_parameters.get("mode"))
        if not mode in modes:
            modes.append(mode)

        if mode not in baseline.modes:
            errors.append(
                f"[Error] Baseline '{baseline_name}' in '{exp_name}' doesn't handle "
                f"mode '{mode}'. Available modes are: {baseline.modes}."
            )
    if len(modes) > 1:
        errors.append(f"[Error] Only one mode is allowed per config file. Conflicts: {modes}")

    if errors:
        print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment baseline conflicts (in '{exp_yaml}'):", "info")
        for error in errors:
            print_msg(ws(4), error, "error")
        sys.exit(1)

    return modes[0]

def check_experiment_sequence_conflicts(exp_data:  Any, exp_yaml: str | Path, config_mode: str) -> None:
    errors: list[str] = []
    configs: set[str] = set()
    for _, settings in exp_data.items():
        config_yaml = Path(settings.get("Config"))
        configs.add(config_yaml)

    for config_yaml in configs:
        config_file = VSLAM_LAB_DIR / 'configs' / config_yaml
        config_file_data = load_yaml_file(config_file)
    
        for dataset_name in config_file_data.keys():
            dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
            if config_mode not in dataset.modes:
                errors.append(
                    f"[Error] Dataset '{dataset_name}' (in config '{config_file}') doesn't handle mode "
                    f"'{config_mode}'. Available modes are: {dataset.modes}."
                )

    if not errors:
        return

    print_msg(f"\n{SCRIPT_LABEL}", f"Checking experiment dataset conflicts (in '{exp_yaml}'):", "info")
    for error in errors:
        print_msg(ws(4), error, "error")

    sys.exit(1)

def validate_experiment_yaml(exp_yaml: str | Path) -> None:
    # Load experiments
    exp_yaml = Path(exp_yaml)
    exp_data = load_yaml_file(exp_yaml)
   
    # Check syntax
    check_experiment_baseline_names(exp_data, exp_yaml)
    check_experiment_sequence_names(exp_data, exp_yaml)

    # Check conflicts
    config_mode = check_experiment_baselines_conflicts(exp_data, exp_yaml)
    check_experiment_sequence_conflicts(exp_data, exp_yaml, config_mode)

    # Print Summary
    print_msg(f"\n{SCRIPT_LABEL}", f"Experiment summary: {exp_yaml}", flag="info", verb='NONE')
    for exp_name, settings in exp_data.items():
        baseline_name = settings.get("Module")
        config = settings.get("Config")
        numRuns = settings.get("NumRuns")
        print(f"{ws(4)} - {exp_name}: \033[96m{baseline_name}\033[0m, \033[38;2;255;165;0m {config}\033[0m x{numRuns}")            