from Datasets.get_dataset import get_dataset
from Baselines.get_baseline import get_baseline
from Evaluate.BenchmarkVSLAMLAB import BenchmarkVSLAMLAB as BM

import numpy as np
import pandas as pd

def generate_latex_tables(values, dataset_sequences, metric, figures_path, experiments):

    MIN_TRAJ_COVERAGE = 0.75
    TRAJ_UNIT = 100
    exp_names = list(experiments.keys())

    all_seq_keys = []        # list of (dataset_name, seq_name) keys for indexing values
    all_seq_labels = []      # list of pretty names for LaTeX header
    for dataset_name, sequence_names in dataset_sequences.items():
        dataset = get_dataset(dataset_name)
        for sequence_name in sequence_names:
            all_seq_keys.append((dataset_name, sequence_name))
            all_seq_labels.append(dataset.get_sequence_nickname(sequence_name))
    
    row_labels = []
    vals_np = np.full((len(exp_names), len(all_seq_keys)), np.nan, dtype=float)
    vals_np_bm = np.full((len(exp_names), len(all_seq_keys)), np.nan, dtype=float)
    vals_np_perc = np.full((len(exp_names), len(all_seq_keys)), np.nan, dtype=float)
    for i, (exp_name, experiment) in enumerate(experiments.items()):
        baseline = get_baseline(experiment.module)
        row_labels.append(baseline.baseline_name)

        for j, (dataset_name, seq_name) in enumerate(all_seq_keys):
            # values[...] assumed to be an array-like over frames/runs for this metric

            if values[dataset_name][seq_name][exp_name].empty:
                values[dataset_name][seq_name][exp_name] = pd.DataFrame([{
                    "rmse": 0.0,                 # or 0.0 if you want a numeric default
                    "num_tracked_frames": 1,   # or 0
                    "num_frames": 100000
                }])
            
            arr = values[dataset_name][seq_name][exp_name][metric]
            arr_filter_num = np.median(values[dataset_name][seq_name][exp_name]['num_tracked_frames'])
            arr_filter_den = int(np.median(values[dataset_name][seq_name][exp_name]['num_frames']) / 50) + 1

            num = pd.to_numeric(arr_filter_num, errors="coerce")
            den = pd.to_numeric(arr_filter_den, errors="coerce")
            if pd.isna(num) or pd.isna(den) or den == 0:
                perc = np.nan   # or 0.0 if you prefer to show "no trajectory"
            else:
                perc = float(num) / float(den)

            # median per your code
            if perc < MIN_TRAJ_COVERAGE:
                vals_np[i, j] = np.nan
            else:
                vals_np[i, j] = TRAJ_UNIT * np.median(arr)

            median_bm = BM().get_median_ate(baseline.baseline_name, dataset_name, seq_name)
            if median_bm == 0.0:
                vals_np_bm[i, j] = np.nan
            else:
                vals_np_bm[i, j] = TRAJ_UNIT *median_bm
            
            vals_np_perc[i, j] = 100.0 * perc


    # ----- Build the LaTeX table from vals_np -----
    col_spec = "l" + "c" * len(all_seq_labels)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = [f"Dataset (mono)"] + [s.replace("_", r"\_") for s in all_seq_labels]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    col_mins = np.nanmin(vals_np, axis=0)

    for i, name in enumerate(row_labels):
        row = [name.replace("_", r"\_")]
        for j in range(vals_np.shape[1]):
            v = vals_np[i, j]
            if np.isnan(v):
                cell = "--"
            else:
                # bold if equals column min (with tolerance for float comparisons)
                if np.isclose(v, col_mins[j], rtol=1e-9, atol=1e-12):
                    cell = rf"$\textbf{{{v:.1f}}}"
                else:
                    cell = f"${v:.1f}"

                if vals_np_perc[i, j] > MIN_TRAJ_COVERAGE:
                    cell = cell + f"_{{{vals_np_perc[i, j]:.0f}}}$"
                else:
                    cell = cell + "$"

            # if name != 'allfeature-dev':
            #     bm_value = vals_np_bm[i, j]
            #     if np.isnan(bm_value):
            #         cell = cell + " / --"
            #     else:
            #         cell = cell + f" / \\textit{{{bm_value:.1f}}}"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    latex_table = "\n".join(lines)
    print(latex_table)
