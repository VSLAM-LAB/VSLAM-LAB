"""
Module: VSLAM-LAB - Compare - plot_functions.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2024-07-04
- Updated: 2024-07-04
- License: GPLv3 License
- List of Known Dependencies;
    * ...

This module provides functions for generating various types of plots to visualize experiment data across multiple datasets and sequences.

Functions included:
- boxplot_exp_seq: Generates box plots for different experiments and sequences within multiple datasets.
- radar_seq: Creates a radar plot showing the relative performance across different sequences and datasets based on a specified metric.
- plot_cum_error: Generates and saves cumulative error plots for different datasets, sequences, and experiments.
- create_and_show_canvas: Creates a canvas of resized images and displays it.
"""

import glob
import math
import os
import random
from bisect import bisect_left, bisect_right
from math import pi

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from path_constants import VSLAM_LAB_EVALUATION_FOLDER, VSLAMLAB_BENCHMARK
from utilities import list_image_files_in_folder
from Baselines.get_baseline import get_baseline
from Datasets.get_dataset import get_dataset

import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import to_hex

random.seed(6)
colors_all = mcolors.CSS4_COLORS
colors = list(colors_all.keys())
random.shuffle(colors)

import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)


def copy_axes_properties(source_ax, target_ax):
    for line in source_ax.get_lines():
        target_ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle())

    for patch in source_ax.patches:
        new_patch = patch.__class__(xy=patch.get_xy(), width=patch.get_width(), height=patch.get_height(),
                                    color=patch.get_facecolor())
        target_ax.add_patch(new_patch)

    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())

    target_ax.set_xticks(source_ax.get_xticks())
    target_ax.set_xticklabels(source_ax.get_xticklabels())


def plot_trajectories(dataset_sequences, exp_names, 
                      dataset_nicknames, experiments,
                        accuracies, comparison_path):
    num_trajectories = 0
    for i_dataset, (dataset_name, sequence_names) in enumerate(dataset_sequences.items()):
        for i_sequence, sequence_name in enumerate(sequence_names):
            num_trajectories = num_trajectories + 1

    # Figure dimensions
    num_rows = math.ceil(num_trajectories / 5)
    xSize = 12
    ySize = num_rows * 2

    fig, axs = plt.subplots(num_rows, 5, figsize=(xSize, ySize))
    axs = axs.flatten()

    # Create legend handles
    legend_handles = []
    legend_handles.append(Patch(color='black', label='gt'))
    for i_exp, exp_name in enumerate(exp_names):
        legend_handles.append(Patch(color=colors[i_exp], label=exp_names[i_exp]), )

    i_traj = 0
    there_is_gt = False
    for i_dataset, (dataset_name, sequence_names) in enumerate(dataset_sequences.items()):
        for i_sequence, sequence_name in enumerate(sequence_names):
            #x_max , y_max = 0, 0
            aligment_with_gt = False
            for i_exp, exp_name in enumerate(exp_names):
                vslam_lab_evaluation_folder_seq = os.path.join(experiments[exp_name].folder, dataset_name.upper(),
                                                               sequence_name, VSLAM_LAB_EVALUATION_FOLDER)

                if accuracies[dataset_name][sequence_name][exp_name].empty:
                    continue

                if not aligment_with_gt:                   
                    accu = accuracies[dataset_name][sequence_name][exp_name]['rmse'] / accuracies[dataset_name][sequence_name][exp_name]['num_tracked_frames']
                    idx = accu.idxmin()
                    gt_file = os.path.join(vslam_lab_evaluation_folder_seq, f'{idx:05d}_gt.tum')
                    there_is_gt = False
                    if os.path.exists(gt_file):
                        there_is_gt = True
                        gt_traj = pd.read_csv(gt_file, delimiter=' ')
                        pca_df = pd.DataFrame(gt_traj, columns=['tx', 'ty', 'tz'])
                        pca = PCA(n_components=2)
                        pca.fit(pca_df)
                        gt_transformed = pca.transform(pca_df)
                        x_shift = 1.2*gt_transformed[:, 0].min()
                        y_shift = 1.2* gt_transformed[:, 1].min()
                        x_max = 1.2* gt_transformed[:, 0].max() - x_shift
                        y_max = 1.2* gt_transformed[:, 1].max() - y_shift
                        axs[i_traj].plot(gt_transformed[:, 0]-x_shift, gt_transformed[:, 1]-y_shift, label='gt', linestyle='-', color='black')
                    else:
                        x_shift = 0
                        y_shift = 0
                        x_max = 1
                        y_max = 1
                    aligment_with_gt = True

                search_pattern = os.path.join(vslam_lab_evaluation_folder_seq, '*_KeyFrameTrajectory.tum*')
                files = glob.glob(search_pattern)
        
                aligned_traj = pd.read_csv(files[idx], delimiter=' ')
                pca_df = pd.DataFrame(aligned_traj, columns=['tx', 'ty', 'tz'])
                if len(files) == 0:
                    continue
                if there_is_gt:
                    traj_transformed = pca.transform(pca_df)
                else:
                    traj_transformed = pca_df
                    traj_transformed = traj_transformed.to_numpy()

                baseline = get_baseline(experiments[exp_name].module)
                axs[i_traj].plot(traj_transformed[:, 0]-x_shift, traj_transformed[:, 1]-y_shift,
                                    label=exp_name, marker='.', linestyle='-', color=baseline.color)

            x_ticks = [round(x_max, 1)]
            y_ticks = [0,round(y_max, 1)]
            axs[i_traj].set_xticks(x_ticks)
            axs[i_traj].set_yticks(y_ticks)

            # Format tick labels to 1 decimal place
            axs[i_traj].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            axs[i_traj].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

            # Add minor ticks for the grid (every 10% of the axis range)
            axs[i_traj].xaxis.set_minor_locator(ticker.MultipleLocator(x_max / 4))
            axs[i_traj].yaxis.set_minor_locator(ticker.MultipleLocator(y_max / 4))

            # Enable the grid for both major and minor ticks, but keep labels only for major ticks
            axs[i_traj].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            axs[i_traj].spines['top'].set_visible(False)   # Remove top border
            axs[i_traj].spines['right'].set_visible(False) # Remove right border
            #axs[i_traj].spines['left'].set_visible(False)  # Remove left border (optional)
            #axs[i_traj].spines['bottom'].set_visible(False) # Remove bottom border (optional)

            # Hide minor tick labels while keeping the minor grid lines
            axs[i_traj].tick_params(axis='both', which='minor', labelbottom=False, labelleft=False)
            axs[i_traj].tick_params(axis='y', labelsize=20, rotation=90) 
            axs[i_traj].tick_params(axis='x', labelsize=20, rotation=0) 

            axs[i_traj].tick_params(axis='x', pad=10) 
            axs[i_traj].set_xticklabels([f"{x_ticks[0]:.2f}"], ha='right')  
            axs[i_traj].set_yticklabels([f"{y_ticks[0]:.0f}",f"{y_ticks[1]:.2f}"])  
            
            i_traj = i_traj + 1


    plt.tight_layout()
    plot_name = os.path.join(comparison_path, f"trajectories.pdf")
    plt.savefig(plot_name, format='pdf')

    i_traj = 0
    for i_dataset, (dataset_name, sequence_names) in enumerate(dataset_sequences.items()):
        for i_sequence, sequence_name in enumerate(sequence_names):
            for i_exp, exp_name in enumerate(exp_names):
                axs[i_traj].set_title(dataset_nicknames[dataset_name][i_sequence])
            i_traj = i_traj + 1    
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles))
    plt.subplots_adjust(bottom=0.3)
    plt.show(block=False)



def boxplot_exp_seq(values, dataset_sequences, metric_name, comparison_path, experiments, shared_scale = False):

    def set_format(tick):
        if tick == 0:
            return f"0"
        return f"{tick:.1e}"

    # Get number of sequences
    num_sequences = 0
    splts = {}
    for dataset_name, sequence_names in dataset_sequences.items():
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        for sequence_name in sequence_names:
            splts[sequence_name]= {}
            splts[sequence_name]['id']= num_sequences
            splts[sequence_name]['dataset_name']= dataset_name
            splts[sequence_name]['nickname']= dataset.get_sequence_nickname(sequence_name)
            splts[sequence_name]['success']= True
            num_sequences += 1

    exp_names = list(experiments.keys())

    # Figure dimensions
    NUM_COLS = 5
    NUM_ROWS = math.ceil(num_sequences / NUM_COLS)
    XSIZE, YSIZE = 12, 2 * NUM_ROWS + 0.5
    WIDTH_PER_SERIES = min(XSIZE / len(exp_names), 0.4)
    FONT_SIZE = 15
    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(XSIZE, YSIZE))
    axs = axs.flatten()

    # Create legend handles
    legend_handles = []
    colors = {}
    for i_exp, exp_name in enumerate(exp_names):
        baseline = get_baseline(experiments[exp_name].module)   
        colors[exp_name] = baseline.color
        legend_handles.append(Patch(color=colors[exp_name], label=exp_names[i_exp]))
        
    # Plot boxplots
    whisker_min = {}
    whisker_max = {}
    for sequence_name, splt in splts.items():
        whisker_min_seq, whisker_max_seq = float('inf'), float('-inf')
        for i_exp, exp_name in enumerate(exp_names):

            values_seq_exp = values[splt['dataset_name']][sequence_name][exp_name]
            if values_seq_exp.empty:
                continue
            boxprops = medianprops = whiskerprops = capprops = dict(color=colors[exp_name])
            flierprops = dict(marker='o', color=colors[exp_name], alpha=1.0)
            positions = [i_exp * WIDTH_PER_SERIES]   
            boxplot_accuracy = axs[splt['id']].boxplot(
                values_seq_exp[metric_name],
                positions=positions, widths=WIDTH_PER_SERIES,
                patch_artist=False,
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops, flierprops=flierprops)
            whisker_values = [line.get_ydata()[1] for line in boxplot_accuracy['whiskers']]
            whisker_min_seq = min(whisker_min_seq, min(whisker_values))
            whisker_max_seq = max(whisker_max_seq, max(whisker_values))

        width = max(0.1 * (whisker_max_seq - whisker_min_seq), 1e-6)
        if np.isinf(whisker_max_seq) or np.isinf(whisker_min_seq):
            splts[sequence_name]['success']= False
            whisker_max[sequence_name] = np.nan
            whisker_min[sequence_name] = np.nan
        else:
            # if whisker_max_seq + width > 0.055 and ('56a0ec536c' in sequence_name) :
            #     whisker_max[sequence_name] = 0.03
            #     whisker_min[sequence_name] = 0.02
            # else:
            whisker_max[sequence_name] = whisker_max_seq + width
            if(whisker_min_seq - width < 0):    
                whisker_min[sequence_name] = whisker_min_seq / 2
            else:
                whisker_min[sequence_name] = whisker_min_seq - width
                         
    # Adjust plot properties for paper
    max_value, min_value = max(whisker_max.values()), min(whisker_min.values())

    if shared_scale:
        whisker_max = {key: max_value for key in whisker_max}
        whisker_min = {key: 0 for key in whisker_min}

    for sequence_name, splt in splts.items():
        if splt['success'] == False:
            axs[splt['id']].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            axs[splt['id']].set_xticklabels([])
            axs[splt['id']].set_yticklabels([])
            continue

        whisker_max_seq = whisker_max[sequence_name]
        whisker_min_seq = whisker_min[sequence_name]
       
        yticks = [whisker_min_seq, whisker_max_seq]

        axs[splt['id']].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[splt['id']].set_xticklabels([])
        axs[splt['id']].set_ylim(yticks)
        axs[splt['id']].tick_params(axis='y', labelsize=FONT_SIZE) 
        axs[splt['id']].yaxis.set_minor_locator(ticker.MultipleLocator((whisker_max_seq - whisker_min_seq) / 4))
        if not shared_scale:    
            axs[splt['id']].set_yticks(yticks)
            tick_labels = axs[splt['id']].get_yticklabels()
            if whisker_max_seq == max_value:
                tick_labels[1].set_color("#CD3232")  
            if whisker_min_seq == min_value:
                tick_labels[0].set_color("#32CD32")      
            tick_labels[0].set_transform(tick_labels[0].get_transform() + ScaledTranslation(0.9, -0.15, fig.dpi_scale_trans))
            tick_labels[1].set_transform(tick_labels[1].get_transform() + ScaledTranslation(0.9, +0.15, fig.dpi_scale_trans))
            axs[splt['id']].set_yticklabels([set_format(tick) for tick in yticks])

        else:
            if splt['id'] == 0:
                axs[splt['id']].set_yticks(yticks)
                axs[splt['id']].tick_params(axis="y", rotation=90)
                axs[splt['id']].set_yticklabels([set_format(tick) for tick in yticks])
            else:
                axs[splt['id']].set_yticks([])   

        
    plt.tight_layout()
    plot_name = os.path.join(comparison_path, f"{metric_name}_boxplot_paper.pdf")
    if shared_scale:
        plot_name = plot_name.replace(".pdf", "_shared_scale.pdf")
    plt.savefig(plot_name, format='pdf')

    # Adjust plot properties for display
    for sequence_name, splt in splts.items():
        if shared_scale:
            axs[splt['id']].set_title(splt['nickname'], fontsize=FONT_SIZE,  fontweight='bold')
        else:
            axs[splt['id']].set_title(splt['nickname'], fontsize=FONT_SIZE, fontweight='bold', pad=30)

    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize=FONT_SIZE)

    if shared_scale:
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    else:
        fig.set_size_inches(XSIZE, 2*YSIZE)
        plt.tight_layout(rect=[0, 0.10, 1, 0.95])

    plot_name = os.path.join(comparison_path, f"{metric_name}_boxplot.pdf")
    if shared_scale:
        plot_name = plot_name.replace(".pdf", "_shared_scale.pdf")
    plt.savefig(plot_name, format='pdf')
    if shared_scale:
        fig.canvas.manager.set_window_title("Accuracy (shared scale)")
    else:
        fig.canvas.manager.set_window_title("Accuracy")
    plt.show(block=False)

def radar_seq(values, dataset_sequences, exp_names, dataset_nicknames, metric_name, comparison_path, experiments):
    """
     ------------ Description:
    This function creates a radar plot showing the relative performance across different sequences and datasets.
    The performance metric (e.g., accuracy) is normalized by the global median value for each sequence.

    ------------ Parameters:
    values : dict
        values[dataset_name][sequence_name][exp_name] = pandas.DataFrame()
    dataset_sequences : dict
        dataset_sequences[dataset_name] = list{sequence_names}
    exp_names : list
        exp_names = list{exp_names}
    dataset_nicknames : dict
        dataset_nicknames[dataset_name] = list{sequence_nicknames}
    metric_name : string
        metric_name = "accuracy"
    """

    # Create legend handles
    legend_handles = []
    for i_exp, exp_name in enumerate(exp_names):
        baseline = get_baseline(experiments[exp_name].module)
        legend_handles.append(Patch(color=baseline.color, label=exp_names[i_exp]), )

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    all_sequence_names = []
    medians = {}
    median_sequence = {}
    for dataset_name, sequence_names in dataset_sequences.items():
        medians[dataset_name] = {}
        all_sequence_names.extend(dataset_nicknames[dataset_name])
        values_sequence = {}
        for sequence_name in sequence_names:
            medians[dataset_name][sequence_name] = {}
            values_sequence[sequence_name] = pd.Series([])

            for exp_name in exp_names:
                values_dataset_sequence_exp = values[dataset_name][sequence_name][exp_name].copy()
                if values_dataset_sequence_exp.empty:
                    values_dataset_sequence_exp['rmse'] = pd.notna

                medians[dataset_name][sequence_name][exp_name] = np.median(values_dataset_sequence_exp['rmse'])    
                
                if values_sequence[sequence_name].empty:
                    values_sequence[sequence_name] = values_dataset_sequence_exp['rmse']
                else:
                    values_sequence[sequence_name] = pd.concat([values_sequence[sequence_name],
                                                                values_dataset_sequence_exp['rmse']],
                                                               ignore_index=True)

            median_sequence[sequence_name] = np.min(values_sequence[sequence_name])

    num_vars = len(all_sequence_names)
    iExp = 0
    y = {}
    for experiment_name in exp_names:
        baseline = get_baseline(experiments[experiment_name].module)
        y[experiment_name] = []
        for dataset_name, sequence_names in dataset_sequences.items():
            for sequence_name in sequence_names:
                y[experiment_name].append(
                    medians[dataset_name][sequence_name][experiment_name] / median_sequence[sequence_name])

        #for i,yi in enumerate(y[experiment_name]): #INVERT ACCURACY
        #y[experiment_name][i] = 1/yi

        values_ = np.clip(y[experiment_name], 0, 3).tolist() 
        angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()

        values_ += values_[:1]
        angles += angles[:1]

        ax.plot(angles, values_, color=baseline.color, marker='o', linewidth=6)
        ax.plot(np.linspace(0, 2 * np.pi, 100), [2.75] * 100, linestyle="dashed", color="red", linewidth=1)
        ax.plot(np.linspace(0, 2 * np.pi, 100), [1.0] * 100, linestyle="dashed", color="lime", linewidth=1)
        #ax.plot(np.linspace(0, 2 * np.pi, 100), [2.72] * 100, linestyle="dashed", color="green", linewidth=2)
        ax.set_ylim(0, 3)
        plt.xticks(angles[:-1], all_sequence_names)
        #ax.set_xticklabels(all_sequence_names, fontsize=26)


        #current_yticks = ax.get_yticks()
        #new_yticks = current_yticks[:-1]  # Exclude the last tick
        #ax.set_yticks(new_yticks)
        #ax.set_yticklabels([str(tick) for tick in new_yticks], fontsize=12)
        #ax.set_yticklabels(['', '', '', '',  '', ''], fontsize=24)   
        yticks = [0, 1, 2, 3, 4, 5]   # choose whatever is appropriate
        ax.set_yticks(yticks)
        ax.set_yticklabels(['', '', '', '', '', ''], fontsize=24)
        
        ax.tick_params(labelsize=30) 
        ax.set_xticklabels(all_sequence_names, fontsize=30, fontweight="bold")
        iExp = iExp + 1

    
    plt.tight_layout()
    plot_name = os.path.join(comparison_path, f"{metric_name}_radar.pdf")
    plt.savefig(plot_name, format='pdf')
    plt.subplots_adjust(top=0.95, bottom=0.15)  # Adjust the top and bottom to make space for the legend
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles))
    plt.show(block=False)


def plot_cum_error(values, dataset_sequences, exp_names, dataset_nicknames, metric_name, comparison_path, experiments):
    """
     ------------ Description:
    This function generates and saves cumulative error plots for different datasets, sequences, and experiments.
    It creates subplots for each sequence within a dataset and plots the cumulative error for each experiment.
    The cumulative error is calculated as the number of values smaller than or equal to each data point.

    ------------ Parameters:
    values : dict
        values[dataset_name][sequence_name][exp_name] = pandas.DataFrame()
    dataset_sequences : dict
        dataset_sequences[dataset_name] = list{sequence_names}
    exp_names : list
        exp_names = list{exp_names}
    dataset_nicknames : dict
        dataset_nicknames[dataset_name] = list{sequence_nicknames}
    metric_name : string
        metric_name = "accuracy"
    """
    num_sequences = 0
    for dataset_name, sequence_names in dataset_sequences.items():
        num_sequences += len(sequence_names)

    num_cols = 5
    num_rows = math.ceil(num_sequences / num_cols)
    x_size = 12
    y_size = num_rows * 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(x_size, y_size))
    axs = axs.flatten()

    # Create legend handles
    legend_handles = []
    for i_exp, exp_name in enumerate(exp_names):
        legend_handles.append(Patch(color=colors[i_exp], label=exp_names[i_exp]), )

    j_seq = 0
    for dataset_name, sequence_names in dataset_sequences.items():
        for i_seq, sequence_name in enumerate(sequence_names):
            min_x = float('inf')
            max_x = float('-inf')
            for i_exp, experiment_name in enumerate(exp_names):
                baseline = get_baseline(experiments[experiment_name].module)
                data = values[dataset_name][sequence_name][experiment_name]['rmse'].tolist()
                sorted_data = sorted(data)
                cumulated_vector = []
                for data_i in sorted_data:
                    count_smaller = bisect_left(sorted_data, 1.00001*data_i)
                    cumulated_vector.append(count_smaller)
                
                axs[j_seq].plot(sorted_data, cumulated_vector, marker='o', linestyle='-', color=baseline.color)
                min_x = min(min_x, min(sorted_data))
                max_x = max(max_x, max(sorted_data))

            y_max = experiments[exp_name].num_runs
            y_ticks = [0, y_max]

            width_x = 0.1*(max_x - min_x)
            min_x = 0# max(min_x - width_x,0)
            max_x = max_x + width_x
            x_ticks = [min_x, max_x]

            axs[j_seq].set_xticks(x_ticks)
            if j_seq == 0:
                axs[j_seq].set_yticks(y_ticks)
            else:
                axs[j_seq].set_yticklabels([])
            
            # Add minor ticks for the grid (every 10% of the axis range)
            axs[j_seq].xaxis.set_minor_locator(ticker.MultipleLocator(max_x / 4))
            axs[j_seq].yaxis.set_minor_locator(ticker.MultipleLocator(y_max / 4))

            axs[j_seq].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            axs[j_seq].spines['top'].set_visible(False)   # Remove top border
            axs[j_seq].spines['right'].set_visible(False) # Remove right border
            axs[j_seq].tick_params(axis='both', which='minor', labelbottom=False, labelleft=False)
            axs[j_seq].tick_params(axis='y', labelsize=20, rotation=90) 
            axs[j_seq].tick_params(axis='x', labelsize=20, rotation=0)            
            axs[j_seq].set_xlim(x_ticks)
            axs[j_seq].set_ylim(y_ticks)

            axs[j_seq].tick_params(axis='x', pad=10) 
            axs[j_seq].set_xticklabels([f"{x_ticks[0]:.2f}", f"{x_ticks[1]:.2f}"], ha='right')  

            def set_format(tick):
                if tick == 0:
                    return f"0"
                return f"{tick:.1e}"
            
            axs[j_seq].set_xticklabels([set_format(tick) for tick in x_ticks])
            j_seq = j_seq + 1

    plot_name = os.path.join(comparison_path, f"{metric_name}_cummulated_error.pdf")
    plt.tight_layout()
    plt.savefig(plot_name, format='pdf')

    j_seq = 0
    for dataset_name, sequence_names in dataset_sequences.items():
        for i_seq, sequence_name in enumerate(sequence_names):
            axs[j_seq].set_title(dataset_nicknames[dataset_name][i_seq])

    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles))
    plt.subplots_adjust(top=0.9, bottom=0.25)  # Adjust the top and bottom to make space for the legend
    plt.show(block=False)

def create_and_show_canvas(dataset_sequences, VSLAMLAB_BENCHMARK, comparison_path, padding=10):
    image_paths = []

    for dataset_name, sequence_names in dataset_sequences.items():
        for sequence_name in sequence_names:
            image_files = list_image_files_in_folder(
                os.path.join(VSLAMLAB_BENCHMARK, dataset_name.upper(), sequence_name, 'rgb_0'))
            image_paths.append(
                os.path.join(VSLAMLAB_BENCHMARK, dataset_name.upper(), sequence_name, 'rgb_0', image_files[0]))

    m = 5  # Number of columns
    n = math.ceil(len(image_paths) / m)  # Number of rows

    img_width = 640
    img_height = 480

    # Calculate canvas size including padding
    canvas_width = m * (img_width + padding) - padding
    canvas_height = n * (img_height + padding) - padding

    # Load and resize images
    images = [Image.open(path).resize((img_width, img_height), Image.LANCZOS) for path in image_paths]

    # Create a blank canvas with a white background
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

    # Paste each image into the correct position with padding
    for i in range(n):
        for j in range(m):
            index = i * m + j
            if index < len(images):
                x_offset = j * (img_width + padding)
                y_offset = i * (img_height + padding)
                canvas.paste(images[index], (x_offset, y_offset))

    # Save the canvas
    plot_name = os.path.join(comparison_path, 'canvas_sequences.png')
    canvas.save(plot_name)

    # Show the canvas
    plt.figure(figsize=(12.8, 6.4))  # Convert pixels to inches for display
    plt.imshow(canvas)
    plt.axis('off')  # Hide the axis
    plt.show(block=False)

def num_tracked_frames(values, dataset_sequences, figures_path, experiments, shared_scale=False):
    # Get number of sequences
    num_sequences = 0
    splts = {}
    for dataset_name, sequence_names in dataset_sequences.items():
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        for sequence_name in sequence_names:
            splts[sequence_name] = {}
            splts[sequence_name]['id'] = num_sequences
            splts[sequence_name]['dataset_name'] = dataset_name
            splts[sequence_name]['nickname']= dataset.get_sequence_nickname(sequence_name)
            num_sequences += 1

    exp_names = list(experiments.keys())

    # Figure dimensions
    NUM_COLS = 5
    NUM_ROWS = math.ceil(num_sequences / NUM_COLS)
    XSIZE, YSIZE = 12, 2 * NUM_ROWS + 0.5
    WIDTH_PER_SERIES = min(XSIZE / len(exp_names), 1.0)/3
    FONT_SIZE = 15
    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(XSIZE, YSIZE))
    axs = axs.flatten()

    # Create legend handles
    legend_handles = []
    colors = {}
    for i_exp, exp_name in enumerate(exp_names):
        baseline = get_baseline(experiments[exp_name].module)   
        colors[exp_name] = baseline.color
        legend_handles.append(Patch(color=colors[exp_name], label=exp_names[i_exp]))

    # Plot boxplots        
    max_rgb = {}      
    for sequence_name, splt in splts.items():
        max_rgb[sequence_name] = 0
        for i_exp, exp_name in enumerate(exp_names):
            values_seq_exp = values[splt['dataset_name']][sequence_name][exp_name]
            if not values_seq_exp.empty:
                num_frames = values[splt['dataset_name']][sequence_name][exp_name]['num_frames']
                max_rgb[sequence_name] = max(max(num_frames), max_rgb[sequence_name])

    for sequence_name, splt in splts.items():
        for i_exp, exp_name in enumerate(exp_names):
            values_seq_exp = values[splt['dataset_name']][sequence_name][exp_name]    
            if values_seq_exp.empty:
                continue

            num_frames = values_seq_exp['num_frames'] 
            num_tracked_frames = values_seq_exp['num_tracked_frames'] 
            num_evaluated_frames = values_seq_exp['num_evaluated_frames']   
         
            if shared_scale:
                num_frames /= max_rgb[sequence_name]
                num_tracked_frames /= max_rgb[sequence_name]
                num_evaluated_frames /= max_rgb[sequence_name]

            median_num_frames = np.median(num_frames)
            median_num_tracked_frames = np.median(num_tracked_frames)
            median_num_evaluated_frames = np.median(num_evaluated_frames)
           
            positions = np.array([3 * i_exp, 3 * i_exp + 1, 3 * i_exp + 2]) * WIDTH_PER_SERIES
            axs[splt['id']].bar(
            positions, 
            [median_num_frames, median_num_tracked_frames, median_num_evaluated_frames], 
            color=colors[exp_name], alpha=0.3, width=WIDTH_PER_SERIES*0.9)
            
            metrics = [num_frames, num_tracked_frames, num_evaluated_frames]
            boxprops = medianprops = whiskerprops = capprops = dict(color=colors[exp_name])
            flierprops = dict(marker='o', color=colors[exp_name], alpha=1.0)    
            for i, metric in enumerate(metrics):
                positions = [(3 * i_exp + i) * WIDTH_PER_SERIES]
                boxplot_accuracy = axs[splt['id']].boxplot(
                    metrics[i],
                    positions=positions, widths=WIDTH_PER_SERIES,
                    patch_artist=False,
                    boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops, flierprops=flierprops)

        if shared_scale:
            yticks = [0, 1]
        else:
            yticks = [0, max_rgb[sequence_name]]
        axs[splt['id']].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[splt['id']].set_xticklabels([])
        axs[splt['id']].set_ylim(yticks)
        axs[splt['id']].tick_params(axis='y', labelsize=FONT_SIZE) 
        axs[splt['id']].yaxis.set_minor_locator(ticker.MultipleLocator(max_rgb[sequence_name] / 4))
        axs[splt['id']].set_yticks(yticks)
        if not shared_scale:    
            axs[splt['id']].set_yticks(yticks)
            tick_labels = axs[splt['id']].get_yticklabels() 
            tick_labels[0].set_transform(tick_labels[0].get_transform() + ScaledTranslation(0.2, -0.15, fig.dpi_scale_trans))
            tick_labels[1].set_transform(tick_labels[1].get_transform() + ScaledTranslation(0.5, +0.15, fig.dpi_scale_trans))
        else:
            if splt['id'] == 0:
                axs[splt['id']].set_yticks(yticks)
            else:
                axs[splt['id']].set_yticks([])   

    plt.tight_layout()
    plot_name = os.path.join(figures_path, f"num_frames_boxplot_paper.pdf")
    if shared_scale:
        plot_name = plot_name.replace(".pdf", "_shared_scale.pdf")
    plt.savefig(plot_name, format='pdf')

    # Adjust plot properties for display
    for sequence_name, splt in splts.items():
        if shared_scale:
            axs[splt['id']].set_title(splt['nickname'], fontsize=FONT_SIZE,  fontweight='bold')
        else:
            axs[splt['id']].set_title(splt['nickname'], fontsize=FONT_SIZE, fontweight='bold', pad=30)

    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize=FONT_SIZE)
    
    if shared_scale:
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    else:
        fig.set_size_inches(XSIZE, 2*YSIZE)
        plt.tight_layout(rect=[0, 0.10, 1, 0.95])

    plot_name = os.path.join(figures_path, f"num_frames_boxplot.pdf")
    if shared_scale:
        plot_name = plot_name.replace(".pdf", "_shared_scale.pdf")
    plt.savefig(plot_name, format='pdf')

    fig.canvas.manager.set_window_title("Number of Frames")
    plt.show(block=False)

import pandas as pd
import matplotlib.pyplot as plt

def plot_table(ax, experiments, label, norm_label, sequence_nicknames, title = '', unit_factor = 1, figures_path = ''):
    colors = {}
    for exp_name, exp in experiments.items():
        baseline = get_baseline(experiments[exp_name].module)   
        colors[experiments[exp_name].module] = baseline.color     
        
    colors['Sequence'] = 'black'

    all_logs = []
    for exp_name, exp in experiments.items():
        exp_log = pd.read_csv(exp.log_csv)
        exp_log = exp_log[
        (exp_log['STATUS'] == 'completed') &
        (exp_log['SUCCESS'] == True) &
        (exp_log['EVALUATION'] != 'none')]
        
        if norm_label is None:
            exp_log['__norm__'] = 1.0
            exp_log['label_per_norm_label'] = unit_factor * exp_log[label] / exp_log['__norm__']
        else:
            exp_log['label_per_norm_label'] = unit_factor * exp_log[label] / exp_log[norm_label]

        all_logs.append(exp_log)
    
    df = pd.concat(all_logs, ignore_index=True)

    # Per-sequence mean ± std
    summary = df.groupby(['method_name', 'sequence_name'])['label_per_norm_label'].agg(['mean', 'std']).reset_index()
    if norm_label == None:
        summary['LABEL'] = summary.apply(lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}" 
                                            if not pd.isna(row['std']) 
                                            else f"{row['mean']:.2f} ± 0.00", axis=1)
    else:
        summary['LABEL'] = summary.apply(lambda row: f"{row['mean']:.2f}", axis=1)
        
    summary['sequence_name'] = summary['sequence_name'].map(sequence_nicknames).fillna(summary['sequence_name'])
    pivot = summary.pivot(index='sequence_name', columns='method_name', values='LABEL').fillna('-')
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={'sequence_name': 'Sequence'})

    # Overall mean ± std per method
    overall = df.groupby('method_name')['label_per_norm_label'].agg(['mean', 'std']).reset_index()
    overall['LABEL'] = overall.apply(lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}"
                                     if not pd.isna(row['std']) 
                                    else f"{row['mean']:.2f} ± 0.00", axis=1)
    
    if norm_label != None:
        overall_row = {'Sequence': 'Overall'}
        overall_row.update(dict(zip(overall['method_name'], overall['LABEL'])))
        pivot = pd.concat([pivot, pd.DataFrame([overall_row])], ignore_index=True)
    
    # Plot the visual table
    ax.axis('off')
    table = ax.table(cellText=pivot.values, colLabels=pivot.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Align first column (sequence_name) to the left
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.get_text().set_ha('left')

    # Format borders
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(1)
        # Show top/bottom borders (horizontal lines)
        if row == 0:
            cell.visible_edges = 'B'  # top row: top, bottom, left
        elif row == len(pivot):
            cell.visible_edges = 'T'  # last row: bottom, top, left
        else:
            cell.visible_edges = ''  # inner rows: top & bottom, left

        # Only first column keeps left border
        if col == 1:
            if 'L' not in cell.visible_edges:
                cell.visible_edges += 'L'
        else:
            cell.visible_edges = cell.visible_edges.replace('R', '').replace('L', '')

    # Define header colors for each column (match number of columns)
    header_colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']  # Extend as needed
    color_map = {'droidslam': '#1f77b4', 'orbslam2': '#d62728'}

    # Apply colors to header row

    for col, cell in table.get_celld().items():
        row_idx, col_idx = col
        if row_idx == 0 or row_idx == len(pivot):
            # Cycle colors if there are more columns than colors defined
            header_label = table._cells[(0, col_idx)].get_text().get_text()
            method = header_label.split('\n')[0] if '\n' in header_label else header_label
            #cell.set_facecolor(color)
            cell.set_text_props(color=colors[method], weight='bold')  # white bold text for contrast

    if title:
        ax.set_title(title, pad=10)
    
    latex_path = os.path.join(figures_path, f"{label}_{norm_label}_label_table.tex")
    latex_code = pivot.to_latex(index=False, escape=False, column_format='l' + 'c' * (pivot.shape[1] - 1))

    latex_code = latex_code.replace('±', r'$\pm$') 
    latex_code = latex_code.replace('_', r'\_') 
    latex_code = latex_code.replace(r'\bottomrule', '') 
    latex_code = latex_code.replace(r'\toprule', '') 
    latex_code = latex_code.replace('Overall', r'\textbf{Overall}') 
    latex_code = latex_code.replace('Sequence', '') 

    for exp_name, exp in experiments.items():
        baseline = get_baseline(experiments[exp_name].module)   
        baseline_name = experiments[exp_name].module
        color_hex = to_hex(baseline.color)  
        #latex_col = rf'\textcolor[HTML]{{{color_hex[1:].upper()}}}{rf'\\textbf{'{baseline_name}'}'}'
        latex_col = rf'\textbf{{\textcolor[HTML]{{{color_hex[1:].upper()}}}{{{baseline_name}}}}}'

        latex_code = latex_code.replace(baseline_name, latex_col)
        # colors[experiments[exp_name].module] = baseline.color     
        # color_hex = to_hex(colors[col])  
        # latex_col = rf'\textcolor[HTML]{{{color_hex[1:].upper()}}}{{{col}}}'
        # colored_columns_latex[experiments[exp_name].module] = latex_col

    lines = latex_code.splitlines()
    for i, line in enumerate(lines):
        if 'Overall' in line:
            lines.insert(i, r'\bottomrule')
            break
    latex_code = '\n'.join(lines)

    with open(latex_path, 'w') as f:
        f.write(latex_code)

def get_baseline_colors(experiments):
    colors = {}
    for exp_name, _ in experiments.items():
        baseline = get_baseline(experiments[exp_name].module)   
        colors[baseline.name_label] = baseline.color     
    colors['Sequence'] = 'black'
    return colors

def get_baseline_labels(experiments):
    baseline_labels = {}
    for exp_name, _ in experiments.items():
        baseline = get_baseline(experiments[exp_name].module)   
        baseline_labels[baseline.baseline_name] = baseline.name_label     
    return baseline_labels    

def combine_exp_log(experiments, label, norm_label, unit_factor):
    all_logs = []
    for exp_name, exp in experiments.items():
        exp_log = pd.read_csv(exp.log_csv)
        exp_log = exp_log[
        (exp_log['STATUS'] == 'completed') &
        (exp_log['SUCCESS'] == True) &
        (exp_log['EVALUATION'] != 'none')]
        
        if norm_label is None:
            exp_log['__norm__'] = 1.0
            exp_log['label_per_norm_label'] = unit_factor * exp_log[label] / exp_log['__norm__']
        else:
            exp_log['label_per_norm_label'] = unit_factor * exp_log[label] / exp_log[norm_label]

        all_logs.append(exp_log)
    
    df = pd.concat(all_logs, ignore_index=True)
    return df

def apply_colors(rows_to_color, table, colors):
    for col, cell in table.get_celld().items():
        row_idx, col_idx = col
        if row_idx in rows_to_color:
            header_label = table._cells[(0, col_idx)].get_text().get_text()
            method = header_label.split('\n')[0] if '\n' in header_label else header_label
            cell.set_text_props(color=colors[method], weight='bold')  # white bold text for contrast
    return table

def plot_table_memory_per_frame(ax, experiments, sequence_nicknames, title = '', unit_factor = 1, figures_path = ''):
    ax.axis('off')
    baseline_colors = get_baseline_colors(experiments)
    baseline_labels = get_baseline_labels(experiments)

    dfs = []
    for exp_name, exp in experiments.items():
        exp_log = pd.read_csv(exp.log_csv)
        exp_log = exp_log[
        (exp_log['STATUS'] == 'completed') &
        (exp_log['SUCCESS'] == True) &
        (exp_log['EVALUATION'] != 'none')]
        dfs.append(exp_log)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['GPU'] *= unit_factor/df_all['num_frames'] 
    df_all['SWAP'] *= unit_factor/df_all['num_frames'] 
    df_all['RAM'] *= unit_factor/df_all['num_frames'] 

    metrics = ['GPU', 'RAM', 'SWAP']
    # Compute both mean and std
    grouped_mean = df_all.groupby(['sequence_name', 'method_name'])[metrics].mean()
    grouped_std = df_all.groupby(['sequence_name', 'method_name'])[metrics].std()

    # Combine them into a formatted string: "mean ± std"
    def format_mean_std(mean, std):
        mean_rounded = mean.round(2)
        return mean_rounded.astype(str)

    grouped = grouped_mean.combine(grouped_std, format_mean_std)

    table = grouped.unstack('method_name')

    table.columns = pd.MultiIndex.from_tuples(
        [(method, metric) for metric, method in table.columns],
        names=['Baseline', 'Metric']
    )
    table = table.sort_index(axis=1, level=0)
    
    # Add 'Sequence' column with value 'Average'
    top_row = [baseline_labels.get(col[0], col[0]) if col[1] == 'RAM' else '' for col in table.columns]
    bottom_row = [col[1] for col in table.columns]
    # Compute average (mean) across sequences for each method and metric
 
    full_table = pd.DataFrame([top_row, bottom_row], columns=table.columns)
    data_rows = pd.DataFrame(table.values, columns=table.columns, index=table.index)
    
    full_matrix = pd.concat([full_table, data_rows])

    ax.axis('tight')
    cell_text = full_matrix.values.tolist()
    mapped_index = [sequence_nicknames.get(seq, seq) for seq in data_rows.index]
    row_labels = [''] * 2 + mapped_index

    table_plot = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        loc='center',
        cellLoc='center'
    )
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.3, 1.3)

    # Format borders
    for (row, col), cell in table_plot.get_celld().items():
        cell.set_linewidth(1)
        cell.visible_edges = '' 
        if row == 0 or row == 1:
            cell.visible_edges = 'B' 

    # Apply colors only to the top header row (row index 0)
    for col_idx, col in enumerate(table.columns):
        method_name = col[0]
        metric = col[1]
        if metric == 'RAM' and baseline_labels[method_name] in baseline_colors:
            cell = table_plot[0, col_idx]
            cell.set_text_props(weight='bold', color=baseline_colors[baseline_labels[method_name]])  # Optional: bold label

    ax.set_title(title, pad=10)
    #ax.tight_layout()
   
    import os

    # Prepare data for LaTeX export
    latex_table = data_rows.copy()
    latex_table.index = mapped_index  # Replace index with sequence nicknames

    # Get the ordered list of baselines and metrics
    baseline_headers = [col[0] for col in latex_table.columns]
    metric_headers = [col[1] for col in latex_table.columns]

    # Group baseline headers and count occurrences
    from collections import OrderedDict

    baseline_counts = OrderedDict()
    for b in baseline_headers:
        baseline_counts[b] = baseline_counts.get(b, 0) + 1

    # STEP 1: Build the LaTeX header with two rows
    header_row_1 = [""]
    header_row_2 = [""]
    for baseline_name, count in baseline_counts.items():
        label = baseline_labels.get(baseline_name, baseline_name)
        color = baseline_colors.get(label, '#FFFFFF').lstrip('#')
        color_hex = to_hex(color)  
        header_row_1.append(
            rf"\multicolumn{{{3}}}{{c}}{{\textbf{{\textcolor[HTML]{{{color_hex[1:].upper()}}}{{{baseline_labels[baseline_name]}}}}}}}"
        )
        header_row_2.extend(["GPU", "RAM", "SWAP"])  # assuming fixed order

    # STEP 2: Format the data rows (with LaTeX-safe escaping)
    body_lines = []
    for idx, row in latex_table.iterrows():
        row_line = [f"\\texttt{{{idx}}}"] + list(row.values)
        body_lines.append(" & ".join(row_line) + " \\\\")

    # STEP 3: Write the complete LaTeX table
    ncols = len(header_row_2)
    col_format = 'l' + 'c' * ncols
    lines = [
        f"\\begin{{tabular}}{{{col_format}}}",
        " & ".join(header_row_1) + " \\\\", 
        " & ".join(header_row_2) + " \\\\ \\midrule"
    ] + body_lines + [
        "\\bottomrule",
        "\\end{tabular}",
    ]

    # Save to file
    latex_path = os.path.join(figures_path, "memory_usage_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))

def plot_table_memory_total(ax, experiments, sequence_nicknames, title = '', unit_factor = 1, figures_path = ''):
    ax.axis('off')
    baseline_colors = get_baseline_colors(experiments)
    baseline_labels = get_baseline_labels(experiments)

    dfs = []
    for exp_name, exp in experiments.items():
        exp_log = pd.read_csv(exp.log_csv)
        exp_log = exp_log[
        (exp_log['STATUS'] == 'completed') &
        (exp_log['SUCCESS'] == True) &
        (exp_log['EVALUATION'] != 'none')]
        dfs.append(exp_log)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['GPU'] /= unit_factor
    df_all['SWAP'] /= unit_factor
    df_all['RAM'] /= unit_factor

    metrics = ['GPU', 'RAM', 'SWAP']
    # Compute both mean and std
    grouped_mean = df_all.groupby(['sequence_name', 'method_name'])[metrics].mean()
    grouped_std = df_all.groupby(['sequence_name', 'method_name'])[metrics].std()

    # Combine them into a formatted string: "mean ± std"
    def format_mean_std(mean, std):
        mean_rounded = mean.round(2)
        std_rounded = std.round(2)
        # Replace NaN std with 0.00 and format
        std_filled = std_rounded.fillna(0.0)
        return mean_rounded.astype(str) + ' ± ' + std_filled.astype(str)

    grouped = grouped_mean.combine(grouped_std, format_mean_std)

    table = grouped.unstack('method_name')

    table.columns = pd.MultiIndex.from_tuples(
        [(method, metric) for metric, method in table.columns],
        names=['Baseline', 'Metric']
    )
    table = table.sort_index(axis=1, level=0)
    top_row = [baseline_labels.get(col[0], col[0]) if col[1] == 'RAM' else '' for col in table.columns]
    bottom_row = [col[1] for col in table.columns]
    full_table = pd.DataFrame([top_row, bottom_row], columns=table.columns)
    data_rows = pd.DataFrame(table.values, columns=table.columns, index=table.index)
    full_matrix = pd.concat([full_table, data_rows])

    ax.axis('tight')
    cell_text = full_matrix.values.tolist()
    mapped_index = [sequence_nicknames.get(seq, seq) for seq in data_rows.index]
    row_labels = [''] * 2 + mapped_index

    table_plot = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        loc='center',
        cellLoc='center'
    )
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.3, 1.3)

    # Format borders
    for (row, col), cell in table_plot.get_celld().items():
        cell.set_linewidth(1)
        cell.visible_edges = '' 
        if row == 0 or row == 1:
            cell.visible_edges = 'B' 

    # Apply colors only to the top header row (row index 0)
    for col_idx, col in enumerate(table.columns):
        method_name = col[0]
        metric = col[1]
        if metric == 'RAM' and baseline_labels[method_name] in baseline_colors:
            cell = table_plot[0, col_idx]
            cell.set_text_props(weight='bold', color=baseline_colors[baseline_labels[method_name]])  # Optional: bold label

    ax.set_title(title, pad=10)

    import os

    # Prepare data for LaTeX export
    latex_table = data_rows.copy()
    latex_table.index = mapped_index  # Replace index with sequence nicknames

    # Get the ordered list of baselines and metrics
    baseline_headers = [col[0] for col in latex_table.columns]
    metric_headers = [col[1] for col in latex_table.columns]

    # Group baseline headers and count occurrences
    from collections import OrderedDict

    baseline_counts = OrderedDict()
    for b in baseline_headers:
        baseline_counts[b] = baseline_counts.get(b, 0) + 1

    # STEP 1: Build the LaTeX header with two rows
    header_row_1 = [""]
    header_row_2 = [""]
    for baseline_name, count in baseline_counts.items():
        label = baseline_labels.get(baseline_name, baseline_name)
        color = baseline_colors.get(label, '#FFFFFF').lstrip('#')
        color_hex = to_hex(color)  
        header_row_1.append(
            rf"\multicolumn{{{3}}}{{c}}{{\textbf{{\textcolor[HTML]{{{color_hex[1:].upper()}}}{{{baseline_labels[baseline_name]}}}}}}}"
        )
        header_row_2.extend(["GPU", "RAM", "SWAP"])  # assuming fixed order

    # STEP 2: Format the data rows (with LaTeX-safe escaping)
    body_lines = []
    for idx, row in latex_table.iterrows():
        row_line = [f"\\texttt{{{idx}}}"] + list(row.values)
        body_lines.append(" & ".join(row_line) + " \\\\")

    # STEP 3: Write the complete LaTeX table
    ncols = len(header_row_2)
    col_format = 'l' + 'c' * ncols
    lines = [
        f"\\begin{{tabular}}{{{col_format}}}",
        " & ".join(header_row_1) + " \\\\", 
        " & ".join(header_row_2) + " \\\\ \\midrule"
    ] + body_lines + [
        "\\bottomrule",
        "\\end{tabular}",
    ]

    # Save to file
    latex_path = os.path.join(figures_path, "memory_usage_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))

def plot_table_time_total(ax, experiments, label, sequence_nicknames, title = '', unit_factor = 1, figures_path = ''):
    baseline_colors = get_baseline_colors(experiments)
    baseline_labels = get_baseline_labels(experiments)
    df = combine_exp_log(experiments, label, None, unit_factor)
        
    # Per-sequence mean ± std
    summary = df.groupby(['method_name', 'sequence_name'])['label_per_norm_label'].agg(['mean', 'std']).reset_index()
    summary['LABEL'] = summary.apply(lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}"
                                    if not pd.isna(row['std']) 
                                    else f"{row['mean']:.2f} ± 0.00", axis=1)

    summary['sequence_name'] = summary['sequence_name'].map(sequence_nicknames).fillna(summary['sequence_name'])
    pivot = summary.pivot(index='sequence_name', columns='method_name', values='LABEL').fillna('-')
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={'sequence_name': 'Sequence'})

    pivot = pivot.rename(columns=baseline_labels)
    # Plot the visual table
    ax.axis('off')
    table = ax.table(cellText=pivot.values, colLabels=pivot.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Align first column (sequence_name) to the left
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.get_text().set_ha('left')

    # Format borders
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(1)
        # Show top/bottom borders (horizontal lines)
        if row == 0:
            cell.visible_edges = 'B'  # top row: top, bottom, left
        else:
            cell.visible_edges = ''  # inner rows: top & bottom, left

        # Only first column keeps left border
        if col == 1:
            if 'L' not in cell.visible_edges:
                cell.visible_edges += 'L'
        else:
            cell.visible_edges = cell.visible_edges.replace('R', '').replace('L', '')

    table =  apply_colors([0], table, baseline_colors)

    if title:
        ax.set_title(title, pad=10)
    
    # Save latex code
    latex_path = os.path.join(figures_path, f"{label}_total_table.tex")
    latex_code = pivot.to_latex(index=False, escape=False, column_format='l' + 'c' * (pivot.shape[1] - 1))

    latex_code = latex_code.replace('±', r'$\pm$') 
    latex_code = latex_code.replace('_', r'\_') 
    latex_code = latex_code.replace(r'\bottomrule', '') 
    latex_code = latex_code.replace(r'\toprule', '') 
    latex_code = latex_code.replace('Overall', r'\textbf{Overall}') 
    latex_code = latex_code.replace('Sequence', '') 

    for exp_name, _ in experiments.items():
        baseline = get_baseline(experiments[exp_name].module)   
        baseline_name = experiments[exp_name].module
        color_hex = to_hex(baseline.color)  
        latex_col = rf'\textbf{{\textcolor[HTML]{{{color_hex[1:].upper()}}}{{{baseline_labels[baseline_name]}}}}}'
        latex_code = latex_code.replace(baseline_labels[baseline_name], latex_col)


    lines = latex_code.splitlines()
    for i, line in enumerate(lines):
        if 'Overall' in line:
            lines.insert(i, r'\bottomrule')
            break
    latex_code = '\n'.join(lines)

    with open(latex_path, 'w') as f:
        f.write(latex_code)

def plot_table_time_per_frame(ax, experiments, label, norm_label, sequence_nicknames, title = '', unit_factor = 1, figures_path = ''):
    baseline_colors = get_baseline_colors(experiments)
    baseline_labels = get_baseline_labels(experiments)

    df = combine_exp_log(experiments, label, norm_label, unit_factor)

    # Per-sequence mean
    summary = df.groupby(['method_name', 'sequence_name'])['label_per_norm_label'].agg(['mean']).reset_index()
    summary['LABEL'] = summary.apply(lambda row: f"{row['mean']:.2f}", axis=1)
        
    summary['sequence_name'] = summary['sequence_name'].map(sequence_nicknames).fillna(summary['sequence_name'])
    pivot = summary.pivot(index='sequence_name', columns='method_name', values='LABEL').fillna('-')
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={'sequence_name': 'Sequence'})

    # Overall mean ± std per method
    overall = df.groupby('method_name')['label_per_norm_label'].agg(['mean', 'std']).reset_index()
    overall['LABEL'] = overall.apply(lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}"
                                     if not pd.isna(row['std']) 
                                    else f"{row['mean']:.2f} ± 0.00", axis=1)
    overall_row = {'Sequence': 'Overall'}
    overall_row.update(dict(zip(overall['method_name'], overall['LABEL'])))
    pivot = pd.concat([pivot, pd.DataFrame([overall_row])], ignore_index=True)

    pivot = pivot.rename(columns=baseline_labels)
    
    # Plot the visual table
    ax.axis('off')
    table = ax.table(cellText=pivot.values, colLabels=pivot.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    # Align first column (sequence_name) to the left
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.get_text().set_ha('left')

    # Format borders
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(1)
        # Show top/bottom borders (horizontal lines)
        if row == 0:
            cell.visible_edges = 'B'  # top row: top, bottom, left
        elif row == len(pivot):
            cell.visible_edges = 'T'  # last row: bottom, top, left
        else:
            cell.visible_edges = ''  # inner rows: top & bottom, left

        # Only first column keeps left border
        if col == 1:
            if 'L' not in cell.visible_edges:
                cell.visible_edges += 'L'
        else:
            cell.visible_edges = cell.visible_edges.replace('R', '').replace('L', '')

    table =  apply_colors([0, len(pivot)], table, baseline_colors)
    if title:
        ax.set_title(title, pad=10)
    
    # Save latex code
    latex_path = os.path.join(figures_path, f"{label}_{norm_label}_table.tex")
    latex_code = pivot.to_latex(index=False, escape=False, column_format='l' + 'c' * (pivot.shape[1] - 1))

    latex_code = latex_code.replace('±', r'$\pm$') 
    latex_code = latex_code.replace('_', r'\_') 
    latex_code = latex_code.replace(r'\bottomrule', '') 
    latex_code = latex_code.replace(r'\toprule', '') 
    latex_code = latex_code.replace('Overall', r'\textbf{Overall}') 
    latex_code = latex_code.replace('Sequence', '') 

    for exp_name, _ in experiments.items():
        baseline = get_baseline(experiments[exp_name].module)   
        baseline_name = experiments[exp_name].module
        color_hex = to_hex(baseline.color)  
        latex_col = rf'\textbf{{\textcolor[HTML]{{{color_hex[1:].upper()}}}{{{baseline_labels[baseline_name]}}}}}'
        latex_code = latex_code.replace(baseline_labels[baseline_name], latex_col)


    lines = latex_code.splitlines()
    for i, line in enumerate(lines):
        if 'Overall' in line:
            lines.insert(i, r'\bottomrule')
            break
    latex_code = '\n'.join(lines)

    with open(latex_path, 'w') as f:
        f.write(latex_code)

def running_time(figures_path, experiments, sequence_nicknames):
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    plot_table_time_per_frame(axs[0], experiments, 'TIME', 'num_frames', title='Processing Time (ms / frame)', unit_factor=1e3, 
               figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    plot_table_time_total(axs[1], experiments, 'TIME', title='Total Processing Time (s)', 
               figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    plt.tight_layout()
    plt.show()
    #table = axs[0].table(cellText=pivot.values, colLabels=pivot.columns, loc='center', cellLoc='center')
    ...
    
    #plot_table(experiments, 'TIME','num_frames')

def plot_memory(figures_path, experiments, sequence_nicknames):
    fig, axs = plt.subplots(2, 1, figsize=(3 + 2*len(experiments), 6))
    #axs = axs.flatten()

    plot_table_memory_per_frame(axs[0], experiments, title='GPU Memory (MB / frame)', unit_factor=1e3, 
               figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    plot_table_memory_total(axs[1], experiments, title='GPU Memory (GB)', unit_factor=1e0, 
               figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    # plot_table(axs[1], experiments, 'RAM', 'num_frames', title='RAM Memory (MB / frame)', unit_factor=1e3, 
    #            figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    # plot_table(axs[2], experiments, 'SWAP', 'num_frames', title='SWAP Memory (MB / frame)', unit_factor=1e3, 
    #            figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    # plot_table(axs[3], experiments, 'GPU', None, title='Total GPU Memory (GB)', 
    #            figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    # plot_table(axs[4], experiments, 'RAM', None, title='Total RAM Memory (GB)', 
    #            figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    # plot_table(axs[5], experiments, 'SWAP', None, title='Total SWAP Memory (GB)', 
    #            figures_path=figures_path, sequence_nicknames= sequence_nicknames)
    
    plt.tight_layout()
    plt.show()
    #table = axs[0].table(cellText=pivot.values, colLabels=pivot.columns, loc='center', cellLoc='center')
    ...
    
    #plot_table(experiments, 'TIME','num_frames')




