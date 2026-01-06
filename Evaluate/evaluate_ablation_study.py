#!/usr/bin/env python3
"""
Ablation Study Evaluation Script

Compares SLAM performance with and without kinematic constraints
on the HAMLYN endoscopy dataset.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse


def load_constraint_statistics(result_dir: str) -> Dict:
    """Load constraint statistics from result directory."""
    stats_file = os.path.join(result_dir, "constraint_statistics.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return {}


def load_trajectory_metrics(result_dir: str) -> Dict:
    """Load trajectory evaluation metrics."""
    # Look for metrics files (ATE, RPE, etc.)
    metrics = {}

    # Check for common metric files
    metric_files = {
        'ate': 'ate_metrics.txt',
        'rpe': 'rpe_metrics.txt',
        'metrics': 'metrics.json'
    }

    for metric_name, filename in metric_files.items():
        filepath = os.path.join(result_dir, filename)
        if os.path.exists(filepath):
            if filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    metrics.update(json.load(f))
            else:
                # Try to parse text file
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # Simple parsing - could be more sophisticated
                        if 'RMSE' in content or 'rmse' in content.lower():
                            # Extract RMSE value
                            import re
                            match = re.search(r'(\d+\.\d+)', content)
                            if match:
                                metrics[f'{metric_name}_rmse'] = float(match.group(1))
                except:
                    pass

    return metrics


def collect_experiment_results(experiment_base_dir: str,
                               experiment_configs: List[str]) -> pd.DataFrame:
    """
    Collect results from multiple experiment configurations.

    Args:
        experiment_base_dir: Base directory containing experiment results
        experiment_configs: List of experiment configuration names

    Returns:
        DataFrame with all results
    """
    results = []

    for config in experiment_configs:
        config_dir = os.path.join(experiment_base_dir, config)

        if not os.path.exists(config_dir):
            print(f"Warning: Config directory not found: {config_dir}")
            continue

        # Walk through dataset/sequence subdirectories
        for dataset in os.listdir(config_dir):
            dataset_dir = os.path.join(config_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue

            for sequence in os.listdir(dataset_dir):
                sequence_dir = os.path.join(dataset_dir, sequence)
                if not os.path.isdir(sequence_dir):
                    continue

                # Load metrics
                constraint_stats = load_constraint_statistics(sequence_dir)
                trajectory_metrics = load_trajectory_metrics(sequence_dir)

                # Combine results
                result = {
                    'experiment': config,
                    'dataset': dataset,
                    'sequence': sequence,
                    **constraint_stats,
                    **trajectory_metrics
                }

                results.append(result)

    return pd.DataFrame(results)


def analyze_ablation_results(df: pd.DataFrame) -> Dict:
    """
    Analyze ablation study results.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with analysis results
    """
    analysis = {}

    # Group by experiment type
    if 'experiment' in df.columns:
        grouped = df.groupby('experiment')

        # Compute mean and std for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = grouped[numeric_cols].agg(['mean', 'std', 'min', 'max'])
        analysis['summary_statistics'] = summary

        # Compute improvements from baseline
        if 'baseline' in df['experiment'].values:
            baseline_results = df[df['experiment'].str.contains('baseline', case=False)]

            for exp in df['experiment'].unique():
                if 'baseline' in exp.lower():
                    continue

                exp_results = df[df['experiment'] == exp]

                # Compare key metrics if available
                for metric in ['mean_translation_correction', 'mean_rotation_correction',
                              'trajectory_smoothness_score']:
                    if metric in df.columns:
                        baseline_val = baseline_results[metric].mean()
                        exp_val = exp_results[metric].mean()

                        if baseline_val != 0:
                            improvement = ((exp_val - baseline_val) / baseline_val) * 100
                            analysis[f'{exp}_{metric}_improvement'] = improvement

    return analysis


def plot_ablation_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create visualization plots for ablation study.

    Args:
        df: DataFrame with results
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Constraint corrections by configuration
    if 'mean_translation_correction' in df.columns and 'experiment' in df.columns:
        plt.figure(figsize=(12, 6))

        experiments = df['experiment'].unique()
        translation_means = []
        rotation_means = []

        for exp in experiments:
            exp_data = df[df['experiment'] == exp]
            translation_means.append(exp_data['mean_translation_correction'].mean()
                                   if 'mean_translation_correction' in exp_data else 0)
            rotation_means.append(exp_data.get('mean_rotation_correction_deg', pd.Series([0])).mean())

        x = np.arange(len(experiments))
        width = 0.35

        plt.subplot(1, 2, 1)
        plt.bar(x, translation_means, width)
        plt.xlabel('Experiment Configuration')
        plt.ylabel('Mean Translation Correction (m)')
        plt.title('Translation Corrections')
        plt.xticks(x, experiments, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(x, rotation_means, width, color='orange')
        plt.xlabel('Experiment Configuration')
        plt.ylabel('Mean Rotation Correction (deg)')
        plt.title('Rotation Corrections')
        plt.xticks(x, experiments, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'constraint_corrections.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/constraint_corrections.png")
        plt.close()

    # Plot 2: Trajectory smoothness comparison
    if 'trajectory_smoothness_score' in df.columns:
        plt.figure(figsize=(10, 6))

        experiments = df['experiment'].unique()
        smoothness_scores = []
        smoothness_std = []

        for exp in experiments:
            exp_data = df[df['experiment'] == exp]
            if 'trajectory_smoothness_score' in exp_data:
                smoothness_scores.append(exp_data['trajectory_smoothness_score'].mean())
                smoothness_std.append(exp_data['trajectory_smoothness_score'].std())
            else:
                smoothness_scores.append(0)
                smoothness_std.append(0)

        x = np.arange(len(experiments))
        plt.bar(x, smoothness_scores, yerr=smoothness_std, capsize=5, color='green', alpha=0.7)
        plt.xlabel('Experiment Configuration')
        plt.ylabel('Trajectory Smoothness Score')
        plt.title('Trajectory Smoothness Comparison')
        plt.xticks(x, experiments, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'smoothness_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/smoothness_comparison.png")
        plt.close()

    # Plot 3: Velocity and acceleration statistics
    if 'mean_velocity' in df.columns and 'mean_acceleration' in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        experiments = df['experiment'].unique()

        # Velocity
        vel_means = []
        vel_stds = []
        for exp in experiments:
            exp_data = df[df['experiment'] == exp]
            vel_means.append(exp_data.get('mean_velocity', pd.Series([0])).mean())
            vel_stds.append(exp_data.get('std_velocity', pd.Series([0])).mean())

        x = np.arange(len(experiments))
        axes[0].bar(x, vel_means, color='blue', alpha=0.7, label='Mean Velocity')
        axes[0].bar(x, vel_stds, color='lightblue', alpha=0.7, label='Velocity Std Dev', bottom=vel_means)
        axes[0].set_ylabel('Velocity (m/s)')
        axes[0].set_title('Motion Velocity Statistics')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Acceleration
        acc_means = []
        acc_stds = []
        for exp in experiments:
            exp_data = df[df['experiment'] == exp]
            acc_means.append(exp_data.get('mean_acceleration', pd.Series([0])).mean())
            acc_stds.append(exp_data.get('std_acceleration', pd.Series([0])).mean())

        axes[1].bar(x, acc_means, color='red', alpha=0.7, label='Mean Acceleration')
        axes[1].bar(x, acc_stds, color='lightcoral', alpha=0.7, label='Acceleration Std Dev', bottom=acc_means)
        axes[1].set_xlabel('Experiment Configuration')
        axes[1].set_ylabel('Acceleration (m/s²)')
        axes[1].set_title('Motion Acceleration Statistics')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(experiments, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'motion_statistics.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/motion_statistics.png")
        plt.close()


def generate_report(df: pd.DataFrame, analysis: Dict, output_file: str):
    """Generate markdown report of ablation study."""
    with open(output_file, 'w') as f:
        f.write("# Kinematic Constraints Ablation Study Report\n\n")
        f.write("## Nasal Endoscopy SLAM - HAMLYN Dataset\n\n")

        f.write("### Experiment Overview\n\n")
        f.write(f"- Total experiments: {len(df['experiment'].unique())}\n")
        f.write(f"- Total sequences tested: {len(df)}\n")
        f.write(f"- Datasets: {', '.join(df['dataset'].unique())}\n\n")

        f.write("### Experiment Configurations\n\n")
        for exp in df['experiment'].unique():
            f.write(f"- **{exp}**\n")

        f.write("\n### Results Summary\n\n")

        # Summary table
        if 'summary_statistics' in analysis:
            f.write("#### Statistical Summary\n\n")
            summary = analysis['summary_statistics']
            f.write(summary.to_markdown() if hasattr(summary, 'to_markdown') else str(summary))
            f.write("\n\n")

        # Key findings
        f.write("### Key Findings\n\n")

        # Compute some key statistics
        if 'experiment' in df.columns:
            for exp in df['experiment'].unique():
                exp_data = df[df['experiment'] == exp]
                f.write(f"\n#### {exp}\n\n")

                if 'mean_translation_correction' in exp_data:
                    mean_trans = exp_data['mean_translation_correction'].mean()
                    f.write(f"- Average translation correction: {mean_trans:.6f} m\n")

                if 'mean_rotation_correction_deg' in exp_data:
                    mean_rot = exp_data['mean_rotation_correction_deg'].mean()
                    f.write(f"- Average rotation correction: {mean_rot:.3f}°\n")

                if 'trajectory_smoothness_score' in exp_data:
                    smooth = exp_data['trajectory_smoothness_score'].mean()
                    f.write(f"- Trajectory smoothness score: {smooth:.4f}\n")

        f.write("\n### Conclusion\n\n")
        f.write("This ablation study evaluates the impact of kinematic constraints ")
        f.write("on SLAM performance for nasal endoscopy. The results show ")
        f.write("the trade-offs between different constraint configurations.\n")

    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ablation study results')
    parser.add_argument('--results_dir', type=str,
                       default='../VSLAM-LAB-Evaluation/exp_hamlyn_ablation',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str,
                       default='ablation_results',
                       help='Output directory for analysis')
    parser.add_argument('--configs', nargs='+',
                       default=[
                           'exp_droidslam_baseline',
                           'exp_droidslam_no_constraints',
                           'exp_droidslam_relaxed',
                           'exp_droidslam_default',
                           'exp_droidslam_strict'
                       ],
                       help='Experiment configurations to analyze')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ABLATION STUDY EVALUATION")
    print("="*60 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect results
    print("Collecting experiment results...")
    df = collect_experiment_results(args.results_dir, args.configs)

    if df.empty:
        print("Warning: No results found!")
        print(f"Checked directory: {args.results_dir}")
        print(f"Configurations: {args.configs}")
        return

    print(f"Found {len(df)} result entries\n")

    # Save raw results
    results_csv = os.path.join(args.output_dir, 'ablation_results.csv')
    df.to_csv(results_csv, index=False)
    print(f"Saved raw results to: {results_csv}\n")

    # Analyze results
    print("Analyzing results...")
    analysis = analyze_ablation_results(df)

    # Save analysis
    analysis_json = os.path.join(args.output_dir, 'ablation_analysis.json')
    # Convert non-serializable objects
    analysis_serializable = {}
    for key, value in analysis.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            analysis_serializable[key] = value.to_dict()
        else:
            analysis_serializable[key] = value

    with open(analysis_json, 'w') as f:
        json.dump(analysis_serializable, f, indent=2)
    print(f"Saved analysis to: {analysis_json}\n")

    # Generate plots
    print("Generating visualization plots...")
    plot_ablation_comparison(df, args.output_dir)

    # Generate report
    print("\nGenerating report...")
    report_file = os.path.join(args.output_dir, 'ABLATION_REPORT.md')
    generate_report(df, analysis, report_file)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
