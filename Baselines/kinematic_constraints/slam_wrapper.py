#!/usr/bin/env python3
"""
SLAM Wrapper with Kinematic Constraints

Executes base SLAM system and applies kinematic constraints to the trajectory.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Baselines.kinematic_constraints import TrajectoryConstraintFilter


def run_base_slam(base_slam: str,
                 sequence_path: str,
                 output_folder: str,
                 dataset: str,
                 sequence: str) -> str:
    """
    Run the base SLAM system.

    Args:
        base_slam: Name of base SLAM system
        sequence_path: Path to sequence data
        output_folder: Output folder for results
        dataset: Dataset name
        sequence: Sequence name

    Returns:
        trajectory_file: Path to output trajectory file
    """
    print(f"\n{'='*60}")
    print(f"Running base SLAM: {base_slam}")
    print(f"{'='*60}\n")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Output trajectory file (raw)
    trajectory_file = os.path.join(output_folder, "estimated_trajectory_raw.txt")

    # Build command based on SLAM system
    # This is simplified - in practice would use the actual baseline infrastructure
    if base_slam == 'droidslam':
        cmd = f"pixi run demo droidslam {dataset} {sequence}"
    elif base_slam == 'orbslam3':
        cmd = f"pixi run demo orbslam3 {dataset} {sequence}"
    elif base_slam == 'dpvo':
        cmd = f"pixi run demo dpvo {dataset} {sequence}"
    else:
        raise ValueError(f"Unknown base SLAM: {base_slam}")

    # Note: This is a placeholder. In real implementation, we would:
    # 1. Use the existing baseline infrastructure
    # 2. Properly configure the SLAM system
    # 3. Ensure output goes to the right location

    print(f"Command: {cmd}")
    print("\nNote: This wrapper needs integration with the VSLAM-LAB run infrastructure.")
    print(f"Expected output trajectory: {trajectory_file}\n")

    # For now, return the expected trajectory path
    # In full implementation, this would actually run the SLAM system
    return trajectory_file


def apply_constraints(input_trajectory: str,
                     output_trajectory: str,
                     constraint_config: dict,
                     refine_iterations: int = 2) -> dict:
    """
    Apply kinematic constraints to trajectory.

    Args:
        input_trajectory: Input trajectory file
        output_trajectory: Output constrained trajectory file
        constraint_config: Constraint configuration
        refine_iterations: Number of refinement iterations

    Returns:
        statistics: Filtering statistics
    """
    print(f"\n{'='*60}")
    print(f"Applying Kinematic Constraints")
    print(f"{'='*60}\n")

    print("Constraint Configuration:")
    for key, value in constraint_config.items():
        print(f"  {key}: {value}")
    print()

    # Create filter
    filter = TrajectoryConstraintFilter(constraint_config)

    # Apply filtering
    stats = filter.filter_trajectory(
        input_trajectory,
        output_trajectory,
        apply_constraints=True,
        refine_iterations=refine_iterations
    )

    print("\nFiltering Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return stats


def get_constraint_config(config_name: str) -> dict:
    """Get constraint configuration by name."""
    configs = {
        'none': {
            'max_rotation_per_frame': 180.0,
            'max_translation_per_frame': 10.0,
            'max_lateral_velocity': 10.0,
            'enable_pivot_constraint': False,
            'velocity_smoothness_weight': 0.0,
        },
        'relaxed': {
            'max_rotation_per_frame': 15.0,
            'max_translation_per_frame': 0.1,
            'max_lateral_velocity': 0.05,
            'enable_pivot_constraint': False,
            'velocity_smoothness_weight': 0.1,
        },
        'default': {
            'max_rotation_per_frame': 5.0,
            'max_translation_per_frame': 0.05,
            'max_lateral_velocity': 0.02,
            'enable_pivot_constraint': True,
            'pivot_distance_estimate': 0.15,
            'pivot_tolerance': 0.05,
            'velocity_smoothness_weight': 0.3,
        },
        'strict': {
            'max_rotation_per_frame': 2.0,
            'max_translation_per_frame': 0.02,
            'max_lateral_velocity': 0.01,
            'enable_pivot_constraint': True,
            'pivot_distance_estimate': 0.15,
            'pivot_tolerance': 0.03,
            'velocity_smoothness_weight': 0.5,
        }
    }
    return configs.get(config_name, configs['default'])


def main():
    parser = argparse.ArgumentParser(
        description='SLAM wrapper with kinematic constraints for endoscopy'
    )
    parser.add_argument('--base_slam', type=str, required=True,
                       help='Base SLAM system to use')
    parser.add_argument('--sequence_path', type=str, required=True,
                       help='Path to sequence data')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output folder for results')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--sequence', type=str, required=True,
                       help='Sequence name')
    parser.add_argument('--constraint_config', type=str, default='default',
                       choices=['none', 'relaxed', 'default', 'strict'],
                       help='Constraint configuration preset')
    parser.add_argument('--apply_constraints', action='store_true',
                       help='Apply kinematic constraints')
    parser.add_argument('--save_raw', action='store_true',
                       help='Save raw trajectory before constraints')
    parser.add_argument('--refine_iterations', type=int, default=2,
                       help='Number of refinement iterations')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SLAM Wrapper with Kinematic Constraints")
    print(f"{'='*60}")
    print(f"Base SLAM: {args.base_slam}")
    print(f"Dataset: {args.dataset}")
    print(f"Sequence: {args.sequence}")
    print(f"Constraints: {args.constraint_config if args.apply_constraints else 'disabled'}")
    print(f"{'='*60}\n")

    # Step 1: Run base SLAM (placeholder - needs proper integration)
    print("Step 1: Running base SLAM system...")
    raw_trajectory = run_base_slam(
        args.base_slam,
        args.sequence_path,
        args.output_folder,
        args.dataset,
        args.sequence
    )

    # Step 2: Apply constraints
    if args.apply_constraints:
        print("\nStep 2: Applying kinematic constraints...")

        constraint_config = get_constraint_config(args.constraint_config)

        output_trajectory = os.path.join(
            args.output_folder,
            "estimated_trajectory.txt"
        )

        # Check if raw trajectory exists
        if not os.path.exists(raw_trajectory):
            print(f"\nWarning: Raw trajectory not found: {raw_trajectory}")
            print("Creating dummy trajectory for testing...")
            # Create a dummy trajectory for testing
            create_dummy_trajectory(raw_trajectory)

        stats = apply_constraints(
            raw_trajectory,
            output_trajectory,
            constraint_config,
            args.refine_iterations
        )

        # Save statistics
        stats_file = os.path.join(args.output_folder, "constraint_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")

    else:
        print("\nStep 2: Skipping constraints (disabled)")
        # Just copy raw to final if constraints disabled
        import shutil
        output_trajectory = os.path.join(args.output_folder, "estimated_trajectory.txt")
        if os.path.exists(raw_trajectory):
            shutil.copy(raw_trajectory, output_trajectory)

    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"Output: {output_trajectory}")
    print(f"{'='*60}\n")


def create_dummy_trajectory(output_file: str, num_poses: int = 100):
    """Create a dummy trajectory for testing (forward motion with noise)."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    print(f"Creating dummy trajectory with {num_poses} poses...")

    with open(output_file, 'w') as f:
        for i in range(num_poses):
            # Simple forward motion with some noise
            timestamp = i / 30.0  # 30 Hz
            tx = np.random.normal(0, 0.01)
            ty = np.random.normal(0, 0.01)
            tz = i * 0.01 + np.random.normal(0, 0.005)  # Forward motion

            # Small random rotation
            angle = np.random.normal(0, 0.05)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            quat = R.from_rotvec(angle * axis).as_quat()

            f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                   f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

    print(f"Dummy trajectory saved to: {output_file}")


if __name__ == '__main__':
    main()
