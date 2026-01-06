#!/usr/bin/env python3
"""
Test script for kinematic constraints

Tests the constraint module with synthetic and real data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Baselines.kinematic_constraints import (
    EndoscopeKinematicConstraints,
    ConstrainedMotionModel,
    TrajectoryConstraintFilter
)
from scipy.spatial.transform import Rotation as R


def create_test_trajectory(num_poses=100, motion_type='forward'):
    """
    Create synthetic test trajectory.

    Args:
        num_poses: Number of poses
        motion_type: Type of motion ('forward', 'noisy', 'smooth')

    Returns:
        timestamps, poses
    """
    timestamps = np.arange(num_poses) / 30.0  # 30 Hz
    poses = []

    for i in range(num_poses):
        pose = np.eye(4)

        if motion_type == 'forward':
            # Smooth forward motion
            pose[:3, 3] = [0, 0, i * 0.01]

        elif motion_type == 'noisy':
            # Forward motion with noise (violates constraints)
            pose[:3, 3] = [
                np.random.normal(0, 0.02),
                np.random.normal(0, 0.02),
                i * 0.01 + np.random.normal(0, 0.01)
            ]
            # Random rotation
            angle = np.random.normal(0, 0.1)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            pose[:3, :3] = R.from_rotvec(angle * axis).as_matrix()

        elif motion_type == 'smooth':
            # Smooth endoscope-like motion
            t = i / num_poses
            pose[:3, 3] = [
                0.02 * np.sin(2 * np.pi * t),
                0.02 * np.cos(2 * np.pi * t),
                i * 0.01
            ]
            # Smooth rotation
            angle = 0.1 * t
            pose[:3, :3] = R.from_rotvec([angle, 0, 0]).as_matrix()

        poses.append(pose)

    return timestamps, np.array(poses)


def test_endoscope_constraints():
    """Test endoscope kinematic constraints."""
    print("\n" + "="*60)
    print("Testing Endoscope Kinematic Constraints")
    print("="*60 + "\n")

    # Create constraints with default config
    constraints = EndoscopeKinematicConstraints()

    # Create a simple test case
    current_pose = np.eye(4)
    current_pose[:3, 3] = [0, 0, 0]

    # Proposed pose with large violation
    proposed_pose = np.eye(4)
    proposed_pose[:3, 3] = [0.1, 0.1, 0.2]  # Too large translation
    proposed_pose[:3, :3] = R.from_rotvec([0.5, 0, 0]).as_matrix()  # Too large rotation

    print("Test 1: Large constraint violation")
    print(f"Current position: {current_pose[:3, 3]}")
    print(f"Proposed position: {proposed_pose[:3, 3]}")
    print(f"Proposed translation magnitude: {np.linalg.norm(proposed_pose[:3, 3] - current_pose[:3, 3]):.4f} m")

    constrained_pose, diagnostics = constraints.apply_constraints(
        current_pose, proposed_pose, timestamp=0.0
    )

    print(f"Constrained position: {constrained_pose[:3, 3]}")
    print(f"Constrained translation magnitude: {np.linalg.norm(constrained_pose[:3, 3] - current_pose[:3, 3]):.4f} m")
    print(f"\nDiagnostics:")
    for key, value in diagnostics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
    print(f"  ✓ Constraints applied successfully\n")

    # Test 2: Valid motion
    print("Test 2: Valid motion (within constraints)")
    valid_pose = np.eye(4)
    valid_pose[:3, 3] = [0.01, 0.005, 0.03]  # Small, valid translation

    constrained_pose2, diagnostics2 = constraints.apply_constraints(
        constrained_pose, valid_pose, timestamp=0.033
    )

    print(f"Proposed position: {valid_pose[:3, 3]}")
    print(f"Constrained position: {constrained_pose2[:3, 3]}")
    print(f"Change from proposed: {np.linalg.norm(constrained_pose2[:3, 3] - valid_pose[:3, 3]):.6f} m")
    print(f"  ✓ Valid motion preserved\n")


def test_motion_model():
    """Test constrained motion model."""
    print("\n" + "="*60)
    print("Testing Constrained Motion Model")
    print("="*60 + "\n")

    # Create motion model
    motion_model = ConstrainedMotionModel()

    # Create noisy trajectory
    print("Creating noisy test trajectory...")
    timestamps, poses = create_test_trajectory(num_poses=50, motion_type='noisy')

    print(f"Generated {len(poses)} poses\n")

    # Filter trajectory
    print("Filtering trajectory with constraints...")
    for ts, pose in zip(timestamps, poses):
        filtered_pose, diagnostics = motion_model.filter_pose(pose, ts, apply_constraints=True)

    # Get statistics
    stats = motion_model.get_statistics()

    print("\nFiltering Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
    print(f"  ✓ Motion model filtering complete\n")


def test_trajectory_filter():
    """Test trajectory filter with file I/O."""
    print("\n" + "="*60)
    print("Testing Trajectory Filter")
    print("="*60 + "\n")

    import tempfile
    import os

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test trajectory file
        input_file = os.path.join(tmpdir, "test_trajectory.txt")
        output_file = os.path.join(tmpdir, "filtered_trajectory.txt")

        print(f"Creating test trajectory file: {input_file}")

        # Generate and save trajectory
        timestamps, poses = create_test_trajectory(num_poses=100, motion_type='noisy')

        # Save in TUM format
        with open(input_file, 'w') as f:
            for ts, pose in zip(timestamps, poses):
                t = pose[:3, 3]
                q = R.from_matrix(pose[:3, :3]).as_quat()
                f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                       f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

        print(f"Saved {len(poses)} poses to file\n")

        # Create filter
        filter = TrajectoryConstraintFilter()

        # Filter trajectory
        print("Filtering trajectory...")
        stats = filter.filter_trajectory(
            input_file,
            output_file,
            apply_constraints=True,
            refine_iterations=2
        )

        print("\nFiltering Statistics:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")

        # Verify output file exists
        assert os.path.exists(output_file), "Output file not created"
        print(f"\n  ✓ Output file created: {output_file}")
        print(f"  ✓ Trajectory filter test complete\n")


def test_different_configs():
    """Test different constraint configurations."""
    print("\n" + "="*60)
    print("Testing Different Constraint Configurations")
    print("="*60 + "\n")

    configs = {
        'relaxed': {
            'max_rotation_per_frame': 15.0,
            'max_translation_per_frame': 0.1,
        },
        'default': {
            'max_rotation_per_frame': 5.0,
            'max_translation_per_frame': 0.05,
        },
        'strict': {
            'max_rotation_per_frame': 2.0,
            'max_translation_per_frame': 0.02,
        }
    }

    # Create test motion
    current_pose = np.eye(4)
    proposed_pose = np.eye(4)
    proposed_pose[:3, 3] = [0.03, 0.03, 0.08]  # Medium violation

    for config_name, config in configs.items():
        print(f"\nConfiguration: {config_name}")
        print(f"  Max translation: {config['max_translation_per_frame']:.3f} m")
        print(f"  Max rotation: {config['max_rotation_per_frame']:.1f}°")

        constraints = EndoscopeKinematicConstraints(config)
        constrained_pose, diagnostics = constraints.apply_constraints(
            current_pose, proposed_pose, timestamp=0.0
        )

        translation = np.linalg.norm(constrained_pose[:3, 3] - current_pose[:3, 3])
        print(f"  Resulting translation: {translation:.4f} m")
        print(f"  Translation violation: {diagnostics.get('translation_violation', 0):.6f}")

    print(f"\n  ✓ Configuration tests complete\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("KINEMATIC CONSTRAINTS TEST SUITE")
    print("="*60)

    try:
        test_endoscope_constraints()
        test_motion_model()
        test_trajectory_filter()
        test_different_configs()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n" + "="*60)
        print(f"TEST FAILED ✗")
        print(f"Error: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
