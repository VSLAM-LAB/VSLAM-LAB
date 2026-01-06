"""
Trajectory Constraint Filter

Post-processes SLAM trajectories with kinematic constraints.
Can be used for offline trajectory refinement.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .endoscope_constraints import EndoscopeKinematicConstraints
from .motion_model import ConstrainedMotionModel


class TrajectoryConstraintFilter:
    """
    Filters and refines SLAM trajectories using kinematic constraints.

    Supports:
    - Loading trajectories from TUM format
    - Applying constraints
    - Saving filtered trajectories
    - Generating comparison metrics
    """

    def __init__(self, constraints_config: Optional[Dict] = None):
        """
        Initialize trajectory filter.

        Args:
            constraints_config: Configuration for kinematic constraints
        """
        self.constraints = EndoscopeKinematicConstraints(constraints_config)
        self.motion_model = ConstrainedMotionModel(self.constraints)

    def load_trajectory(self, trajectory_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load trajectory from TUM format file.

        Format: timestamp tx ty tz qx qy qz qw

        Args:
            trajectory_file: Path to trajectory file

        Returns:
            timestamps: Array of timestamps
            poses: Array of 4x4 transformation matrices
        """
        data = pd.read_csv(trajectory_file, sep=' ', header=None,
                          names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        timestamps = data['timestamp'].values
        poses = []

        from scipy.spatial.transform import Rotation as R

        for _, row in data.iterrows():
            pose = np.eye(4)
            pose[:3, 3] = [row['tx'], row['ty'], row['tz']]
            quat = [row['qx'], row['qy'], row['qz'], row['qw']]
            pose[:3, :3] = R.from_quat(quat).as_matrix()
            poses.append(pose)

        return timestamps, np.array(poses)

    def save_trajectory(self,
                       timestamps: np.ndarray,
                       poses: np.ndarray,
                       output_file: str):
        """
        Save trajectory in TUM format.

        Args:
            timestamps: Array of timestamps
            poses: Array of 4x4 transformation matrices
            output_file: Output file path
        """
        from scipy.spatial.transform import Rotation as R

        with open(output_file, 'w') as f:
            for ts, pose in zip(timestamps, poses):
                t = pose[:3, 3]
                q = R.from_matrix(pose[:3, :3]).as_quat()
                f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                       f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    def filter_trajectory(self,
                         input_file: str,
                         output_file: str,
                         apply_constraints: bool = True,
                         refine_iterations: int = 0) -> Dict:
        """
        Filter trajectory file with kinematic constraints.

        Args:
            input_file: Input trajectory file (TUM format)
            output_file: Output filtered trajectory file
            apply_constraints: Whether to apply constraints
            refine_iterations: Number of refinement iterations (0 = single pass)

        Returns:
            statistics: Dictionary with filtering statistics
        """
        # Load trajectory
        timestamps, poses = self.load_trajectory(input_file)
        print(f"Loaded {len(poses)} poses from {input_file}")

        # Reset motion model
        self.motion_model.reset()

        # Filter poses
        filtered_poses = []
        all_diagnostics = []

        for ts, pose in zip(timestamps, poses):
            filtered_pose, diagnostics = self.motion_model.filter_pose(
                pose, ts, apply_constraints=apply_constraints
            )
            filtered_poses.append(filtered_pose)
            all_diagnostics.append(diagnostics)

        filtered_poses = np.array(filtered_poses)

        # Optional refinement pass
        if refine_iterations > 0 and apply_constraints:
            print(f"Applying {refine_iterations} refinement iterations...")
            filtered_poses = self.motion_model.refine_trajectory(
                list(filtered_poses), list(timestamps), refine_iterations
            )
            filtered_poses = np.array(filtered_poses)

        # Save filtered trajectory
        self.save_trajectory(timestamps, filtered_poses, output_file)
        print(f"Saved filtered trajectory to {output_file}")

        # Compute statistics
        stats = self._compute_statistics(poses, filtered_poses, all_diagnostics)

        return stats

    def _compute_statistics(self,
                           raw_poses: np.ndarray,
                           filtered_poses: np.ndarray,
                           diagnostics: List[Dict]) -> Dict:
        """
        Compute filtering statistics.

        Args:
            raw_poses: Original poses
            filtered_poses: Filtered poses
            diagnostics: Per-pose diagnostics

        Returns:
            statistics: Comprehensive statistics dictionary
        """
        stats = {}

        # Pose differences
        translation_errors = []
        rotation_errors = []

        from scipy.spatial.transform import Rotation as R

        for raw, filtered in zip(raw_poses, filtered_poses):
            # Translation difference
            t_err = np.linalg.norm(raw[:3, 3] - filtered[:3, 3])
            translation_errors.append(t_err)

            # Rotation difference
            R_diff = filtered[:3, :3] @ raw[:3, :3].T
            r_err = np.linalg.norm(R.from_matrix(R_diff).as_rotvec())
            rotation_errors.append(r_err)

        stats['mean_translation_error'] = float(np.mean(translation_errors))
        stats['max_translation_error'] = float(np.max(translation_errors))
        stats['std_translation_error'] = float(np.std(translation_errors))

        stats['mean_rotation_error_deg'] = float(np.rad2deg(np.mean(rotation_errors)))
        stats['max_rotation_error_deg'] = float(np.rad2deg(np.max(rotation_errors)))
        stats['std_rotation_error_deg'] = float(np.rad2deg(np.std(rotation_errors)))

        # Constraint violations
        if diagnostics:
            violations = {
                'rotation': [],
                'translation': [],
                'pivot': [],
                'smoothness': []
            }

            for diag in diagnostics:
                violations['rotation'].append(diag.get('rotation_violation', 0))
                violations['translation'].append(diag.get('translation_violation', 0))
                violations['pivot'].append(diag.get('pivot_violation', 0))
                violations['smoothness'].append(diag.get('smoothness_violation', 0))

            for key, values in violations.items():
                if values:
                    stats[f'mean_{key}_violation'] = float(np.mean(values))
                    stats[f'max_{key}_violation'] = float(np.max(values))

        # Trajectory smoothness metrics
        stats.update(self._compute_smoothness_metrics(filtered_poses))

        return stats

    def _compute_smoothness_metrics(self, poses: np.ndarray) -> Dict:
        """
        Compute trajectory smoothness metrics.

        Args:
            poses: Array of poses

        Returns:
            metrics: Smoothness metrics
        """
        if len(poses) < 3:
            return {}

        # Compute velocities
        velocities = []
        for i in range(len(poses) - 1):
            delta_t = poses[i + 1][:3, 3] - poses[i][:3, 3]
            velocities.append(delta_t)

        velocities = np.array(velocities)

        # Compute accelerations
        accelerations = []
        for i in range(len(velocities) - 1):
            delta_v = velocities[i + 1] - velocities[i]
            accelerations.append(delta_v)

        accelerations = np.array(accelerations)

        # Compute jerk (rate of acceleration change)
        jerks = []
        for i in range(len(accelerations) - 1):
            delta_a = accelerations[i + 1] - accelerations[i]
            jerks.append(np.linalg.norm(delta_a))

        return {
            'mean_velocity': float(np.mean(np.linalg.norm(velocities, axis=1))),
            'std_velocity': float(np.std(np.linalg.norm(velocities, axis=1))),
            'mean_acceleration': float(np.mean(np.linalg.norm(accelerations, axis=1))),
            'std_acceleration': float(np.std(np.linalg.norm(accelerations, axis=1))),
            'mean_jerk': float(np.mean(jerks)) if jerks else 0.0,
            'trajectory_smoothness_score': float(1.0 / (1.0 + np.mean(jerks))) if jerks else 1.0
        }

    def batch_filter(self,
                    input_dir: str,
                    output_dir: str,
                    pattern: str = "*.txt",
                    apply_constraints: bool = True) -> Dict[str, Dict]:
        """
        Batch filter multiple trajectory files.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: File pattern to match
            apply_constraints: Whether to apply constraints

        Returns:
            all_stats: Dictionary mapping filenames to statistics
        """
        from pathlib import Path

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_stats = {}

        for traj_file in input_path.glob(pattern):
            print(f"\nProcessing {traj_file.name}...")
            output_file = output_path / traj_file.name

            stats = self.filter_trajectory(
                str(traj_file),
                str(output_file),
                apply_constraints=apply_constraints
            )

            all_stats[traj_file.name] = stats

        return all_stats
