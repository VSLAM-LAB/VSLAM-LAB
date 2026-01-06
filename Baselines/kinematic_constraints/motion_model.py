"""
Constrained Motion Model for Endoscopic SLAM

Integrates kinematic constraints into pose estimation and refinement.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize
from .endoscope_constraints import EndoscopeKinematicConstraints


class ConstrainedMotionModel:
    """
    Motion model that enforces endoscope kinematic constraints.

    Can be used to:
    1. Filter raw SLAM poses
    2. Predict next pose with constraints
    3. Refine trajectory smoothness
    """

    def __init__(self, constraints: Optional[EndoscopeKinematicConstraints] = None):
        """
        Initialize constrained motion model.

        Args:
            constraints: Endoscope kinematic constraints instance
        """
        self.constraints = constraints or EndoscopeKinematicConstraints()
        self.filtered_poses = []
        self.raw_poses = []
        self.timestamps = []

    def filter_pose(self,
                   raw_pose: np.ndarray,
                   timestamp: float,
                   apply_constraints: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Filter a single pose using kinematic constraints.

        Args:
            raw_pose: Raw pose from SLAM [4x4 matrix]
            timestamp: Timestamp
            apply_constraints: Whether to apply constraints

        Returns:
            filtered_pose: Constrained pose
            diagnostics: Constraint diagnostics
        """
        self.raw_poses.append(raw_pose.copy())
        self.timestamps.append(timestamp)

        if not apply_constraints or len(self.filtered_poses) == 0:
            # First pose or constraints disabled
            self.filtered_poses.append(raw_pose.copy())
            self.constraints.reset()
            return raw_pose.copy(), {}

        # Apply constraints relative to previous filtered pose
        current_pose = self.filtered_poses[-1]
        filtered_pose, diagnostics = self.constraints.apply_constraints(
            current_pose, raw_pose, timestamp
        )

        self.filtered_poses.append(filtered_pose)
        return filtered_pose, diagnostics

    def predict_next_pose(self,
                         dt: float,
                         use_history: int = 3) -> Optional[np.ndarray]:
        """
        Predict next pose based on motion history.

        Args:
            dt: Time step for prediction
            use_history: Number of previous poses to use

        Returns:
            predicted_pose: Predicted next pose [4x4 matrix]
        """
        if len(self.filtered_poses) < use_history:
            return None

        # Simple constant velocity model
        poses = self.filtered_poses[-use_history:]

        # Estimate velocity from recent poses
        velocities = []
        for i in range(len(poses) - 1):
            t_delta = poses[i + 1][:3, 3] - poses[i][:3, 3]
            velocities.append(t_delta)

        # Average velocity
        avg_velocity = np.mean(velocities, axis=0)

        # Predict translation
        last_pose = poses[-1].copy()
        predicted_pose = last_pose.copy()
        predicted_pose[:3, 3] = last_pose[:3, 3] + avg_velocity

        # Predict rotation (assume constant)
        # Could be extended to angular velocity model
        predicted_pose[:3, :3] = last_pose[:3, :3]

        return predicted_pose

    def refine_trajectory(self,
                         poses: List[np.ndarray],
                         timestamps: List[float],
                         iterations: int = 5) -> List[np.ndarray]:
        """
        Refine full trajectory using constraint optimization.

        Args:
            poses: List of poses to refine
            timestamps: Corresponding timestamps
            iterations: Number of refinement iterations

        Returns:
            refined_poses: Smoothed trajectory
        """
        if len(poses) < 2:
            return poses

        refined = [poses[0].copy()]  # Keep first pose fixed

        for i in range(1, len(poses)):
            current = refined[-1]
            target = poses[i]

            # Iteratively refine toward target while respecting constraints
            refined_pose = current.copy()
            for _ in range(iterations):
                # Blend toward target
                alpha = 0.5  # Blending factor
                blended = self._interpolate_poses(refined_pose, target, alpha)

                # Apply constraints
                constrained, _ = self.constraints.apply_constraints(
                    current, blended, timestamps[i]
                )
                refined_pose = constrained

            refined.append(refined_pose)

        return refined

    def _interpolate_poses(self,
                          pose1: np.ndarray,
                          pose2: np.ndarray,
                          alpha: float) -> np.ndarray:
        """
        Interpolate between two poses.

        Args:
            pose1: First pose
            pose2: Second pose
            alpha: Interpolation factor [0, 1]

        Returns:
            interpolated_pose: Blended pose
        """
        from scipy.spatial.transform import Rotation as R, Slerp

        # Interpolate translation
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]
        t_interp = (1 - alpha) * t1 + alpha * t2

        # Interpolate rotation using SLERP
        r1 = R.from_matrix(pose1[:3, :3])
        r2 = R.from_matrix(pose2[:3, :3])

        # Create interpolator
        key_times = [0, 1]
        key_rots = R.from_quat([r1.as_quat(), r2.as_quat()])
        slerp = Slerp(key_times, key_rots)
        r_interp = slerp(alpha)

        # Construct interpolated pose
        pose_interp = np.eye(4)
        pose_interp[:3, :3] = r_interp.as_matrix()
        pose_interp[:3, 3] = t_interp

        return pose_interp

    def get_statistics(self) -> Dict:
        """
        Get statistics about constraint violations and corrections.

        Returns:
            stats: Dictionary with statistics
        """
        if len(self.raw_poses) < 2:
            return {}

        # Compute differences between raw and filtered
        translation_diffs = []
        rotation_diffs = []

        for raw, filtered in zip(self.raw_poses, self.filtered_poses):
            # Translation difference
            t_diff = np.linalg.norm(raw[:3, 3] - filtered[:3, 3])
            translation_diffs.append(t_diff)

            # Rotation difference (angle)
            R_diff = filtered[:3, :3] @ raw[:3, :3].T
            from scipy.spatial.transform import Rotation as R
            angle = np.linalg.norm(R.from_matrix(R_diff).as_rotvec())
            rotation_diffs.append(angle)

        return {
            'mean_translation_correction': np.mean(translation_diffs),
            'max_translation_correction': np.max(translation_diffs),
            'mean_rotation_correction': np.rad2deg(np.mean(rotation_diffs)),
            'max_rotation_correction': np.rad2deg(np.max(rotation_diffs)),
            'total_poses_filtered': len(self.filtered_poses)
        }

    def reset(self):
        """Reset motion model state."""
        self.filtered_poses = []
        self.raw_poses = []
        self.timestamps = []
        self.constraints.reset()
