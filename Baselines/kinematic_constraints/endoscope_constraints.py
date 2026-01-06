"""
Endoscope Kinematic Constraints for Nasal Endoscopy

This module implements physical motion constraints based on:
1. Rigid endoscope structure
2. Nasal cavity anatomical constraints
3. Insertion point pivot constraints
4. Smooth motion requirements
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, Dict


class EndoscopeKinematicConstraints:
    """
    Implements kinematic constraints for nasal endoscopy procedures.

    Key constraints:
    - Limited rotation around insertion point (pivot constraint)
    - Preferred forward/backward motion along endoscope axis
    - Smooth motion (velocity and acceleration limits)
    - Anatomical bounds (limited lateral movement)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize endoscope kinematic constraints.

        Args:
            config: Dictionary with constraint parameters
        """
        # Default configuration
        default_config = {
            # Rotation constraints (degrees)
            'max_rotation_per_frame': 5.0,  # Maximum rotation between frames
            'max_tilt_angle': 45.0,  # Maximum tilt from insertion axis

            # Translation constraints (meters)
            'max_translation_per_frame': 0.05,  # 5cm max movement per frame
            'max_lateral_velocity': 0.02,  # Limited lateral movement
            'preferred_forward_ratio': 0.7,  # Prefer forward motion

            # Smoothness constraints
            'velocity_smoothness_weight': 0.3,
            'acceleration_limit': 0.1,  # m/s^2

            # Pivot point constraints
            'enable_pivot_constraint': True,
            'pivot_distance_estimate': 0.15,  # Estimated distance to insertion point (m)
            'pivot_tolerance': 0.05,

            # Temporal consistency
            'temporal_window': 5,  # Frames for smoothing
        }

        self.config = {**default_config, **(config or {})}

        # Convert angles to radians
        self.max_rotation_rad = np.deg2rad(self.config['max_rotation_per_frame'])
        self.max_tilt_rad = np.deg2rad(self.config['max_tilt_angle'])

        # State tracking
        self.pose_history = []
        self.velocity_history = []

    def apply_constraints(self,
                         current_pose: np.ndarray,
                         proposed_pose: np.ndarray,
                         timestamp: float) -> Tuple[np.ndarray, Dict]:
        """
        Apply kinematic constraints to a proposed pose.

        Args:
            current_pose: Current pose [4x4 transformation matrix]
            proposed_pose: Proposed next pose [4x4 transformation matrix]
            timestamp: Current timestamp

        Returns:
            constrained_pose: Adjusted pose respecting constraints
            diagnostics: Dictionary with constraint violation information
        """
        diagnostics = {
            'rotation_violation': 0.0,
            'translation_violation': 0.0,
            'pivot_violation': 0.0,
            'smoothness_violation': 0.0,
            'applied_corrections': []
        }

        constrained_pose = proposed_pose.copy()

        # Extract current and proposed transformations
        R_curr = current_pose[:3, :3]
        t_curr = current_pose[:3, 3]
        R_prop = proposed_pose[:3, :3]
        t_prop = proposed_pose[:3, 3]

        # 1. Rotation constraint
        R_prop, rot_viol = self._constrain_rotation(R_curr, R_prop)
        diagnostics['rotation_violation'] = rot_viol

        # 2. Translation constraint
        t_prop, trans_viol = self._constrain_translation(t_curr, t_prop, R_curr)
        diagnostics['translation_violation'] = trans_viol

        # 3. Pivot point constraint (if enabled)
        if self.config['enable_pivot_constraint'] and len(self.pose_history) > 3:
            t_prop, pivot_viol = self._apply_pivot_constraint(t_curr, t_prop, R_curr)
            diagnostics['pivot_violation'] = pivot_viol

        # 4. Smoothness constraint
        if len(self.pose_history) >= 2:
            t_prop, smooth_viol = self._apply_smoothness_constraint(t_curr, t_prop, timestamp)
            diagnostics['smoothness_violation'] = smooth_viol

        # Update constrained pose
        constrained_pose[:3, :3] = R_prop
        constrained_pose[:3, 3] = t_prop

        # Update history
        self._update_history(constrained_pose, timestamp)

        return constrained_pose, diagnostics

    def _constrain_rotation(self, R_curr: np.ndarray, R_prop: np.ndarray) -> Tuple[np.ndarray, float]:
        """Limit rotation magnitude between frames."""
        # Compute relative rotation
        R_rel = R_prop @ R_curr.T
        rotvec = R.from_matrix(R_rel).as_rotvec()
        angle = np.linalg.norm(rotvec)

        violation = max(0, angle - self.max_rotation_rad)

        if angle > self.max_rotation_rad:
            # Scale down rotation
            scale = self.max_rotation_rad / angle
            rotvec_constrained = rotvec * scale
            R_rel_constrained = R.from_rotvec(rotvec_constrained).as_matrix()
            R_constrained = R_rel_constrained @ R_curr
            return R_constrained, violation

        return R_prop, violation

    def _constrain_translation(self,
                               t_curr: np.ndarray,
                               t_prop: np.ndarray,
                               R_curr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Constrain translation based on endoscope motion model."""
        delta_t = t_prop - t_curr
        distance = np.linalg.norm(delta_t)

        violation = max(0, distance - self.config['max_translation_per_frame'])

        if distance > self.config['max_translation_per_frame']:
            # Scale down translation
            delta_t = delta_t * (self.config['max_translation_per_frame'] / distance)

        # Decompose into forward and lateral components
        # Forward direction is the -Z axis of the camera (OpenCV convention)
        forward_dir = R_curr @ np.array([0, 0, -1])
        forward_component = np.dot(delta_t, forward_dir)
        lateral_component = delta_t - forward_component * forward_dir

        # Constrain lateral movement
        lateral_magnitude = np.linalg.norm(lateral_component)
        if lateral_magnitude > self.config['max_lateral_velocity']:
            lateral_component = lateral_component * (self.config['max_lateral_velocity'] / lateral_magnitude)
            violation += lateral_magnitude - self.config['max_lateral_velocity']

        # Reconstruct translation with preference for forward motion
        delta_t_constrained = forward_component * forward_dir + lateral_component

        return t_curr + delta_t_constrained, violation

    def _apply_pivot_constraint(self,
                               t_curr: np.ndarray,
                               t_prop: np.ndarray,
                               R_curr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply pivot point constraint - endoscope rotates around insertion point.
        """
        # Estimate pivot point location (behind the camera)
        pivot_direction = R_curr @ np.array([0, 0, 1])  # Backward direction
        estimated_pivot = t_curr + pivot_direction * self.config['pivot_distance_estimate']

        # Check if motion is consistent with pivot constraint
        # The distance from pivot should remain approximately constant
        dist_curr = np.linalg.norm(t_curr - estimated_pivot)
        dist_prop = np.linalg.norm(t_prop - estimated_pivot)

        violation = abs(dist_prop - dist_curr)

        if violation > self.config['pivot_tolerance']:
            # Project proposed position onto sphere around pivot
            direction = t_prop - estimated_pivot
            direction_norm = direction / np.linalg.norm(direction)
            t_constrained = estimated_pivot + direction_norm * dist_curr
            return t_constrained, violation

        return t_prop, violation

    def _apply_smoothness_constraint(self,
                                    t_curr: np.ndarray,
                                    t_prop: np.ndarray,
                                    timestamp: float) -> Tuple[np.ndarray, float]:
        """Apply velocity smoothness constraint."""
        if len(self.pose_history) < 2:
            return t_prop, 0.0

        # Compute current velocity
        t_prev = self.pose_history[-1][:3, 3]
        dt = 1.0 / 30.0  # Assume 30 fps (from HAMLYN dataset)

        v_prev = (t_curr - t_prev) / dt
        v_prop = (t_prop - t_curr) / dt

        # Compute acceleration
        acceleration = (v_prop - v_prev) / dt
        acc_magnitude = np.linalg.norm(acceleration)

        violation = max(0, acc_magnitude - self.config['acceleration_limit'])

        if acc_magnitude > self.config['acceleration_limit']:
            # Limit acceleration
            acc_constrained = acceleration * (self.config['acceleration_limit'] / acc_magnitude)
            v_constrained = v_prev + acc_constrained * dt
            t_constrained = t_curr + v_constrained * dt
            return t_constrained, violation

        return t_prop, violation

    def _update_history(self, pose: np.ndarray, timestamp: float):
        """Update pose history for temporal consistency."""
        self.pose_history.append(pose.copy())

        # Keep only recent history
        if len(self.pose_history) > self.config['temporal_window']:
            self.pose_history.pop(0)

    def get_constraint_weights(self) -> Dict[str, float]:
        """Return current constraint weights for ablation studies."""
        return {
            'rotation_weight': 1.0,
            'translation_weight': 1.0,
            'pivot_weight': 1.0 if self.config['enable_pivot_constraint'] else 0.0,
            'smoothness_weight': self.config['velocity_smoothness_weight']
        }

    def reset(self):
        """Reset constraint state."""
        self.pose_history = []
        self.velocity_history = []
