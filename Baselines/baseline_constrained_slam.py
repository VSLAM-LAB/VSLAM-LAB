"""
Constrained SLAM Baseline

Wraps existing SLAM systems and applies kinematic constraints for endoscopy.
Can be used with ORB-SLAM3, DROID-SLAM, DPVO, etc.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab
from Baselines.kinematic_constraints import (
    EndoscopeKinematicConstraints,
    ConstrainedMotionModel,
    TrajectoryConstraintFilter
)

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class ConstrainedSLAM_baseline(BaselineVSLAMLab):
    """
    SLAM baseline with kinematic constraints for endoscopy.

    This baseline:
    1. Runs an underlying SLAM system
    2. Post-processes the trajectory with kinematic constraints
    3. Saves both raw and constrained trajectories
    """

    def __init__(self,
                 base_slam: str = 'droidslam',
                 constraint_config: str = 'default'):
        """
        Initialize constrained SLAM baseline.

        Args:
            base_slam: Base SLAM system to use ('droidslam', 'orbslam3', 'dpvo')
            constraint_config: Constraint configuration ('default', 'strict', 'relaxed', 'none')
        """
        baseline_name = f'{base_slam}_constrained'
        baseline_folder = f'{base_slam.upper()}_CONSTRAINED'

        default_parameters = {
            'verbose': 1,
            'apply_constraints': True,
            'constraint_config': constraint_config,
            'save_raw_trajectory': True,
            'refine_iterations': 2
        }

        super().__init__(baseline_name, baseline_folder, default_parameters)

        self.base_slam = base_slam
        self.constraint_config = constraint_config
        self.color = 'purple'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        """
        Build execution command that wraps base SLAM and applies constraints.
        """
        # This will be a Python script that:
        # 1. Runs the base SLAM system
        # 2. Loads the output trajectory
        # 3. Applies kinematic constraints
        # 4. Saves the constrained trajectory

        sequence_path = os.path.join(dataset.dataset_path, sequence_name)
        exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)

        # Create wrapper script path
        wrapper_script = os.path.join(
            os.path.dirname(__file__),
            'kinematic_constraints',
            'slam_wrapper.py'
        )

        # Build command
        command = f"python {wrapper_script}"
        command += f" --base_slam {self.base_slam}"
        command += f" --sequence_path {sequence_path}"
        command += f" --output_folder {exp_folder}"
        command += f" --dataset {dataset.dataset_label}"
        command += f" --sequence {sequence_name}"
        command += f" --constraint_config {self.constraint_config}"

        if self.parameters.get('apply_constraints', True):
            command += " --apply_constraints"

        if self.parameters.get('save_raw_trajectory', True):
            command += " --save_raw"

        refine_iters = self.parameters.get('refine_iterations', 2)
        command += f" --refine_iterations {refine_iters}"

        return command

    def is_installed(self):
        """Check if baseline is installed."""
        # Check if base SLAM is installed
        # This is a simplified check - could be more sophisticated
        return (True, f'Wrapper for {self.base_slam}')

    def get_constraint_config(self, config_name: str = 'default') -> dict:
        """
        Get constraint configuration by name.

        Args:
            config_name: Configuration name

        Returns:
            config: Constraint configuration dictionary
        """
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
                'velocity_smoothness_weight': 0.3,
            },
            'strict': {
                'max_rotation_per_frame': 2.0,
                'max_translation_per_frame': 0.02,
                'max_lateral_velocity': 0.01,
                'enable_pivot_constraint': True,
                'pivot_distance_estimate': 0.15,
                'velocity_smoothness_weight': 0.5,
            }
        }

        return configs.get(config_name, configs['default'])


# Specific instantiations for different base SLAM systems

class DROIDSLAM_Constrained_baseline(ConstrainedSLAM_baseline):
    """DROID-SLAM with kinematic constraints."""

    def __init__(self, constraint_config: str = 'default'):
        super().__init__(base_slam='droidslam', constraint_config=constraint_config)


class ORBSLAM3_Constrained_baseline(ConstrainedSLAM_baseline):
    """ORB-SLAM3 with kinematic constraints."""

    def __init__(self, constraint_config: str = 'default'):
        super().__init__(base_slam='orbslam3', constraint_config=constraint_config)


class DPVO_Constrained_baseline(ConstrainedSLAM_baseline):
    """DPVO with kinematic constraints."""

    def __init__(self, constraint_config: str = 'default'):
        super().__init__(base_slam='dpvo', constraint_config=constraint_config)
