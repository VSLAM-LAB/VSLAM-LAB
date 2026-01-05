from Baselines.BaselineVSLAMLab import BaselineVSLAMLab


class DSO_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'dso'
        baseline_folder = 'dso'
        default_parameters = ['Preset: preset:0', 'Mode: mode:1']

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)