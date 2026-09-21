class BenchmarkVSLAMLAB:
    BM = {
        "droidslam": {
            "euroc": {
                "MH_01_easy": {"ATE": {"median": 0.013}},
                "MH_02_easy": {"ATE": {"median": 0.014}},
                "MH_03_medium": {"ATE": {"median": 0.022}},
                "MH_04_difficult": {"ATE": {"median": 0.043}},
                "MH_05_difficult": {"ATE": {"median": 0.043}},
                "V1_01_easy": {"ATE": {"median": 0.037}},
                "V1_02_medium": {"ATE": {"median": 0.012}},
                "V1_03_difficult": {"ATE": {"median": 0.020}},
                "V2_01_easy": {"ATE": {"median": 0.017}},
                "V2_02_medium": {"ATE": {"median": 0.013}},
                "V2_03_difficult": {"ATE": {"median": 0.014}},
            },
            "rgbdtum": {
                "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.012}},
                "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.026}},
                "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.111}},
                "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.021}},
                "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.018}},
                "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.042}},
                "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.049}},
                "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.016}},
                "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.048}}
            }
        },
        "droidslam-dev": {
            "euroc": {
                "MH_01_easy": {"ATE": {"median": 0.163}},
                "MH_02_easy": {"ATE": {"median": 0.121}},
                "MH_03_medium": {"ATE": {"median": 0.242}},
                "MH_04_difficult": {"ATE": {"median": 0.399}},
                "MH_05_difficult": {"ATE": {"median": 0.270}},
                "V1_01_easy": {"ATE": {"median": 0.103}},
                "V1_02_medium": {"ATE": {"median": 0.165}},
                "V1_03_difficult": {"ATE": {"median": 0.158}},
                "V2_01_easy": {"ATE": {"median": 0.102}},
                "V2_02_medium": {"ATE": {"median": 0.115}},
                "V2_03_difficult": {"ATE": {"median": 0.204}},
            },
            "rgbdtum": {
                "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.0}}
            }
        },
        "dpvo": {
            "euroc": {
                "MH_01_easy": {"ATE": {"median": 0.013}},
                "MH_02_easy": {"ATE": {"median": 0.016}},
                "MH_03_medium": {"ATE": {"median": 0.021}},
                "MH_04_difficult": {"ATE": {"median": 0.041}},
                "MH_05_difficult": {"ATE": {"median": 0.041}},
                "V1_01_easy": {"ATE": {"median": 0.035}},
                "V1_02_medium": {"ATE": {"median": 0.010}},
                "V1_03_difficult": {"ATE": {"median": 0.015}},
                "V2_01_easy": {"ATE": {"median": 0.021}},
                "V2_02_medium": {"ATE": {"median": 0.011}},
                "V2_03_difficult": {"ATE": {"median": 0.023}},
            },
            "rgbdtum": {
                "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.010}},
                "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.032}},
                "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.132}},
                "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.050}},
                "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.018}},
                "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.029}},
                "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.096}},
                "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.022}},
                "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.098}}
            }
        },
        # "dpvo-dev": {
        #     "euroc": {
        #         "MH_01_easy": {"ATE": {"median": 0.013}},
        #         "MH_02_easy": {"ATE": {"median": 0.014}},
        #         "MH_03_medium": {"ATE": {"median": 0.022}},
        #         "MH_04_difficult": {"ATE": {"median": 0.043}},
        #         "MH_05_difficult": {"ATE": {"median": 0.043}},
        #         "V1_01_easy": {"ATE": {"median": 0.037}},
        #         "V1_02_medium": {"ATE": {"median": 0.012}},
        #         "V1_03_difficult": {"ATE": {"median": 0.020}},
        #         "V2_01_easy": {"ATE": {"median": 0.017}},
        #         "V2_02_medium": {"ATE": {"median": 0.013}},
        #         "V2_03_difficult": {"ATE": {"median": 0.014}},
        #     },
        #     "rgbdtum": {
        #         "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.012}},
        #         "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.026}},
        #         "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.111}},
        #         "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.021}},
        #         "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.018}},
        #         "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.042}},
        #         "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.049}},
        #         "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.016}},
        #         "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.048}}
        #     }
        # },
        "orbslam2": {
            # "euroc": {
            #     "MH_01_easy": {"ATE": {"median": 0.071}},
            #     "MH_02_easy": {"ATE": {"median": 0.067}},
            #     "MH_03_medium": {"ATE": {"median": 0.071}},
            #     "MH_04_difficult": {"ATE": {"median": 0.082}},
            #     "MH_05_difficult": {"ATE": {"median": 0.060}},
            #     "V1_01_easy": {"ATE": {"median": 0.015}},
            #     "V1_02_medium": {"ATE": {"median": 0.020}},
            #     "V1_03_difficult": {"ATE": {"median": 0.0}},
            #     "V2_01_easy": {"ATE": {"median": 0.021}},
            #     "V2_02_medium": {"ATE": {"median": 0.018}},
            #     "V2_03_difficult": {"ATE": {"median": 0.0}},
            # },
            "rgbdtum": {
                "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.010}},
                "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.023}},
                "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.071}},
                "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.0}}
            }
        },
        "orbslam2-dev": {
            # "euroc": {
            #     "MH_01_easy": {"ATE": {"median": 0.016}},
            #     "MH_02_easy": {"ATE": {"median": 0.027}},
            #     "MH_03_medium": {"ATE": {"median": 0.028}},
            #     "MH_04_difficult": {"ATE": {"median": 0.138}},
            #     "MH_05_difficult": {"ATE": {"median": 0.072}},
            #     "V1_01_easy": {"ATE": {"median": 0.033}},
            #     "V1_02_medium": {"ATE": {"median": 0.015}},
            #     "V1_03_difficult": {"ATE": {"median": 0.033 }},
            #     "V2_01_easy": {"ATE": {"median": 0.023}},
            #     "V2_02_medium": {"ATE": {"median": 0.029}},
            #     "V2_03_difficult": {"ATE": {"median": 0.0}},
            # },
            "rgbdtum": {
                "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.010}},
                "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.023}},
                "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.071}},
                "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.0}}
            }
        },
        "orbslam3": {
            "euroc": {
                "MH_01_easy": {"ATE": {"median": 0.016}},
                "MH_02_easy": {"ATE": {"median": 0.027}},
                "MH_03_medium": {"ATE": {"median": 0.028}},
                "MH_04_difficult": {"ATE": {"median": 0.138}},
                "MH_05_difficult": {"ATE": {"median": 0.072}},
                "V1_01_easy": {"ATE": {"median": 0.033}},
                "V1_02_medium": {"ATE": {"median": 0.015}},
                "V1_03_difficult": {"ATE": {"median": 0.033 }},
                "V2_01_easy": {"ATE": {"median": 0.023}},
                "V2_02_medium": {"ATE": {"median": 0.029}},
                "V2_03_difficult": {"ATE": {"median": 0.0}},
            },
            "rgbdtum": {
                "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.009}},
                "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.017}},
                "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.210}},
                "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.034}},
                "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.0}}
            }
        },
        "orbslam3-dev": {
            "euroc": {
                "MH_01_easy": {"ATE": {"median": 0.016}},
                "MH_02_easy": {"ATE": {"median": 0.027}},
                "MH_03_medium": {"ATE": {"median": 0.028}},
                "MH_04_difficult": {"ATE": {"median": 0.138}},
                "MH_05_difficult": {"ATE": {"median": 0.072}},
                "V1_01_easy": {"ATE": {"median": 0.033}},
                "V1_02_medium": {"ATE": {"median": 0.015}},
                "V1_03_difficult": {"ATE": {"median": 0.033 }},
                "V2_01_easy": {"ATE": {"median": 0.023}},
                "V2_02_medium": {"ATE": {"median": 0.029}},
                "V2_03_difficult": {"ATE": {"median": 0.0}},
            },
            "rgbdtum": {
                "rgbd_dataset_freiburg1_xyz": {"ATE": {"median": 0.009}},
                "rgbd_dataset_freiburg1_rpy": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_360": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_floor": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_desk": {"ATE": {"median": 0.017}},
                "rgbd_dataset_freiburg1_desk2": {"ATE": {"median": 0.210}},
                "rgbd_dataset_freiburg1_room": {"ATE": {"median": 0.0}},
                "rgbd_dataset_freiburg1_plant": {"ATE": {"median": 0.034}},
                "rgbd_dataset_freiburg1_teddy": {"ATE": {"median": 0.0}}
            }
        },
    }


    def get_median_ate(self, baseline_name: str, dataset_name: str, sequence_name: str) -> float:
        return (self.BM
                .get(baseline_name, {})
                .get(dataset_name, {})
                .get(sequence_name, {})
                .get("ATE", {})
                .get("median", 0.0))
