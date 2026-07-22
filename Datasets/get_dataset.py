import os
from pathlib import Path
from path_constants import VSLAM_LAB_DIR

# ADD your imports here

# Monocular datasets
from Datasets.dataset_files.dataset_tartanair import TARTANAIR_dataset
from Datasets.dataset_files.dataset_squidle import SESOKO_dataset
from Datasets.dataset_files.dataset_sweetcorals import SWEETCORALS_dataset
from Datasets.dataset_files.dataset_soneva import SONEVA_dataset
from Datasets.dataset_files.dataset_monotum import MONOTUM_dataset
from Datasets.dataset_files.dataset_hamlyn import HAMLYN_dataset
from Datasets.dataset_files.dataset_scannetplusplus import SCANNETPLUSPLUS_dataset
from Datasets.dataset_files.dataset_caves import CAVES_dataset

# RGBD datasets
from Datasets.dataset_files.dataset_eth import ETH_dataset
from Datasets.dataset_files.dataset_rgbdtum import RGBDTUM_dataset
from Datasets.dataset_files.dataset_replica import REPLICA_dataset
from Datasets.dataset_files.dataset_nuim import NUIM_dataset
from Datasets.dataset_files.dataset_7scenes import SEVENSCENES_dataset
from Datasets.dataset_files.dataset_drunkards import DRUNKARDS_dataset

# Stereo datasets
from Datasets.dataset_files.dataset_kitti import KITTI_dataset
from Datasets.dataset_files.dataset_ut_coda import UT_CODA_dataset
from Datasets.dataset_files.dataset_ariel import ARIEL_dataset

# Stereo-VI datasets
from Datasets.dataset_files.dataset_euroc import EUROC_dataset
from Datasets.dataset_files.dataset_rover import ROVER_t265_dataset
from Datasets.dataset_files.dataset_rover import ROVER_d435i_dataset
from Datasets.dataset_files.dataset_rover import ROVER_picam_dataset
from Datasets.dataset_files.dataset_s3li import S3LI_dataset
from Datasets.dataset_files.dataset_msd import MSD_dataset
from Datasets.dataset_files.dataset_openloris import OPENLORIS_d400_dataset
from Datasets.dataset_files.dataset_openloris import OPENLORIS_t265_dataset
from Datasets.dataset_files.dataset_blt import BLT_dataset
from Datasets.dataset_files.dataset_hilti2022 import HILTI2022_dataset
from Datasets.dataset_files.dataset_madmax import MADMAX_dataset
from Datasets.dataset_files.dataset_hilti2026 import HILTI2026_dataset
from Datasets.dataset_files.dataset_vitum import VITUM_dataset

# Development
from Datasets.dataset_files.dataset_videos import VIDEOS_dataset
from Datasets.dataset_files.dataset_iphone import IPHONE_dataset
from Datasets.dataset_files.dataset_youtube import YOUTUBE_dataset
from Datasets.dataset_files.dataset_strayscanner import StrayScanner_dataset

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

def get_dataset(dataset_name, benchmark_path):
    dataset_name = dataset_name.lower()
    switcher = {
        # ADD your datasets here
        "tartanair": lambda: TARTANAIR_dataset(benchmark_path),
        "eth": lambda: ETH_dataset(benchmark_path),
        "rgbdtum": lambda: RGBDTUM_dataset(benchmark_path),
        "replica": lambda: REPLICA_dataset(benchmark_path),
        "nuim": lambda: NUIM_dataset(benchmark_path),
        "kitti": lambda: KITTI_dataset(benchmark_path),
        "ut-coda": lambda: UT_CODA_dataset(benchmark_path),
        "euroc": lambda: EUROC_dataset(benchmark_path),
        "rover-t265": lambda: ROVER_t265_dataset(benchmark_path),
        "rover-d435i": lambda: ROVER_d435i_dataset(benchmark_path),
        "rover-picam": lambda: ROVER_picam_dataset(benchmark_path),
        "s3li": lambda: S3LI_dataset(benchmark_path),
        "msd": lambda: MSD_dataset(benchmark_path),
        "sesoko": lambda: SESOKO_dataset(benchmark_path),
        "7scenes": lambda: SEVENSCENES_dataset(benchmark_path),
        "openloris-d400": lambda: OPENLORIS_d400_dataset(benchmark_path),
        "openloris-t265": lambda: OPENLORIS_t265_dataset(benchmark_path),
        "blt": lambda: BLT_dataset(benchmark_path),
        "sweetcorals": lambda: SWEETCORALS_dataset(benchmark_path),
        "soneva": lambda: SONEVA_dataset(benchmark_path),
        "monotum": lambda: MONOTUM_dataset(benchmark_path),
        "ariel": lambda: ARIEL_dataset(benchmark_path),
        "hilti2022": lambda: HILTI2022_dataset(benchmark_path),
        "madmax": lambda: MADMAX_dataset(benchmark_path),
        "hamlyn": lambda: HAMLYN_dataset(benchmark_path),
        "hilti2026": lambda: HILTI2026_dataset(benchmark_path),
        "drunkards": lambda: DRUNKARDS_dataset(benchmark_path),
        "scannetplusplus": lambda: SCANNETPLUSPLUS_dataset(benchmark_path),
        "caves": lambda: CAVES_dataset(benchmark_path),
        "vitum": lambda: VITUM_dataset(benchmark_path),

        # Development
        "videos": lambda: VIDEOS_dataset(benchmark_path),
        "iphone": lambda: IPHONE_dataset(benchmark_path),
        "youtube": lambda: YOUTUBE_dataset(benchmark_path),
        "strayscanner": lambda: StrayScanner_dataset(benchmark_path)
    }

    return switcher.get(dataset_name, lambda: "Invalid case")()

def list_available_datasets() -> list[str]:
    dataset_scripts_path = VSLAM_LAB_DIR /  'Datasets' / 'dataset_files'
    dataset_scripts = []
    for filename in os.listdir(dataset_scripts_path):
        if (
            'dataset_' in filename
            and filename.endswith('.yaml')
            and 'utilities' not in filename
            and '_calibration' not in filename
        ):
            dataset_scripts.append(filename)

    dataset_scripts = [item.replace('dataset_', '').replace('.yaml', '') for item in dataset_scripts]

    return dataset_scripts
