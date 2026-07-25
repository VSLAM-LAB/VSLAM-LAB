import os
from pathlib import Path
from path_constants import VSLAM_LAB_DIR

# ADD your imports here

# Monocular datasets
from Datasets.dataset_files.dataset_tartanair import TartanairDataset
from Datasets.dataset_files.dataset_squidle import SesokoDataset
from Datasets.dataset_files.dataset_sweetcorals import SweetcoralsDataset
from Datasets.dataset_files.dataset_soneva import SonevaDataset
from Datasets.dataset_files.dataset_monotum import MonotumDataset
from Datasets.dataset_files.dataset_hamlyn import HamlynDataset
from Datasets.dataset_files.dataset_scannetplusplus import ScannetplusplusDataset
from Datasets.dataset_files.dataset_caves import CavesDataset
from Datasets.dataset_files.dataset_eiffel_tower import EiffelTowerDataset

# RGBD datasets
from Datasets.dataset_files.dataset_eth import EthDataset
from Datasets.dataset_files.dataset_rgbdtum import RgbdtumDataset
from Datasets.dataset_files.dataset_replica import ReplicaDataset
from Datasets.dataset_files.dataset_nuim import NuimDataset
from Datasets.dataset_files.dataset_7scenes import SevenscenesDataset
from Datasets.dataset_files.dataset_drunkards import DrunkardsDataset

# Stereo datasets
from Datasets.dataset_files.dataset_kitti import KittiDataset
from Datasets.dataset_files.dataset_ut_coda import UtCodaDataset
from Datasets.dataset_files.dataset_ariel import ArielDataset

# Stereo-VI datasets
from Datasets.dataset_files.dataset_euroc import EurocDataset
from Datasets.dataset_files.dataset_rover import RoverT265Dataset
from Datasets.dataset_files.dataset_rover import RoverD435iDataset
from Datasets.dataset_files.dataset_rover import RoverPicamDataset
from Datasets.dataset_files.dataset_s3li import S3liDataset
from Datasets.dataset_files.dataset_msd import MsdDataset
from Datasets.dataset_files.dataset_openloris import OpenlorisD400Dataset
from Datasets.dataset_files.dataset_openloris import OpenlorisT265Dataset
from Datasets.dataset_files.dataset_hilti2022 import Hilti2022Dataset
from Datasets.dataset_files.dataset_madmax import MadmaxDataset
from Datasets.dataset_files.dataset_hilti2026 import Hilti2026Dataset
from Datasets.dataset_files.dataset_vitum import VitumDataset

# Development
from Datasets.dataset_files.dataset_videos import VideosDataset
from Datasets.dataset_files.dataset_iphone import IphoneDataset
from Datasets.dataset_files.dataset_youtube import YoutubeDataset
from Datasets.dataset_files.dataset_strayscanner import StrayscannerDataset

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

def get_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    switcher = {
        # ADD your datasets here
        "tartanair": lambda: TartanairDataset(),
        "eth": lambda: EthDataset(),
        "rgbdtum": lambda: RgbdtumDataset(),
        "replica": lambda: ReplicaDataset(),
        "nuim": lambda: NuimDataset(),
        "kitti": lambda: KittiDataset(),
        "ut-coda": lambda: UtCodaDataset(),
        "euroc": lambda: EurocDataset(),
        "rover-t265": lambda: RoverT265Dataset(),
        "rover-d435i": lambda: RoverD435iDataset(),
        "rover-picam": lambda: RoverPicamDataset(),
        "s3li": lambda: S3liDataset(),
        "msd": lambda: MsdDataset(),
        "sesoko": lambda: SesokoDataset(),
        "7scenes": lambda: SevenscenesDataset(),
        "openloris-d400": lambda: OpenlorisD400Dataset(),
        "openloris-t265": lambda: OpenlorisT265Dataset(),
        "sweetcorals": lambda: SweetcoralsDataset(),
        "soneva": lambda: SonevaDataset(),
        "monotum": lambda: MonotumDataset(),
        "ariel": lambda: ArielDataset(),
        "hilti2022": lambda: Hilti2022Dataset(),
        "madmax": lambda: MadmaxDataset(),
        "hamlyn": lambda: HamlynDataset(),
        "hilti2026": lambda: Hilti2026Dataset(),
        "drunkards": lambda: DrunkardsDataset(),
        "scannetplusplus": lambda: ScannetplusplusDataset(),
        "caves": lambda: CavesDataset(),
        "vitum": lambda: VitumDataset(),
        "eiffel-tower": lambda: EiffelTowerDataset(),

        # Development
        "videos": lambda: VideosDataset(),
        "iphone": lambda: IphoneDataset(),
        "youtube": lambda: YoutubeDataset(),
        "strayscanner": lambda: StrayscannerDataset()
    }

    return switcher.get(dataset_name, lambda: "Invalid case")()

def list_available_datasets() -> list[str]:
    dataset_scripts_path = VSLAM_LAB_DIR /  'Datasets' / 'dataset_files'
    dataset_scripts = []
    for filename in os.listdir(dataset_scripts_path):
        if 'dataset_' in filename and filename.endswith('.yaml') and 'utilities' not in filename:
            dataset_scripts.append(filename)

    dataset_scripts = [item.replace('dataset_', '').replace('.yaml', '') for item in dataset_scripts]

    return dataset_scripts
