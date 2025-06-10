# ADD your imports here
from Baselines.baseline_droidslam import DROIDSLAM_baseline
from Baselines.baseline_droidslam import DROIDSLAM_baseline_dev
from Baselines.baseline_mast3rslam import MAST3RSLAM_baseline
from Baselines.baseline_mast3rslam import MAST3RSLAM_baseline_dev
from Baselines.baseline_orbslam2 import ORBSLAM2_baseline
from Baselines.baseline_orbslam2 import ORBSLAM2_baseline_dev
from Baselines.baseline_dpvo import DPVO_baseline
from Baselines.baseline_dpvo import DPVO_baseline_dev
from Baselines.baseline_monogs import MONOGS_baseline
from Baselines.baseline_monogs import MONOGS_baseline_dev
from Baselines.baseline_colmap import COLMAP_baseline
from Baselines.baseline_glomap import GLOMAP_baseline
from Baselines.baseline_gensfm import GENSFM_baseline_dev
from Baselines.baseline_mast3r import MAST3R_baseline_dev
from Baselines.baseline_vggt import VGGT_baseline_dev

from Baselines.baseline_orbslam3 import ORBSLAM3_baseline_dev
from Baselines.baseline_okvis2 import OKVIS2_baseline_dev

def get_baseline_switcher():
    return {
        "droidslam": lambda: DROIDSLAM_baseline(),
        "droidslam-dev": lambda: DROIDSLAM_baseline_dev(),
        "mast3rslam": lambda: MAST3RSLAM_baseline(),
        "mast3rslam-dev": lambda: MAST3RSLAM_baseline_dev(),
        "orbslam2": lambda: ORBSLAM2_baseline(),
        "orbslam2-dev": lambda: ORBSLAM2_baseline_dev(),  
        "dpvo": lambda: DPVO_baseline(),
        "dpvo-dev": lambda: DPVO_baseline_dev(),
        "monogs": lambda: MONOGS_baseline(),
        "monogs-dev": lambda: MONOGS_baseline_dev(),
        "colmap": lambda: COLMAP_baseline(),
        "glomap": lambda: GLOMAP_baseline(),
        "gensfm-dev": lambda: GENSFM_baseline_dev(),
        "mast3r-dev": lambda: MAST3R_baseline_dev(),
        "vggt-dev": lambda: VGGT_baseline_dev(),
        "orbslam3-dev": lambda: ORBSLAM3_baseline_dev(),
        "okvis2-dev": lambda: OKVIS2_baseline_dev(),
    }

def get_baseline(baseline_name):
    baseline_name = baseline_name.lower()
    switcher = get_baseline_switcher()
    return switcher.get(baseline_name, lambda: "Invalid case")()

def list_available_baselines():
    return list(get_baseline_switcher().keys())