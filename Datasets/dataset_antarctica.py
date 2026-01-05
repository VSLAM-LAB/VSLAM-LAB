
from pathlib import Path
import csv

import piexif
import numpy as np
from PIL import Image
from tqdm import tqdm
import os, yaml, shutil
from datetime import datetime
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.DatasetVSLAMLab_utilities import undistort_rgb_rad_tan, resize_rgb_images

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class ANTARCTICA_dataset(DatasetVSLAMLab):
    """ANTARCTICA dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "antarctica") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Path to the raw dataset folder
        self.raw_data_folder = cfg["folder_with_raw_data"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Get resolution size
        self.resolution_size = cfg['resolution_size']

    def download_sequence_data(self, sequence_name: str) -> None:
        source_rgb_path = self.get_source_rgb_path(sequence_name)

        # Fix known issues with filenames
        if sequence_name == 'Q3_DJI_202401111932_002':
            if os.path.exists(os.path.join(source_rgb_path,'DJI_20240111193348_0001.JPG')):
                os.rename(os.path.join(source_rgb_path,'DJI_20240111193348_0001.JPG'), os.path.join(source_rgb_path,'DJI_20240111193348_0001.JPG.GENERAL_VIEW'))
        
        if sequence_name == 'Robbos':
            files_to_fix = ["DSC09166.JPG","DSC09167.JPG","DSC09168.JPG","DSC09169.JPG","DSC09170.JPG","DSC09171.JPG",]
            for fname in files_to_fix:
                original_name = os.path.join(source_rgb_path, fname)
                modified_name = os.path.join(source_rgb_path, fname + '.WRONG')
                if not os.path.exists(modified_name):
                    os.rename(original_name, modified_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        if rgb_path.exists() and any(rgb_path.iterdir()):
            return

        os.makedirs(rgb_path, exist_ok=True)  

        source_rgb_path = self.get_source_rgb_path(sequence_name)
       
        for file in tqdm(sorted(os.listdir(source_rgb_path)), desc='Copying images'):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                shutil.copy2(source_rgb_path / file, rgb_path / file)
               
    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        source_rgb_path = self.get_source_rgb_path(sequence_name)
        tmp = rgb_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb0 (s)", "path_rgb0"])
            for file in sorted(os.listdir(source_rgb_path)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    timestamp_seconds = self.extract_timestamp_from_filename(source_rgb_path / file)
                    formatted_timestamp = f"{timestamp_seconds:.3f}"         
                    w.writerow([formatted_timestamp, f"rgb_0/{file}"])
        tmp.replace(rgb_csv)
     
    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = self.get_calibration(sequence_name)
        
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion_coeffs = np.array([k1, k2, p1, p2, k3])
        
        fx, fy, cx, cy = resize_rgb_images(rgb_csv, sequence_path, self.resolution_size[0], self.resolution_size[1], camera_matrix)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        fx, fy, cx, cy = undistort_rgb_rad_tan(rgb_csv, sequence_path, camera_matrix, distortion_coeffs)

        camera0 = {"model": "Pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0)

    def get_utm_transformer(self, lat, lon):
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        proj_str = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        utm_crs = CRS.from_proj4(proj_str)
        return Transformer.from_crs("epsg:4326", utm_crs, always_xy=True)

    def extract_timestamp_from_filename(self, filename: str | Path) -> int:
        with Image.open(filename) as img:
            exif_data = img.info.get("exif")
            exif_dict = piexif.load(exif_data)
            timestamp_raw = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
            timestamp_str = timestamp_raw.decode("utf-8") 
            dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
            timestamp_seconds = int(dt.timestamp())
        return timestamp_seconds

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_csv = sequence_path / "groundtruth.csv"
        source_rgb_path = self.get_source_rgb_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts","tx","ty","tz","qx","qy","qz","qw"])
            for fname in sorted(os.listdir(source_rgb_path)):
                if not fname.lower().endswith(".jpg"):
                    continue
                fpath = os.path.join(source_rgb_path, fname)
                ts = self.extract_timestamp_from_filename(fpath)
                gps_data = self.get_gps_from_exif(fpath)
                if gps_data == None:
                    continue
                lat, lon, alt = gps_data

                transformer = self.get_utm_transformer(lat, lon)
                x, y = transformer.transform(lon, lat)
                z = alt
                qx = qy = qz = 0.0
                qw = 1.0
                w.writerow([f"{ts:.3f}" , f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{qx:.3f}", f"{qy:.3f}", f"{qz:.3f}", f"{qw:.3f}"])
        tmp.replace(groundtruth_csv)
     
    def dms_to_decimal(self, dms, ref):
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal

    def get_gps_from_exif(self, image_path: str | Path) -> tuple | None:
        img = Image.open(image_path)
        exif_data = img.info.get("exif")
        if not exif_data:
            return None
        exif_dict = piexif.load(exif_data)
        gps = exif_dict.get("GPS")
        if not gps:
            return None

        latitude = self.dms_to_decimal(gps[piexif.GPSIFD.GPSLatitude], gps[piexif.GPSIFD.GPSLatitudeRef].decode())
        longitude = self.dms_to_decimal(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef].decode())

        altitude = gps.get(piexif.GPSIFD.GPSAltitude)
        if altitude:
            altitude = altitude[0] / altitude[1]
        else:
            altitude = None

        return latitude, longitude, altitude
    
    def get_source_rgb_path(self, sequence_name: str) -> Path:
        if sequence_name == 'ASPA135':
            source_rgb_path = os.path.join(self.raw_data_folder, 'Sites', sequence_name, '2023-02-01_ASPA135_UAS-mapping', 'raw')
        if sequence_name == 'ASPA136':
            source_rgb_path = os.path.join(self.raw_data_folder, 'Sites', sequence_name, '2023-01-31_ASPA136_scouting-SC-rocky-area', 'raw_images')      
        if sequence_name == 'Robbos':
            source_rgb_path = os.path.join(self.raw_data_folder, 'Sites', sequence_name, '230114_F1')   
        if 'Q3_DJI_' in sequence_name:
            source_rgb_path = os.path.join(self.raw_data_folder, 'Sites', 'Bunger-Hills_ROI_01', 'Q3_2024-01-11_P1', 
                                           sequence_name.replace('Q3_DJI_', 'DJI_'))    
        if sequence_name == 'Q1_2023-12-26_P1':
            source_rgb_path = os.path.join(self.raw_data_folder, 'Sites', 'Bunger-Hills_ROI_01', 'Q1_2023-12-26_P1', 'raw_photos') 
        if 'Q2_DJI_' in sequence_name:
            source_rgb_path = os.path.join(self.raw_data_folder, 'Sites', 'Bunger-Hills_ROI_01', 'Q2_2024-01-12_P1', 
                                           sequence_name.replace('Q2_DJI_', 'DJI_'))    
        return Path(source_rgb_path)
    
    def get_calibration(self, sequence_name: str) -> tuple:
        if sequence_name == 'ASPA135':
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                3596.02, 3596.02, (4032/2) + 24.5481, (3024/2) -37.3991 , 0.19876, -0.578379, 0.000520889, -0.000552955, 0.678218)
        if sequence_name == 'ASPA136':
             fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                3647.53, 3647.53, (4032/2) + 47.8637, (3024/2) -28.4652 , 0.195633, -0.628398, 0.000614648, -0.000380725, 0.743683)
        if sequence_name == 'Robbos':
             fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                4000.96, 3647.53, (6000/2) -28.8893, (4000/2) -28.2806 , -0.0677109, 0.0901919, 0.000729385, 0.00105102, 0.00261822)
        if 'Q3_DJI_' in sequence_name:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                8154.29, 8154.29, (8192/2) -19.2094, (5460/2) +58.075 , -0.0331103, 0.0312571, -0.00133488, 0.00277011, -0.109366) 
        if sequence_name == 'Q1_2023-12-26_P1':
             fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                8188.58, 8188.58, (8192/2) -22.4518, (5460/2) +50.8968 , -0.0321469, 0.0196234, -0.00130358, 0.00279268, -0.0951689)
        if 'Q2_DJI_' in sequence_name:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                8102.23, 8102.23, (8192/2) -19.6184, (5460/2) +56.6874 , -0.0334521, 0.0433995, -0.00137348, 0.00279438, -0.12867) 
        return fx, fy, cx, cy, k1, k2, p1, p2, k3