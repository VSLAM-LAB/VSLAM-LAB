import os
import yaml
import shutil
import glob

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics


class EUROC_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
    
        # Initialize the dataset
        super().__init__('euroc', benchmark_path)
        
        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            
        # Get download url
        self.url_download_root = data['url_download_root']
        
        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):
 
        # Variables
        compressed_name = sequence_name
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = sequence_name
        if 'MH' in sequence_name:
            download_url = os.path.join(self.url_download_root, 'machine_hall', sequence_name, compressed_name_ext)
        if 'V1' in sequence_name:
            download_url = os.path.join(self.url_download_root, 'vicon_room1', sequence_name, compressed_name_ext)
        if 'V2' in sequence_name:
            download_url = os.path.join(self.url_download_root, 'vicon_room2', sequence_name, compressed_name_ext)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)      
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)
        
        # Download the compressed file
        if not os.path.exists(compressed_file):   
            downloadFile(download_url, self.dataset_path)
        
        # Decompress the file
        if os.path.exists(decompressed_folder): 
            shutil.rmtree(decompressed_folder)    
        sequence_path = os.path.join(self.dataset_path, sequence_name)           
        decompressFile(compressed_file, sequence_path)
        
        # Delete the compressed file
        if os.path.exists(compressed_file):  
            os.remove(compressed_file)  
        
        # Download groundtruth from TUM repository
        if not os.path.exists(os.path.join(self.dataset_path, 'supp_v2')):
            compressed_name = 'supp_v2'
            compressed_name_ext = compressed_name + '.zip'
            decompressed_name = compressed_name    
            
            compressed_file = os.path.join(self.dataset_path, compressed_name_ext)      
            decompressed_folder = os.path.join(self.dataset_path, decompressed_name)
            
            download_url = os.path.join('https://cvg.cit.tum.de/mono', compressed_name_ext)
            if not os.path.exists(os.path.join(self.dataset_path, compressed_name_ext)):      
                downloadFile(download_url, self.dataset_path)
                           
            if os.path.exists(decompressed_folder):
                shutil.rmtree(decompressed_folder)  
                   
            decompressFile(compressed_file, os.path.join(self.dataset_path, decompressed_name))
            if os.path.exists(compressed_file):  
                os.remove(compressed_file)  
                                                    
    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path, exist_ok=True)
            rgb_files = glob.glob(os.path.join(sequence_path, 'mav0', 'cam0', 'data', '*.png'))
            for png_path in rgb_files:
                rgb_name = os.path.basename(png_path)
                shutil.copy(png_path, os.path.join(rgb_path, rgb_name))   
       
    def create_rgb_txt(self, sequence_name):        
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):             
                name, ext = os.path.splitext(filename)
                ts = float(name) / 10e8
                file.write(f"{ts:.5f} rgb/{filename}\n") 
    def create_imu_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path, exist_ok=True)
            rgb_files = glob.glob(os.path.join(sequence_path, 'mav0', 'cam0', 'data', '*.png'))
            for png_path in rgb_files:
                rgb_name = os.path.basename(png_path)
                shutil.copy(png_path, os.path.join(rgb_path, rgb_name))   
       
    def create_imu_txt(self, sequence_name):        
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        imu_path = os.path.join(sequence_path, 'imu')
        imu_txt = os.path.join(sequence_path, 'imu.txt')
        
        imu_files = [f for f in os.listdir(imu_path) if os.path.isfile(os.path.join(imu_path, f))]
        imu_files.sort()
        with open(imu_txt, 'w') as file:
            for iIMU, filename in enumerate(imu_files, start=0):             
                name, ext = os.path.splitext(filename)
                ts = float(name) / 10e8
                file.write(f"{ts:.5f} imu/{filename}\n") 

    def create_calibration_yaml(self, sequence_name):

        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_file_yaml = os.path.join(sequence_path, 'mav0', 'cam0', 'sensor.yaml')
               
        # Load calibration from .yaml file
        with open(calibration_file_yaml, 'r') as file:
            data = yaml.safe_load(file)
        
        intrinsics = data['intrinsics']
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        distortion = data['distortion_coefficients']
        k1, k2, p1, p2, k3 = distortion[0], distortion[1], distortion[2], distortion[3], 0.0
        
        self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)
            
    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        
        with open(os.path.join(self.dataset_path, 'supp_v2', 'gtFiles', f'mav_{sequence_name}.txt')) as source_file:
            with open(groundtruth_txt, 'w') as destination_file:
                for line in source_file:
                    if 'NaN' not in line:
                        destination_file.write(line)
                  
    def remove_unused_files(self, sequence_name):
        sequence_folder = os.path.join(self.dataset_path, sequence_name, 'mav0')
        if os.path.exists(sequence_folder):
            shutil.rmtree(sequence_folder)
        
        sequence_folder = os.path.join(self.dataset_path, sequence_name, '__MACOSX')
        if os.path.exists(sequence_folder):  
            shutil.rmtree(sequence_folder)