import os
import yaml
import shutil
import subprocess

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import downloadFile
from utilities import decompressFile

from utilities import replace_string_in_files
from path_constants import VSLAM_LAB_DIR


class MonotumDataset(DatasetVSLAMLAB):
    """MONOTUM dataset helper for VSLAM-LAB benchmark."""
    
    def __init__(self, dataset_name: str = "monotum") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root = cfg['url_download_root']

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('sequence_', 'seq ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        if sequence_path.exists():
            return
        
        # Get monotum dataset code to undistort images
        # self.get_mono_dataset_code()

        # Variables
        compressed_name = sequence_name
        compressed_name_ext = compressed_name + '.zip'
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = self.dataset_path / compressed_name_ext

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        decompressFile(compressed_file, self.dataset_path)


    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        rgb_path.mkdir(parents=True, exist_ok=True)


        command = f"pixi run -e monodataset undistort {os.path.join(sequence_path, '')} {sequence_path}"
        subprocess.run(command, shell=True)

        os.remove(os.path.join(sequence_path, 'images.zip'))

    def create_rgb_csv(self, sequence_name):
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        times = []
        times_txt = os.path.join(sequence_path, 'times.txt')
        with open(times_txt, 'r') as file:
            for line in file:
                columns = line.split()
                if columns:
                    times.append(columns[1])

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                ts = float(times[iRGB])
                file.write(f'{ts:.5f} rgb/{filename}\n')

    def create_calibration_yaml(self, sequence_name):

        sequence_path = self.sequence_path(sequence_name)
        calibration_txt = os.path.join(sequence_path, 'calibration.txt')
        with open(calibration_txt, 'r') as file:
            calibration = [value for value in file.readline().split()]

        fx, fy, cx, cy = calibration[0], calibration[1], calibration[2], calibration[3]
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_csv(self, sequence_name):
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')

        with open(os.path.join(sequence_path, 'groundtruthSync.txt')) as source_file:
            with open(groundtruth_txt, 'w') as destination_file:
                for line in source_file:
                    if 'NaN' not in line:
                        destination_file.write(line)

    def remove_unused_files(self, sequence_name):
        sequence_path = self.sequence_path(sequence_name)
        #os.remove(os.path.join(sequence_path, 'calibration.txt'))
        #os.remove(os.path.join(sequence_path, 'camera.txt'))
        #os.remove(os.path.join(sequence_path, 'pcalib.txt'))
        #os.remove(os.path.join(sequence_path, 'statistics.txt'))
        #os.remove(os.path.join(sequence_path, 'times.txt'))
        #os.remove(os.path.join(sequence_path, 'vignette.png'))
        #os.remove(os.path.join(sequence_path, 'groundtruthSync.txt'))

    def get_mono_dataset_code(self):

        # Clone and compile "https://github.com/tum-vision/mono_dataset_code.git"
        self.mono_dataset_code_directory = os.path.join(VSLAM_LAB_DIR, 'Baselines', 'mono_dataset_code')

        if not os.path.exists(os.path.join(self.mono_dataset_code_directory, 'bin', 'playbackDataset')):

            command = f"pixi run -e monodataset git-clone"
            subprocess.run(command, shell=True)

            replace_string_in_files(self.mono_dataset_code_directory, 'CV_LOAD_IMAGE_UNCHANGED', 'cv::IMREAD_UNCHANGED')
            replace_string_in_files(self.mono_dataset_code_directory, 'CV_LOAD_IMAGE_GRAYSCALE', 'cv::IMREAD_GRAYSCALE')

            CMakeLists_txt = os.path.join(self.mono_dataset_code_directory, 'CMakeLists.txt')
            CMakeLists_txt_new = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles', 'CMakeLists.txt')
            os.remove(CMakeLists_txt)
            shutil.copy(CMakeLists_txt_new, CMakeLists_txt)

            main_playbackDataset_cpp = os.path.join(self.mono_dataset_code_directory, 'src', 'main_playbackDataset.cpp')
            main_playbackDataset_cpp_new = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles',
                                                        'main_playbackDataset.cpp')
            os.remove(main_playbackDataset_cpp)
            shutil.copy(main_playbackDataset_cpp_new, main_playbackDataset_cpp)

            build_sh = os.path.join(self.mono_dataset_code_directory, 'build.sh')
            build_sh_new = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles', 'build.sh')
            shutil.copy(build_sh_new, build_sh)

            command = f"pixi run -e monodataset build"
            subprocess.run(command, shell=True)

        else:
            print('[dataset_monotum.py] \'' + self.mono_dataset_code_directory + '\' already built')
