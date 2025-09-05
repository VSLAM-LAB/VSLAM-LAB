import sys, os, yaml
import shutil
import pandas as pd
import numpy as np
import cv2
import tqdm

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.dataset_utilities import undistort_fisheye
from utilities import downloadFile, decompressFile


class ROVER_dataset(DatasetVSLAMLab):
    DATES = {
        "campus_small": {
            "autumn": "2023-11-23",
            "winter": "2024-02-19",
            "spring": "2024-04-14",
            "summer": "2023-09-11",
            "day": "2024-05-07",
            "dusk": "2024-05-08_1",
            "night": "2024-05-08_2",
            "night-light": "2024-05-24_1",
        },
        "campus_large": {
            "autumn": "2023-11-07",
            "winter": "2024-01-27",
            "spring": "2024-04-14",
            "summer": "2023-07-20",
            "day": "2024-09-25",
            "dusk": "2024-09-24_2",
            "night": "2024-09-24_3",
            "night-light": "2024-09-24_4",
        },
        "garden_small": {
            "autumn": "2023-09-15",
            "winter": "2024-01-13",
            "spring": "2024-04-11",
            "summer": "2023-08-18",
            "day": "2024-05-29_1",
            "dusk": "2024-05-29_2",
            "night": "2024-05-29_3",
            "night-light": "2024-05-29_4",
        },
        "garden_large": {
            "autumn": "2023-12-21",
            "winter": "2024-01-13",
            "spring": "2024-04-11",
            "summer": "2023-08-18",
            "day": "2024-05-29_1",
            "dusk": "2024-05-29_2",
            "night": "2024-05-30_1",
            "night-light": "2024-05-30_2",
        },
        "park": {
            "autumn": "2023-11-07",
            "spring": "2024-04-14",
            "summer": "2023-07-31",
            "day": "2024-05-08",
            "dusk": "2024-05-13_1",
            "night": "2024-05-13_2",
            "night-light": "2024-05-24_2",
            # Note: no winter data for "park"
        },
    }
    
    
    SENSOR_NICKNAMES = {"picam": "pi_cam", "d435i": "realsense_D435i", "t265": "realsense_T265", "vn100": "vn100"}
    
    # persist between get_dataset calls
    seq2group = {}
    
    
    def __init__(self, benchmark_path):
        super().__init__("rover", benchmark_path)
        
        with open(self.yaml_file, "r") as file:
            data = yaml.safe_load(file)
            
        self.url_download_root = data["url_download_root"]
        
        self.sequence_nicknames = self.sequence_names[:]
        
        self.master_calibration_path = os.path.join(self.dataset_path, "calibration")


    def download_sequence_data(self, sequence_name):
        location, setting, sensor, date = self._sequence_data_from_name(sequence_name)
        sequence_group_name = "_".join([location, setting, date])
        resource_name = "_".join([location, date])
        sequence_path = os.path.abspath(os.path.join(self.dataset_path, sequence_name))
        sequence_group_path = os.path.abspath(os.path.join(self.dataset_path, sequence_group_name))
        sequence_subdir = os.path.join(sequence_group_path, self.SENSOR_NICKNAMES[sensor])
        
        def flatten_subdir():
            # $datapath/ROVER/sequence_id_with_date/date -> $datapath/ROVER/sequence_id_with_date
            sequence_group_path_child = os.path.join(sequence_group_path, date)
            if not os.path.isdir(sequence_group_path_child): return
            for item_name in os.listdir(sequence_group_path_child):
                src_item = os.path.join(sequence_group_path_child, item_name)
                dst_item = os.path.join(sequence_group_path, item_name)
                shutil.move(src_item, dst_item)
            os.rmdir(sequence_group_path_child)
            
        self._ensure_data_exists(
            data_path=sequence_group_path,
            target=resource_name, target_path=sequence_group_path,
            callback=flatten_subdir
        )
        
        if not os.path.isdir(sequence_subdir):
            raise FileNotFoundError(f"Source directory for symlink not found after decompression: {sequence_subdir}")
        
        os.symlink(sequence_subdir, sequence_path)
        self.seq2group[sequence_name] = sequence_group_path
               

    def create_calibration_yaml(self, sequence_name):
        self._download_master_calibration_archive()
        _, _, sensor, _ = self._sequence_data_from_name(sequence_name)
        calibration_file = os.path.join(self.master_calibration_path, f"calib_{sensor}.yaml")
        with open(calibration_file, "r") as file:
            data = yaml.safe_load(file)
            
        if sensor == "t265":
            sequence_path = os.path.join(self.dataset_path, sequence_name)
            rgb_path = os.path.join(sequence_path, 'rgb')
            rgb_txt = os.path.join(sequence_path, 'rgb.txt')
            cam_intrinsics_data = data["CamLeft_Intrinsics"]
            fx, fy, cx, cy = cam_intrinsics_data["intrinsics"]
            k1, k2, k3, k4 = cam_intrinsics_data["distortion_coeffs"]
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            distortion_coeffs = np.array([k1, k2, k3, k4])
            fx, fy, cx, cy = self._undistort_fish_pinhole(rgb_txt, camera_matrix, distortion_coeffs, sequence_path)
            self.write_calibration_yaml(
                'PINHOLE',
                float(fx), float(fy),
                float(cx), float(cy),
                0.0, 0.0, 0.0, 0.0, 0.0,
                sequence_name
            )
        else:
            cam_intrinsics_data = data["Cam_Intrinsics"]
            fx, fy, cx, cy = cam_intrinsics_data["intrinsics"]
            k1, k2, p1, p2 = cam_intrinsics_data["distortion_coeffs"]
            k3 = 0.0
            self.write_calibration_yaml(
                "PINHOLE", 
                float(fx), float(fy), 
                float(cx), float(cy), 
                float(k1), float(k2), 
                float(p1), float(p2), float(k3), 
                sequence_name
            )


    def create_rgb_folder(self, sequence_name):
        _, _, sensor, _ = self._sequence_data_from_name(sequence_name)
        if sensor == "t265":
            sequence_path = os.path.join(self.dataset_path, sequence_name)
            rgb_path = os.path.join(sequence_path, "rgb")
            left_rgb = os.path.join(sequence_path, "cam_left")
            if not os.path.exists(rgb_path):
                os.symlink(left_rgb, rgb_path)
        

    def create_rgb_txt(self, sequence_name):
        _, _, sensor, _ = self._sequence_data_from_name(sequence_name)
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # rgbd sensor, see `dataset_rgbdtum.py`
        if sensor == "d435i":
            rgb_original_txt = os.path.join(sequence_path, 'rgb_original.txt')
            rgb_txt = os.path.join(sequence_path, 'rgb.txt')
            depth_txt = os.path.join(sequence_path, 'depth.txt')
            os.rename(rgb_txt, rgb_original_txt)
        
            rgb_df = pd.read_csv(rgb_original_txt, sep=' ', comment='#', header=None, names=['timestamp', 'rgb_filename'])
            depth_df = pd.read_csv(depth_txt, sep=' ', comment='#', header=None, names=['timestamp', 'depth_filename'])

            time_difference_threshold = 0.02 
            def find_closest_depth(rgb_ts):
                diff = abs(depth_df['timestamp'] - rgb_ts)
                min_diff_idx = diff.idxmin()
                if diff[min_diff_idx] <= time_difference_threshold:
                    return depth_df.loc[min_diff_idx, 'depth_filename']
                return None

            rgb_df['matched_depth'] = rgb_df['timestamp'].apply(find_closest_depth)
            matched_pairs = rgb_df.dropna(subset=['matched_depth']).copy()
            matched_pairs['depth_timestamp'] = matched_pairs.apply(
                lambda row: depth_df.loc[depth_df['depth_filename'] == row['matched_depth'], 'timestamp'].values[0],
                axis=1
            )

            matched_pairs['timestamp'] = matched_pairs['timestamp'].apply(lambda x: f"{x:.6f}")
            matched_pairs['depth_timestamp'] = matched_pairs['depth_timestamp'].apply(lambda x: f"{x:.6f}")

            matched_pairs[['timestamp', 'rgb_filename', 'depth_timestamp', 'matched_depth']].to_csv(
                rgb_txt, sep=' ', index=False, header=False
            )
            
        # stereo not currently supported
        elif sensor == "t265":
            cam_left = os.path.join(sequence_path, "cam_left.txt")
            output_path = os.path.join(sequence_path, "rgb.txt")
            shutil.copy(cam_left, output_path)
        elif sensor == "picam":
            pass
    
     
    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        gt_src = os.path.join(self.seq2group[sequence_name], "groundtruth.txt")
        gt_dst_txt = os.path.join(sequence_path, "groundtruth.txt")
        gt_dst_csv = os.path.join(sequence_path, "groundtruth.csv")
        header = "ts,tx,ty,tz,qx,qy,qz,qw\n"
        
        shutil.copy(gt_src, gt_dst_txt)
             
        with open(gt_dst_csv, "w") as dst, open(gt_src, "r") as src:
            dst.write(header)
            for line in src:
                cleaned = line.strip()
                dst.write(",".join(cleaned.split(" ")) + "\n")
    

    def remove_unused_files(self, sequence_name):
        # sequence_path = os.path.join(self.dataset_path, sequence_name)
        # os.remove(os.path.join(sequence_path, 'rgb_original.txt'))
        pass
    
    
    def _ensure_data_exists(self, data_path, target, target_path, callback=None):
        if not os.path.exists(data_path):
            archive_name = target + ".zip"
            download_url = f"{self.url_download_root}/{archive_name}"
            archive_path = os.path.join(self.dataset_path, archive_name)
            if not os.path.exists(archive_path):
                downloadFile(download_url, self.dataset_path)
            decompressFile(archive_path, target_path)
        if callback:
            callback()
            
        
    def _sequence_data_from_name(self, sequence_name):
        location, setting, sensor, date = None, None, None, None
        for location_candidate in self.DATES.keys():
            if sequence_name.startswith(location_candidate):
                location = location_candidate
                break
        else:
            print(f"unknown location in sequence {sequence_name}")

        sequence_name = sequence_name[len(location)+1:]
        setting, sensor = sequence_name.split("_")
        date = self.DATES[location][setting]
        return location, setting, sensor, date

    
    def _undistort_fish_pinhole(self, rgb_txt, camera_matrix, distortion_coeffs, sequence_path, alpha=0.0):
        image_list = []
        with open(rgb_txt, 'r') as file:
            for line in file:
                timestamp, path, *extra = line.strip().split(' ')
                image_list.append(path)

        first = True
        for image_name in tqdm.tqdm(image_list):
            image_path = os.path.join(sequence_path, image_name)
            image = cv2.imread(image_path)
            if first:
                first = False
                h, w = image.shape[:2]
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, distortion_coeffs, (w, h), alpha, (w, h))

            undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs, None, new_camera_matrix)
            cv2.imwrite(image_path, undistorted_image)

        fx, fy, cx, cy = (new_camera_matrix[0, 0], new_camera_matrix[1, 1],
                        new_camera_matrix[0, 2], new_camera_matrix[1, 2])
        return fx, fy, cx, cy
    
    
    def _download_master_calibration_archive(self):
        macosx_path = os.path.join(self.dataset_path, "__MACOSX")
        cleanup_macosx = lambda: shutil.rmtree(macosx_path) if os.path.exists(macosx_path) else None
        self._ensure_data_exists(
            data_path=self.master_calibration_path,
            target="calibration", target_path=self.dataset_path,
            callback=cleanup_macosx)