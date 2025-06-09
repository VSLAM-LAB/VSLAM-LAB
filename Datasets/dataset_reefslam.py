import os, yaml
import cv2
import numpy as np

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

class REEFSLAM_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name = 'reefslam'):
        # Initialize the dataset
        super().__init__(dataset_name, benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.videos_path = data['videos_path']

        # Create sequence_nicknames
        self.sequence_nicknames = self.sequence_names

        # Get resolution size
        self.resolution_size = data['resolution_size']

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        for filename in os.listdir(self.videos_path):
            if sequence_name in filename :
                video_path = os.path.join(self.videos_path, filename)
                break
                
        if not os.path.exists(rgb_path):
            self.extract_png_frames(video_path=video_path, output_dir=rgb_path)  # extract at 30Hz

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                file.write(f"{ts:.5f} rgb/{filename}\n")
   
    def create_calibration_yaml(self, sequence_name):
        
        fx, fy, cx, cy = 390.29, 390.29, 369.5, 207.5
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        with open(groundtruth_txt, 'w') as f:
            pass
             
    def extract_png_frames(self, video_path, output_dir="rgb"):
        """
        Extract frames from a video based on a frequency in Hertz (frames per second) and save as PNG images.
        Also creates an rgb.txt file with timestamps and image paths.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save the PNG files.
            frequency_hz (int or float): How many frames to save per second of video.
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Failed to get FPS from video.")

        frame_interval = int(round(fps / self.rgb_hz))

        print(f"Video opened: {video_path}")
        print(f"Video FPS: {fps:.2f}")
        print(f"Extracting {30} frames per second (every {frame_interval} frames).")

        frame_idx = 0
        saved_idx = 0
        timestamp_list = []

        estimate_new_resolution = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Compute timestamp
                timestamp_sec = frame_idx / fps

                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if estimate_new_resolution:
                    rgb_frame_height, rgb_frame_width = rgb_frame.shape[:2]
                    scaled_height = np.sqrt(self.resolution_size[0] * self.resolution_size[1] * rgb_frame_height / rgb_frame_width)
                    scaled_width = self.resolution_size[0] * self.resolution_size[1] / scaled_height
                    scaled_height = int(scaled_height)
                    scaled_width = int(scaled_width)
                    estimate_new_resolution = False
                
                resized_img = cv2.resize(rgb_frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LANCZOS4)

                # Save as PNG with 5-digit padded integer filename
                filename = os.path.join(output_dir, f"{saved_idx:05d}.png")
                cv2.imwrite(filename, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

                # Save timestamp and image path
                image_relative_path = os.path.join(output_dir, f"{saved_idx:05d}.png")
                timestamp_list.append((timestamp_sec, image_relative_path))

                saved_idx += 1

            frame_idx += 1

        cap.release()