## List of Known Dependencies
### VSLAM-LAB

In this document we list all the pieces of code included  by VSLAM-LAB and linked libraries which are not property of the authors of VSLAM-LAB.

##### Code in ORB-SLAM2
Please follow the license of ORB-SLAM2.

License URL: https://github.com/raulmur/ORB_SLAM2/blob/master/LICENSE.txt

##### Code in DSO
Please follow the license of 

Licence URL: https://github.com/JakobEngel/dso/blob/master/LICENSE

##### Datasets
Please follow the licenses of each dataset. We provide a spreadsheet with citation for each dataset and VSLAM system for your convenience:
https://docs.google.com/spreadsheets/d/1V8_TLqlccipJ6x_TXkgLsw9zWszHU9M-0mGgDT92TEs/edit?usp=drive_link 

##### Other tools
The processed ground truth for the EuRoC dataset is obtained from the supplementary material of the 'Monocular Visual Odometry Dataset', 
which can be accessed at https://cvg.cit.tum.de/data/datasets/mono-dataset

The images of the 'Monocular Visual Odometry Dataset' (monotum) are undistorted using the undistortion methods implemented in the following repository: 
https://github.com/tum-vision/mono_dataset_code.git

The aligning functions in align_trajectories.py are a modified version of the code in 'Useful tools for the RGB-D benchmark':
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools#evaluation
