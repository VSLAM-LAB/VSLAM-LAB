#!/bin/bash
    
VSLAM_LAB_DIR="/home/alejandro/VSLAM-LAB-NEXT-ITERATION/VSLAM-LAB"
VSLAM_LAB_BENCHMARK_DIR="/home/alejandro/VSLAM-LAB-NEXT-ITERATION/VSLAM-LAB-Benchmark"

CACHE_DIR=/home/alejandro/.cache

echo "Resetting VSLAM-LAB directory: $VSLAM_LAB_DIR"

echo "Removing __pycache__ directories..."
find "$VSLAM_LAB_DIR" -type d -name "__pycache__" -exec rm -rf {} +

echo "Removing pixi envs: $VSLAM_LAB_DIR/.pixi/envs* and $VSLAM_LAB_DIR/.pixi/task-cache-v0*"
rm -rf "$VSLAM_LAB_DIR"/.pixi/envs*
rm -rf "$VSLAM_LAB_DIR"/.pixi/task-cache-v0*

echo "Removing pixi.lock: $VSLAM_LAB_DIR/pixi.lock*"
rm -rf "$VSLAM_LAB_DIR"/pixi.lock*

echo "Removing system .cache directory: $CACHE_DIR"
rm -rf "$CACHE_DIR"

echo "Removing Baselines and Benchmark datasets..."
BASELINES_DIR="$VSLAM_LAB_DIR/Baselines"
rm -rf "$BASELINES_DIR"/*-DEV*
rm -rf "$BASELINES_DIR"/*git_clone_*
rm -rf "$BASELINES_DIR"/AnyFeature-VSLAM
rm -rf "$BASELINES_DIR"/colmap
rm -rf "$BASELINES_DIR"/DPVO
rm -rf "$BASELINES_DIR"/DROID-SLAM
rm -rf "$BASELINES_DIR"/glomap
rm -rf "$BASELINES_DIR"/LightGlue
rm -rf "$BASELINES_DIR"/MASt3R-SLAM
rm -rf "$BASELINES_DIR"/MonoGS
rm -rf "$BASELINES_DIR"/OKVIS2
rm -rf "$BASELINES_DIR"/ORB-SLAM2
rm -rf "$BASELINES_DIR"/ORB-SLAM3
rm -rf "$BASELINES_DIR"/PyCuVSLAM
rm -rf "$BASELINES_DIR"/VGGT-SLAM
rm -rf "$BASELINES_DIR"/VGGT

rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/7SCENES
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ETH
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/EUROC
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/KITTI
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/MSD
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/NUIM
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/REPLICA
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/RGBDTUM
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER_D435I
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER_PICAM
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER_T265
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/S3LI
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/SESOKO
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/TARTANAIR
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/UT_CODA
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/OPENLORIS
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/OPENLORIS-D400
rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/OPENLORIS-T265