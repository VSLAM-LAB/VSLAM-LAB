#!/bin/bash

DO_PIXI=false
DO_DATASETS=false
DO_BASELINES=false
DO_EVALUATION=false
for arg in "$@"; do
    case "$arg" in
        --pixi) DO_PIXI=true ;;
        --datasets) DO_DATASETS=true ;;
        --baselines) DO_BASELINES=true ;;
        --evaluation) DO_EVALUATION=true ;;
        *) echo "Unknown flag: $arg" >&2; echo "Usage: $0 [--pixi] [--datasets] [--baselines] [--evaluation]" >&2; exit 1 ;;
    esac
done

if ! $DO_PIXI && ! $DO_DATASETS && ! $DO_BASELINES && ! $DO_EVALUATION; then
    echo "Usage: $0 [--pixi] [--datasets] [--baselines] [--evaluation]"
    echo "  --pixi        Remove __pycache__ dirs, pixi envs/lock, and the system .cache directory"
    echo "  --datasets    Remove downloaded benchmark datasets"
    echo "  --baselines   Remove cloned baseline source directories"
    echo "  --evaluation  Remove the VSLAM-LAB-Evaluation folder"
    exit 0
fi

VSLAM_LAB_DIR="/home/alejandro/VSLAM-LAB"
VSLAM_LAB_BENCHMARK_DIR="/home/alejandro/VSLAM-LAB-Benchmark"
VSLAM_LAB_EVALUATION_DIR="/home/alejandro/VSLAM-LAB-Evaluation"
BASELINES_DIR="$VSLAM_LAB_DIR/Baselines"

CACHE_DIR=/home/alejandro/.cache

echo "Resetting VSLAM-LAB directory: $VSLAM_LAB_DIR"

if $DO_PIXI; then
    echo "Removing __pycache__ directories..."
    find "$VSLAM_LAB_DIR" -type d -name "__pycache__" -exec rm -rf {} +

    echo "Removing pixi envs: $VSLAM_LAB_DIR/.pixi/envs* and $VSLAM_LAB_DIR/.pixi/task-cache-v0*"
    rm -rf "$VSLAM_LAB_DIR"/.pixi/envs*
    rm -rf "$VSLAM_LAB_DIR"/.pixi/task-cache-v0*

    echo "Removing pixi.lock: $VSLAM_LAB_DIR/pixi.lock*"
    rm -rf "$VSLAM_LAB_DIR"/pixi.lock*

    echo "Removing system .cache directory: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
fi

if $DO_BASELINES; then
    echo "Removing Baselines..."
    rm -rf "$BASELINES_DIR"/*-DEV*
    rm -rf "$BASELINES_DIR"/*git_clone_*
    rm -rf "$BASELINES_DIR"/AnyFeature-VSLAM
    rm -rf "$BASELINES_DIR"/colmap
    rm -rf "$BASELINES_DIR"/DPVO
    rm -rf "$BASELINES_DIR"/DROID-SLAM
    rm -rf "$BASELINES_DIR"/DROID-SLAM-DEV
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
    rm -rf "$BASELINES_DIR"/AllFeature-VSLAM-DEV
    rm -rf "$BASELINES_DIR"/Depth-Anything-3
    rm -rf "$BASELINES_DIR"/VPR-LAB
fi

if $DO_DATASETS; then
    echo "Removing Benchmark datasets..."
    rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ETH
    rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/RGBDTUM
    rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/REPLICA
    rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/EUROC
    rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/TARTANAIR
    rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/NUIM
    rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/7SCENES

    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/KITTI
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/MSD
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER-D435I
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER-PICAM
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ROVER-T265
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/S3LI
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/SESOKO
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/UT-CODA
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/OPENLORIS
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/OPENLORIS-D400
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/OPENLORIS-T265
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/SWEETCORALS
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/ARIEL
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/CAVES
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/DRUNKARDS
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/HAMLYN
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/HILTI2022
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/HILTI2026
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/MADMAX
    # rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/VITUM
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/Replica.zip
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/SONEVA
    #rm -rf "$VSLAM_LAB_BENCHMARK_DIR"/EIFFEL-TOWER
fi

if $DO_EVALUATION; then
    echo "Removing VSLAM-LAB-Evaluation directory: $VSLAM_LAB_EVALUATION_DIR"
    rm -rf "$VSLAM_LAB_EVALUATION_DIR"
fi
