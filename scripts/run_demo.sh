#!/bin/bash

# run_demo.sh - Basketball YOLO inference demo script for ZCU104
#
# Usage examples:
# ./run_demo.sh yolo11n /home/xilinx/quantized_dpuczdx8g/yolo11n/yolo11n_dpuczdx8g_int8.onnx /home/xilinx/basketball_vision/test_images
# ./run_demo.sh yolo11s /home/xilinx/quantized_dpuczdx8g/yolo11s/yolo11s_dpuczdx8g_int8.onnx /home/xilinx/basketball_vision/test_images
# ./run_demo.sh yolo11m /home/xilinx/quantized_dpuczdx8g/yolo11m/yolo11m_dpuczdx8g_optimized.onnx /home/xilinx/basketball_vision/test_images

set -e

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name> <onnx_model_path> <frames_or_video_path>"
    echo ""
    echo "Examples:"
    echo "  $0 yolo11n /path/to/yolo11n_dpuczdx8g_int8.onnx /path/to/frames"
    echo "  $0 yolo11s /path/to/yolo11s_dpuczdx8g_optimized.onnx /path/to/video.mp4"
    exit 1
fi

MODEL_NAME="$1"
MODEL_PATH="$2"
INPUT_PATH="$3"

# Validate inputs
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -e "$INPUT_PATH" ]; then
    echo "ERROR: Input path not found: $INPUT_PATH"
    exit 1
fi

# Setup cache directory
CACHE_BASE="/home/xilinx/.vaip_cache"
CACHE_DIR="$CACHE_BASE/$MODEL_NAME"

echo "Creating cache directory: $CACHE_DIR"
mkdir -p "$CACHE_DIR"

# Optional environment variables for VAIP/VART (uncomment if needed)
# export XLNX_VART_FIRMWARE=/usr/lib/dpu.xclbin
# export VART_LOG_LEVEL=1
# export XLNX_ENABLE_FINGERPRINT_CHECK=0

# Print header
echo ""
echo "======================================================="
echo "Basketball YOLO Inference Demo"
echo "======================================================="
echo "Model:       $MODEL_NAME"
echo "ONNX file:   $MODEL_PATH"
echo "Input:       $INPUT_PATH"
echo "Cache dir:   $CACHE_DIR"
echo "Board:       ZCU104 (DPUCZDX8G_B4096)"
echo "======================================================="
echo ""

# Determine input type and run appropriate command
if [ -d "$INPUT_PATH" ]; then
    echo "Running inference on image directory..."
    python3 deploy_onnx_vaip.py \
        --model "$MODEL_PATH" \
        --frames "$INPUT_PATH" \
        --cache "$CACHE_DIR" \
        --conf 0.25 \
        --iou 0.45 \
        --size 640 640 \
        --warmup 3 \
        --show
elif [ -f "$INPUT_PATH" ]; then
    echo "Video input detected (not yet implemented)"
    echo "Please use a directory of images instead"
    exit 1
else
    echo "ERROR: Input path is neither file nor directory: $INPUT_PATH"
    exit 1
fi

echo ""
echo "Demo completed!"
echo "Cache persisted at: $CACHE_DIR"
