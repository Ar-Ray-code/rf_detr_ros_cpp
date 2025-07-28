#!/bin/bash

# RF-DETR OpenVINO model download script
# This script downloads RF-DETR models from Hugging Face
# Usage: ./download_openvino.bash [m|s|n] [base|medium|small|nano]
# Arguments:
#   m/s/n: Model size (m=medium, s=small, n=nano) or 'base' for base model
#   Alternatively, you can specify the full model name

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR"

# Function to show usage
show_usage() {
    echo "RF-DETR OpenVINO Model Download Script"
    echo "======================================"
    echo ""
    echo "Usage: $0 [MODEL_SIZE]"
    echo ""
    echo "MODEL_SIZE options:"
    echo "  n, nano    - RF-DETR Nano model (from onnx-community)"
    echo "  s, small    - RF-DETR Small model (from onnx-community)"
    echo "  m, medium   - RF-DETR Medium model (from onnx-community)"
    echo "  base       - RF-DETR Base model (default, from PierreMarieCurie)"
    echo ""
    echo "Examples:"
    echo "  $0           # Downloads base model"
    echo "  $0 n         # Downloads nano model"
    echo "  $0 s         # Downloads small model"
    echo "  $0 m         # Downloads medium model"
    echo "  $0 nano      # Downloads nano model"
    echo "  $0 small     # Downloads small model"
    echo "  $0 medium    # Downloads medium model"
    echo "  $0 base      # Downloads base model"
    echo ""
}

# Parse command line arguments
MODEL_SIZE="base"
if [ $# -gt 0 ]; then
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        n|nano)
            MODEL_SIZE="nano"
            ;;
        base)
            MODEL_SIZE="base"
            ;;
        m|medium)
            MODEL_SIZE="medium"
            ;;
        s|small)
            MODEL_SIZE="small"
            ;;
        *)
            echo "Error: Unknown model size '$1'"
            echo ""
            show_usage
            exit 1
            ;;
    esac
fi

# Set model parameters based on size
case "$MODEL_SIZE" in
    nano)
        MODEL_NAME="rf_detr_nano_coco"
        MODEL_URL="https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx"
        ;;
    small)
        MODEL_NAME="rf_detr_small_coco"
        MODEL_URL="https://huggingface.co/onnx-community/rfdetr_small-ONNX/resolve/main/onnx/model.onnx"
        ;;
    medium)
        MODEL_NAME="rf_detr_medium_coco"
        MODEL_URL="https://huggingface.co/onnx-community/rfdetr_medium-ONNX/resolve/main/onnx/model.onnx"
        ;;
    base)
        MODEL_NAME="rf_detr_base_coco" 
        MODEL_URL="https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-base.onnx"
        ;;
    *)
        echo "Error: Model size '$MODEL_SIZE' is not available."
        echo "Available models: nano, small, medium, base"
        exit 1
        ;;
esac

ONNX_FILE="$MODELS_DIR/${MODEL_NAME}.onnx"

echo "RF-DETR OpenVINO Model Download Script"
echo "======================================"
echo ""
echo "Model size: $MODEL_SIZE"
echo "Model name: $MODEL_NAME"
echo "Download URL: $MODEL_URL"
echo "Output file: $ONNX_FILE"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Check if model already exists
if [ -f "$ONNX_FILE" ]; then
    echo "Model already exists: $ONNX_FILE"
    echo "Skipping download."
    echo ""
    echo "To re-download, remove the existing file:"
    echo "  rm $ONNX_FILE"
    echo ""
    echo "Model ready for use with rf_detr_ros_cpp:"
    echo "  ros2 launch rf_detr_ros_cpp rf_detr_openvino.launch.py model_path:=$ONNX_FILE"
    exit 0
fi

# Download the model
echo ""
echo "Downloading RF-DETR $MODEL_SIZE model..."
echo ""

# Try wget first (most common and reliable)
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    if wget --progress=bar:force:noscroll -O "$ONNX_FILE" "$MODEL_URL"; then
        echo "Download completed successfully with wget!"
    else
        echo "wget download failed, removing partial file..."
        rm -f "$ONNX_FILE"
        DOWNLOAD_SUCCESS=false
    fi
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    if curl -L --progress-bar -o "$ONNX_FILE" "$MODEL_URL"; then
        echo "Download completed successfully with curl!"
    else
        echo "curl download failed, removing partial file..."
        rm -f "$ONNX_FILE"
        DOWNLOAD_SUCCESS=false
    fi
else
    echo "Neither wget nor curl available."
    DOWNLOAD_SUCCESS=false
fi

# Verify download
if [ -f "$ONNX_FILE" ]; then
    file_size=$(stat -c%s "$ONNX_FILE" 2>/dev/null || stat -f%z "$ONNX_FILE" 2>/dev/null || echo "unknown")
    echo ""
    echo "✓ Download successful!"
    echo "  File: $ONNX_FILE"
    echo "  Size: $file_size bytes"
else
    echo "✗ Download failed!"
    exit 1
fi

echo ""
echo "Model ready for use!"
echo "==================="
echo ""
echo "OpenVINO can directly use the ONNX model without conversion."
echo ""
echo "To use with rf_detr_ros_cpp, set the model_path parameter to:"
echo "  $ONNX_FILE"
echo ""
echo "Example launch command:"
echo "  ros2 launch rf_detr_ros_cpp rf_detr_openvino.launch.py model_path:=$ONNX_FILE"
echo ""