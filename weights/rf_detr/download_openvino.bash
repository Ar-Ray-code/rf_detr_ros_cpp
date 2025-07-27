#!/bin/bash

# RF-DETR OpenVINO model download script
# This script downloads RF-DETR models from Hugging Face and converts them to OpenVINO format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR"

echo "RF-DETR OpenVINO Model Download Script"
echo "======================================"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# RF-DETR Base model from Hugging Face
echo "Available RF-DETR models:"
echo "1. RF-DETR Base (COCO dataset) - from Hugging Face"

# RF-DETR Base model (COCO dataset)
MODEL_NAME="rf_detr_base_coco"
MODEL_URL="https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-base.onnx"
ONNX_FILE="$MODELS_DIR/${MODEL_NAME}.onnx"
XML_FILE="$MODELS_DIR/${MODEL_NAME}.xml"
BIN_FILE="$MODELS_DIR/${MODEL_NAME}.bin"

echo ""
echo "Downloading RF-DETR Base model from Hugging Face..."
echo "URL: $MODEL_URL"
echo ""

# Download ONNX model
if [ ! -f "$ONNX_FILE" ]; then
    echo "Downloading ONNX model..."
    if command -v wget &> /dev/null; then
        wget -O "$ONNX_FILE" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$ONNX_FILE" "$MODEL_URL"
    else
        echo "Error: Neither wget nor curl is available!"
        echo "Please install wget or curl and try again."
        echo ""
        echo "Manual download command:"
        echo "wget -O $ONNX_FILE $MODEL_URL"
        echo "or"
        echo "curl -L -o $ONNX_FILE $MODEL_URL"
        exit 1
    fi
    
    if [ -f "$ONNX_FILE" ]; then
        echo "Downloaded: $ONNX_FILE"
    else
        echo "Error: Download failed!"
        exit 1
    fi
else
    echo "ONNX model already exists: $ONNX_FILE"
fi

echo ""
echo "Download completed!"
echo "Model files location: $MODELS_DIR"
echo ""
echo "Files created:"
echo "  - ONNX model: $ONNX_FILE"
echo ""
echo "OpenVINO can directly use the ONNX model without conversion."
echo "To use with rf_detr_ros_cpp, set the model_path parameter to:"
echo "  $ONNX_FILE"
echo ""
echo "Example launch command:"
echo "  ros2 launch rf_detr_ros_cpp rf_detr_openvino.launch.py model_path:=$ONNX_FILE"
echo ""
echo "Optional: Convert to OpenVINO IR format for potential optimization"
echo "========================================================================"
# read -p "Do you want to convert to OpenVINO IR format (XML/BIN)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Converting ONNX to OpenVINO IR format..."
    if [ ! -f "$XML_FILE" ] || [ ! -f "$BIN_FILE" ]; then
        # Check if OpenVINO model optimizer is available
        if command -v mo &> /dev/null; then
            mo --input_model "$ONNX_FILE" --output_dir "$MODELS_DIR" --model_name "$MODEL_NAME"
            echo "Converted to OpenVINO IR format: $XML_FILE, $BIN_FILE"
            echo ""
            echo "You can now use either:"
            echo "  - ONNX: $ONNX_FILE"
            echo "  - OpenVINO IR: $XML_FILE"
        else
            echo "Error: OpenVINO model optimizer (mo) not found!"
            echo "Please install OpenVINO toolkit if you want to convert."
            echo ""
            echo "Installation instructions:"
            echo "1. Download OpenVINO from: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html"
            echo "2. Install and source the environment:"
            echo "   source /opt/intel/openvino/setupvars.sh"
            echo ""
            echo "Manual conversion command:"
            echo "mo --input_model $ONNX_FILE --output_dir $MODELS_DIR --model_name $MODEL_NAME"
            echo ""
            echo "For now, you can use the ONNX model directly: $ONNX_FILE"
        fi
    else
        echo "OpenVINO IR format already exists: $XML_FILE, $BIN_FILE"
    fi
else
    echo "Skipping conversion. Using ONNX model directly: $ONNX_FILE"
fi