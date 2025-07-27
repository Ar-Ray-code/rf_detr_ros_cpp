# rf_detr_ros_cpp
RF-DETR wrapper for ROS 2 C++ implementation.

[![ci_jazzy](https://github.com/Ar-Ray-code/rf_detr_ros_cpp/actions/workflows/ci_jazzy.yml/badge.svg)](https://github.com/Ar-Ray-code/rf_detr_ros_cpp/actions/workflows/ci_jazzy.yml)

## Target device

- Intel CPU, GPU (OpenVINO)

## Dependencies

- ROS 2 Humble or later
- OpenVINO 2023.1.0 or later

## Installation

```bash
mkdir -p ~/ros2_ws/src/
cd ~/ros2_ws/src/
git clone https://github.com/Ar-Ray-code/rf_detr_ros_cpp.git
cd ~/ros2_ws/
colcon build --packages-up-to rf_detr_ros_cpp --cmake-args -DRF_DETR_USE_OPENVINO=ON
```

### Download models

```bash
cd ~/ros2_ws/
./src/rf_detr_ros_cpp/weights/rf_detr/download_openvino.bash
```


## Usage

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch rf_detr_ros_cpp rf_detr_openvino.launch.py  model_path:=./src/rf_detr_ros_cpp/weights/rf_detr/rf_detr_base_coco.onnx openvino_device:=GPU
```

## References

- [RF-DETR](https://github.com/roboflow/rf-detr) : RF-DETR is a real-time object detection model architecture developed by Roboflow, SOTA on COCO and designed for fine-tuning.

- [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) : YOLOX + ROS2 object detection package (C++ only support). This is a stable package that has been maintained for over four years.