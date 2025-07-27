import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='input video source'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='./src/rf_detr_ros_cpp/weights/rf_detr/rf_detr_base_coco.onnx',
            description='RF-DETR model path (ONNX or OpenVINO XML).'
        ),
        DeclareLaunchArgument(
            'class_labels_path',
            default_value='',
            description='if use custom model, set class name labels. '
        ),
        DeclareLaunchArgument(
            'openvino_device',
            default_value='CPU',
            description='model device. CPU, GPU, MYRIAD, etc...'
        ),
        DeclareLaunchArgument(
            'conf',
            default_value='0.30',
            description='RF-DETR confidence threshold.'
        ),
        DeclareLaunchArgument(
            'nms',
            default_value='0.45',
            description='RF-DETR nms threshold'
        ),
        DeclareLaunchArgument(
            'imshow_isshow',
            default_value='true',
            description=''
        ),
        DeclareLaunchArgument(
            'src_image_topic_name',
            default_value='/image_raw',
            description='topic name for source image'
        ),
        DeclareLaunchArgument(
            'publish_image_topic_name',
            default_value='/rf_detr/image_raw',
            description='topic name for publishing image with bounding box drawn'
        ),
        DeclareLaunchArgument(
            'publish_boundingbox_topic_name',
            default_value='/rf_detr/bounding_boxes',
            description='topic name for publishing bounding box message.'
        ),
        DeclareLaunchArgument(
            'publish_resized_image',
            default_value='false',
            description='publish resized image.'
        ),
    ]
    container = ComposableNodeContainer(
        name='rf_detr_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='usb_cam',
                plugin='usb_cam::UsbCamNode',
                name='usb_cam_node',
                parameters=[{
                    'video_device': LaunchConfiguration('video_device'),
                    'brightness': 100
                }]),
            ComposableNode(
                package='rf_detr_ros_cpp',
                plugin='rf_detr_ros_cpp::RfDetrNode',
                name='rf_detr_ros_cpp',
                parameters=[{
                    'model_path': LaunchConfiguration('model_path'),
                    'class_labels_path': LaunchConfiguration('class_labels_path'),
                    'model_type': 'openvino',
                    'openvino_device': LaunchConfiguration('openvino_device'),
                    'conf': LaunchConfiguration('conf'),
                    'nms': LaunchConfiguration('nms'),
                    'imshow_isshow': LaunchConfiguration('imshow_isshow'),
                    'src_image_topic_name': LaunchConfiguration('src_image_topic_name'),
                    'publish_image_topic_name': LaunchConfiguration('publish_image_topic_name'),
                    'publish_boundingbox_topic_name': LaunchConfiguration('publish_boundingbox_topic_name'),
                    'publish_resized_image': LaunchConfiguration('publish_resized_image'),
                }],
                ),
        ],
        output='screen',
    )

    return launch.LaunchDescription(
        launch_args +
        [
            container
        ]
    )