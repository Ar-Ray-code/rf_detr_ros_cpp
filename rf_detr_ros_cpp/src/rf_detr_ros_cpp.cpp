#include "rf_detr_ros_cpp/rf_detr_ros_cpp.hpp"

namespace rf_detr_ros_cpp
{
RfDetrNode::RfDetrNode(const rclcpp::NodeOptions & options)
: Node("rf_detr_ros_cpp", options)
{
  using namespace std::chrono_literals;
  this->init_timer_ = this->create_wall_timer(
    0s, std::bind(&RfDetrNode::onInit, this));
}

void RfDetrNode::onInit()
{
  this->init_timer_->cancel();

  // Declare parameters
  this->declare_parameter("imshow_isshow", false);
  this->declare_parameter("model_path", std::string("example.onnx"));
  this->declare_parameter("class_labels_path", std::string(""));
  this->declare_parameter("conf", 0.3);
  this->declare_parameter("nms", 0.45);
  this->declare_parameter("openvino_device", std::string("AUTO"));
  this->declare_parameter("model_type", std::string("openvino"));
  this->declare_parameter("src_image_topic_name", std::string("image_raw"));
  this->declare_parameter("publish_boundingbox_topic_name", std::string("rf_detr/bounding_boxes"));
  this->declare_parameter("publish_image_topic_name", std::string("rf_detr/image_raw"));
  this->declare_parameter("publish_resized_image", false);

  // Get parameters
  this->imshow_isshow_ = this->get_parameter("imshow_isshow").as_bool();
  this->model_path_ = this->get_parameter("model_path").as_string();
  this->class_labels_path_ = this->get_parameter("class_labels_path").as_string();
  this->conf_ = this->get_parameter("conf").as_double();
  this->nms_ = this->get_parameter("nms").as_double();
  this->openvino_device_ = this->get_parameter("openvino_device").as_string();
  this->model_type_ = this->get_parameter("model_type").as_string();
  this->src_image_topic_name_ = this->get_parameter("src_image_topic_name").as_string();
  this->publish_boundingbox_topic_name_ =
    this->get_parameter("publish_boundingbox_topic_name").as_string();
  this->publish_image_topic_name_ = this->get_parameter("publish_image_topic_name").as_string();
  this->publish_resized_image_ = this->get_parameter("publish_resized_image").as_bool();

  if (this->imshow_isshow_) {
    cv::namedWindow("rf_detr", cv::WINDOW_AUTOSIZE);
  }

  if (this->class_labels_path_ != "") {
    RCLCPP_INFO(
      this->get_logger(), "read class labels from '%s'",
      this->class_labels_path_.c_str());
    this->class_names_ = rf_detr_cpp::utils::read_class_labels_file(this->class_labels_path_);
  } else {
    this->class_names_ = rf_detr_cpp::COCO_CLASSES;
  }

  if (this->model_type_ == "openvino") {
#ifdef ENABLE_OPENVINO
    RCLCPP_INFO(this->get_logger(), "Model Type is OpenVINO");
    this->rf_detr_ = std::make_unique<rf_detr_cpp::RfDetrOpenVINO>(
      this->model_path_, this->openvino_device_,
      this->nms_, this->conf_);
#else
    RCLCPP_ERROR(this->get_logger(), "rf_detr_cpp is not built with OpenVINO");
    rclcpp::shutdown();
#endif
  } else {
    RCLCPP_ERROR(this->get_logger(), "Unsupported model type: %s", this->model_type_.c_str());
    rclcpp::shutdown();
  }

  RCLCPP_INFO(this->get_logger(), "model loaded");

  this->sub_image_ = image_transport::create_subscription(
    this, this->src_image_topic_name_,
    std::bind(&RfDetrNode::colorImageCallback, this, std::placeholders::_1),
    "raw");

  this->pub_detection2d_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
    this->publish_boundingbox_topic_name_,
    10);

  if (this->publish_resized_image_) {
    this->pub_image_ = image_transport::create_publisher(this, this->publish_image_topic_name_);
  }
}

void RfDetrNode::colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & ptr)
{
  auto img = cv_bridge::toCvCopy(ptr, "bgr8");
  cv::Mat frame = img->image;

  auto now = std::chrono::system_clock::now();
  auto objects = this->rf_detr_->inference(frame);
  auto end = std::chrono::system_clock::now();

  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - now);
  RCLCPP_INFO(
    this->get_logger(), "Inference time: %5ld us, Detected objects: %zu",
    elapsed.count(), objects.size());

  rf_detr_cpp::utils::draw_objects(frame, objects, this->class_names_);
  if (this->imshow_isshow_) {
    cv::imshow("rf_detr", frame);
    auto key = cv::waitKey(1);
    if (key == 27) {
      rclcpp::shutdown();
    }
  }

  if (this->pub_detection2d_ == nullptr) {
    RCLCPP_ERROR(this->get_logger(), "pub_detection2d_ is nullptr");
    return;
  }
  vision_msgs::msg::Detection2DArray detections = objects_to_detection2d(objects, img->header);
  this->pub_detection2d_->publish(detections);

  if (this->publish_resized_image_) {
    sensor_msgs::msg::Image::SharedPtr pub_img =
      cv_bridge::CvImage(img->header, "bgr8", frame).toImageMsg();
    this->pub_image_.publish(pub_img);
  }
}

vision_msgs::msg::Detection2DArray RfDetrNode::objects_to_detection2d(
  const std::vector<rf_detr_cpp::Object> & objects, const std_msgs::msg::Header & header)
{
  vision_msgs::msg::Detection2DArray detection2d;
  detection2d.header = header;
  for (const auto & obj : objects) {
    vision_msgs::msg::Detection2D det;
    det.bbox.center.position.x = obj.rect.x + obj.rect.width / 2;
    det.bbox.center.position.y = obj.rect.y + obj.rect.height / 2;
    det.bbox.size_x = obj.rect.width;
    det.bbox.size_y = obj.rect.height;

    det.results.resize(1);
    det.results[0].hypothesis.class_id = std::to_string(obj.label);
    det.results[0].hypothesis.score = obj.prob;
    detection2d.detections.emplace_back(det);
  }
  return detection2d;
}
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(rf_detr_ros_cpp::RfDetrNode)
