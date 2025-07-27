#pragma once

#include <cmath>
#include <chrono>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "rf_detr_cpp/rf_detr.hpp"
#include "rf_detr_cpp/utils.hpp"
#include "rf_detr_cpp/coco_names.hpp"

namespace rf_detr_ros_cpp{
class RfDetrNode : public rclcpp::Node
{
public:
    RfDetrNode(const rclcpp::NodeOptions &);
private:
    void onInit();
    void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);

    static vision_msgs::msg::Detection2DArray objects_to_detection2d(const std::vector<rf_detr_cpp::Object> &, const std_msgs::msg::Header &);

private:
    std::unique_ptr<rf_detr_cpp::AbcRfDetr> rf_detr_;
    std::vector<std::string> class_names_;

    rclcpp::TimerBase::SharedPtr init_timer_;
    image_transport::Subscriber sub_image_;

    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_detection2d_;
    image_transport::Publisher pub_image_;

    // Parameters
    std::string model_path_;
    std::string model_type_;
    std::string class_labels_path_;
    std::string openvino_device_;
    std::string src_image_topic_name_;
    std::string publish_boundingbox_topic_name_;
    std::string publish_image_topic_name_;
    bool imshow_isshow_;
    bool publish_resized_image_;
    double conf_;
    double nms_;
};
}