#ifndef _RF_DETR_CPP_UTILS_HPP
#define _RF_DETR_CPP_UTILS_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "core.hpp"

namespace rf_detr_cpp
{
namespace utils
{
std::vector<std::string> read_class_labels_file(const std::string & file_name)
{
  std::vector<std::string> class_names;
  std::ifstream file(file_name);
  std::string line;
  while (std::getline(file, line)) {
    class_names.push_back(line);
  }
  return class_names;
}

void draw_objects(
  cv::Mat & image, const std::vector<Object> & objects,
  const std::vector<std::string> & class_names)
{
  static const cv::Scalar colors[6] = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 255, 255)
  };

  for (size_t i = 0; i < objects.size(); i++) {
    const Object & obj = objects[i];

    const cv::Scalar & color = colors[i % 6];

    cv::rectangle(image, obj.rect, color, 2);

    char text[256];
    if (obj.label < static_cast<int>(class_names.size())) {
      sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);
    } else {
      sprintf(text, "Class %d %.1f%%", obj.label, obj.prob * 100);
    }

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = obj.rect.x;
    int y = obj.rect.y - label_size.height - baseLine;
    if (y < 0) {
      y = 0;
    }
    if (x + label_size.width > image.cols) {
      x = image.cols - label_size.width;
    }

    cv::rectangle(
      image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
      color, -1);

    cv::putText(
      image, text, cv::Point(x, y + label_size.height),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
  }
}
}
}

#endif
