#include "rf_detr_cpp/rf_detr_openvino.hpp"

#ifdef ENABLE_OPENVINO

#include <opencv2/opencv.hpp>
#include <algorithm>

namespace rf_detr_cpp
{
RfDetrOpenVINO::RfDetrOpenVINO(
    const file_name_t &model_path,
    const std::string &device,
    float nms_th,
    float conf_th)
    : AbcRfDetr(nms_th, conf_th)
{
    auto model = core_.read_model(model_path);
    
    // Get input shape
    auto input_shape = model->input().get_shape();
    input_h_ = input_shape[2];
    input_w_ = input_shape[3];
    
    
    compiled_model_ = core_.compile_model(model, device);
    infer_request_ = compiled_model_.create_infer_request();
    
    input_tensor_ = infer_request_.get_input_tensor();
    // RF-DETR has two outputs: boxes and class scores
    boxes_tensor_ = infer_request_.get_output_tensor(0);   // [1, 300, 4]
    scores_tensor_ = infer_request_.get_output_tensor(1);  // [1, 300, 91]
}

RfDetrOpenVINO::~RfDetrOpenVINO()
{
}

std::vector<Object> RfDetrOpenVINO::inference(const cv::Mat &frame)
{
    preprocess(frame);
    infer_request_.infer();
    return postprocess(frame);
}

void RfDetrOpenVINO::preprocess(const cv::Mat &image)
{
    cv::Mat resized_image = this->static_resize(image);
    
    scale_ = std::min(
        static_cast<float>(input_w_) / static_cast<float>(image.cols),
        static_cast<float>(input_h_) / static_cast<float>(image.rows));


    float *input_data = input_tensor_.data<float>();
    this->blobFromImage(resized_image, input_data);
    
}

std::vector<Object> RfDetrOpenVINO::postprocess(const cv::Mat &image)
{
    std::vector<Object> objects;
    
    const float *boxes_data = boxes_tensor_.data<float>();
    const float *scores_data = scores_tensor_.data<float>();
    
    auto boxes_shape = boxes_tensor_.get_shape();
    auto scores_shape = scores_tensor_.get_shape();
    
    
    if (boxes_shape.size() != 3 || scores_shape.size() != 3) {
        std::cerr << "Unexpected output shape dimensions" << std::endl;
        return objects;
    }
    
    size_t num_queries = boxes_shape[1];
    size_t num_classes = scores_shape[2];
    
    
    for (size_t i = 0; i < num_queries; ++i)
    {
        // Get bounding box coordinates [x1, y1, x2, y2] (normalized)
        const float *box = boxes_data + i * 4;
        const float *class_scores = scores_data + i * num_classes;
        
        // Find best class and confidence
        int best_class = 0;
        float max_score = 0.0f;
        
        for (size_t j = 0; j < num_classes; ++j)
        {
            if (class_scores[j] > max_score)
            {
                max_score = class_scores[j];
                best_class = static_cast<int>(j);
            }
        }
        
        
        // Apply confidence threshold and skip background class (usually class 0 or last class)
        // For COCO, class 0 might be background, so we check best_class > 0
        if (max_score > bbox_conf_thresh_ && best_class > 0 && best_class < 91)
        {
            // RF-DETR outputs normalized coordinates in cxcywh format [center_x, center_y, width, height]
            float cx_norm = box[0];  // center x (normalized)
            float cy_norm = box[1];  // center y (normalized)
            float w_norm = box[2];   // width (normalized)
            float h_norm = box[3];   // height (normalized)
            
            // Convert cxcywh to xyxy format (still normalized)
            float x1_norm = cx_norm - 0.5f * w_norm;
            float y1_norm = cy_norm - 0.5f * h_norm;
            float x2_norm = cx_norm + 0.5f * w_norm;
            float y2_norm = cy_norm + 0.5f * h_norm;
            
            // Convert normalized coordinates to model input coordinates
            float x1_model = x1_norm * static_cast<float>(input_w_);
            float y1_model = y1_norm * static_cast<float>(input_h_);
            float x2_model = x2_norm * static_cast<float>(input_w_);
            float y2_model = y2_norm * static_cast<float>(input_h_);
            
            // Scale back to original image coordinates using the scale factor
            float x1 = x1_model / scale_;
            float y1 = y1_model / scale_;
            float x2 = x2_model / scale_;
            float y2 = y2_model / scale_;
            
            // Ensure coordinates are valid
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.cols - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.rows - 1)));
            x2 = std::max(x1, std::min(x2, static_cast<float>(image.cols - 1)));
            y2 = std::max(y1, std::min(y2, static_cast<float>(image.rows - 1)));
            
            // Only add if the box has valid size
            if (x2 > x1 && y2 > y1)
            {
                Object obj;
                obj.rect.x = x1;
                obj.rect.y = y1;
                obj.rect.width = x2 - x1;
                obj.rect.height = y2 - y1;
                obj.label = best_class - 1;  // Convert to 0-indexed (COCO classes 1-80 -> 0-79)
                obj.prob = max_score;
                objects.push_back(obj);
            }
        }
    }
    
    
    // Sort by confidence
    std::sort(objects.begin(), objects.end(),
                [](const Object &a, const Object &b) {
                    return a.prob > b.prob;
                });
    
    // Apply NMS
    std::vector<int> picked;
    this->nms_sorted_bboxes(objects, picked, nms_thresh_);
    
    std::vector<Object> final_objects;
    for (int idx : picked)
    {
        final_objects.push_back(objects[idx]);
    }
    
    
    return final_objects;
}
}

#endif // ENABLE_OPENVINO