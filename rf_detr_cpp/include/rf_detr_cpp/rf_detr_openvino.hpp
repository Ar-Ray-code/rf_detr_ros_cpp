#ifndef _RF_DETR_CPP_RF_DETR_OPENVINO_HPP
#define _RF_DETR_CPP_RF_DETR_OPENVINO_HPP

#include "rf_detr_cpp/config.h"

#ifdef ENABLE_OPENVINO

#include <openvino/openvino.hpp>
#include "core.hpp"

namespace rf_detr_cpp
{
class RfDetrOpenVINO : public AbcRfDetr
{
public:
    RfDetrOpenVINO(
        const file_name_t &model_path,
        const std::string &device = "AUTO",
        float nms_th = 0.45,
        float conf_th = 0.3);

    virtual ~RfDetrOpenVINO();
    std::vector<Object> inference(const cv::Mat &frame) override;

private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    ov::Tensor boxes_tensor_;  // Output 0: bounding boxes
    ov::Tensor scores_tensor_; // Output 1: class scores
    
    float scale_;
    
    void preprocess(const cv::Mat &image);
    std::vector<Object> postprocess(const cv::Mat &image);
};
}

#endif // ENABLE_OPENVINO
#endif // _RF_DETR_CPP_RF_DETR_OPENVINO_HPP