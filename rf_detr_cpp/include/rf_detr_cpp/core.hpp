#ifndef _RF_DETR_CPP_CORE_HPP
#define _RF_DETR_CPP_CORE_HPP

#include <opencv2/opencv.hpp>

namespace rf_detr_cpp
{
#define file_name_t std::string

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class AbcRfDetr
{
public:
    AbcRfDetr() {}
    AbcRfDetr(float nms_th = 0.45, float conf_th = 0.3)
        : nms_thresh_(nms_th), bbox_conf_thresh_(conf_th)
    {
    }
    virtual ~AbcRfDetr() {}
    virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

protected:
    int input_w_;
    int input_h_;
    float nms_thresh_;
    float bbox_conf_thresh_;

    cv::Mat static_resize(const cv::Mat &img)
    {
        const float r = std::min(
            static_cast<float>(input_w_) / static_cast<float>(img.cols),
            static_cast<float>(input_h_) / static_cast<float>(img.rows));
        const int unpad_w = r * img.cols;
        const int unpad_h = r * img.rows;
        cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        cv::resize(img, re, re.size());
        cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        return out;
    }

    void blobFromImage(const cv::Mat &img, float *blob_data)
    {
        const size_t channels = 3;
        const size_t img_h = img.rows;
        const size_t img_w = img.cols;
        const size_t img_hw = img_h * img_w;
        float *blob_data_ch0 = blob_data;
        float *blob_data_ch1 = blob_data + img_hw;
        float *blob_data_ch2 = blob_data + img_hw * 2;
        
        // RF-DETR typically uses ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3] = {0.229f, 0.224f, 0.225f};
        
        for (size_t i = 0; i < img_hw; ++i)
        {
            const size_t src_idx = i * channels;
            // Convert BGR to RGB and normalize to [0, 1], then apply ImageNet normalization
            blob_data_ch0[i] = (static_cast<float>(img.data[src_idx + 2]) / 255.0f - mean[0]) / std[0];  // R
            blob_data_ch1[i] = (static_cast<float>(img.data[src_idx + 1]) / 255.0f - mean[1]) / std[1];  // G
            blob_data_ch2[i] = (static_cast<float>(img.data[src_idx + 0]) / 255.0f - mean[2]) / std[2];  // B
        }
    }

    float intersection_area(const Object &a, const Object &b)
    {
        const cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, const float nms_threshold)
    {
        picked.clear();

        const int n = objects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; ++i)
        {
            areas[i] = objects[i].rect.area();
        }

        for (int i = 0; i < n; ++i)
        {
            const Object &a = objects[i];
            const int picked_size = picked.size();

            bool keep = true;
            for (int j = 0; j < picked_size; ++j)
            {
                const Object &b = objects[picked[j]];

                const float inter_area = intersection_area(a, b);
                const float union_area = areas[i] + areas[picked[j]] - inter_area;
                if (inter_area / union_area > nms_threshold)
                {
                    keep = false;
                    break;
                }
            }

            if (keep)
                picked.push_back(i);
        }
    }
};
}

#endif