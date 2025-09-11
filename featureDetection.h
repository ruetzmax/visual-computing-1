#pragma once
#include <opencv2/opencv.hpp>
#include <matplot/matplot.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>

enum class FeatureDetectorMethod {
    SIFT,
    ORB
};

struct ImageFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat imageWithKeypoints;
};

struct FeatureMatches {
    std::vector<cv::DMatch> matches;
    std::vector<float> distances;
    double matchingTimeMs;
};

cv::Mat load_image(const std::string &path);
ImageFeatures extract_features(const cv::Mat &image, const FeatureDetectorMethod method = FeatureDetectorMethod::SIFT);
FeatureMatches match_features(const ImageFeatures &features1, const ImageFeatures &features2);
void doFeatureDetection();
