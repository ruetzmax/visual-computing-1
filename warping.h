#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "featureDetection.h"

struct HomographyEstimation {
    cv::Mat H;
    int numInliers;
    float estimationTimeMs;
    float alignmentError;
};


HomographyEstimation estimateHomography(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, FeatureMatches matches, float threshold);
cv::Mat stitchImages(cv::Mat image1, cv::Mat image2, cv::Mat H);
