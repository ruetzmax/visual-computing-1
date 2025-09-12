#include <opencv2/opencv.hpp>
#include <featureDetection.h>

struct HomographyEstimation {
    cv::Mat H;
    int numInliers;
    float estimationTimeMs;
};

HomographyEstimation estimateHomography(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, FeatureMatches matches, float threshold){
    HomographyEstimation result;

    std::vector<cv::Point2f> points1, points2;
        for (const auto &match : matches.matches)
        {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        auto mask = cv::Mat();
        
        auto estimationStart = std::chrono::high_resolution_clock::now();
        result.H = cv::findHomography(points1, points2, cv::RANSAC, threshold, mask);
        auto estimationEnd = std::chrono::high_resolution_clock::now();
        result.estimationTimeMs = std::chrono::duration<float, std::milli>(estimationEnd - estimationStart).count();
        
        result.numInliers = cv::countNonZero(mask);

        return result;
}