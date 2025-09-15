#include <opencv2/opencv.hpp>
#include <featureDetection.h>

struct HomographyEstimation {
    cv::Mat H;
    int numInliers;
    float estimationTimeMs;
    float alignmentError;
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

        double totalError = 0.0;
        std::vector<cv::Point2f> projectedPoints;
        cv::perspectiveTransform(points2, projectedPoints, result.H);
        for (size_t i = 0; i < points1.size(); ++i) {
            double dx = projectedPoints[i].x - points2[i].x;
            double dy = projectedPoints[i].y - points2[i].y;
            totalError += abs(dx) + abs(dy);
        }
        result.alignmentError = totalError / points1.size();

        return result;
}

cv::Mat stitchImages(cv::Mat image1, cv::Mat image2, cv::Mat H){
    cv::Mat stitchedImage;
    int h1, h2, w1, w2;
    h1 = image1.rows;
    w1 = image1.cols;
    h2 = image2.rows;
    w2 = image2.cols;

    cv::warpPerspective(image2, stitchedImage, H, cv::Size(w1 + w2, std::max(h1, h2)));

    for (int y = 0; y < h1; y++) {
        for (int x = 0; x < w1; x++) {
            stitchedImage.at<cv::Vec3b>(y, x) = image1.at<cv::Vec3b>(y, x);
        }
    }

    return stitchedImage;
};