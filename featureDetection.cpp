#include <iostream>
#include <opencv2/opencv.hpp>
#include <matplot/matplot.h>

enum class FeatureDetectorMethod {
    SIFT,
    ORB
};

cv::Mat load_image(const std::string &path)
{
    cv::Mat image = cv::imread(path);
    if (image.empty())
    {
        throw std::runtime_error("Could not open or find the image!");
    }

    return image;
}

struct ImageFeatures
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat imageWithKeypoints;
};

struct FeatureMatches
{
    std::vector<cv::DMatch> matches;
    std::vector<float> distances;
    double matchingTimeMs;
};

ImageFeatures extract_features(const cv::Mat &image, const FeatureDetectorMethod method = FeatureDetectorMethod::SIFT)
{
    ImageFeatures features;
    switch(method){
        case FeatureDetectorMethod::SIFT: {
            cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
            sift->detectAndCompute(image, cv::noArray(), features.keypoints, features.descriptors);
            break;
        }
        case FeatureDetectorMethod::ORB: {
            cv::Ptr<cv::ORB> orb = cv::ORB::create();
            orb->detectAndCompute(image, cv::noArray(), features.keypoints, features.descriptors);
            break;
        }
    default:
        throw std::invalid_argument("Unsupported feature detector method");
    }

    cv::drawKeypoints(image, features.keypoints, features.imageWithKeypoints);
    return features;
}

FeatureMatches match_features(const ImageFeatures &features1, const ImageFeatures &features2)
{
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;

    auto matchStart = std::chrono::high_resolution_clock::now();
    matcher.match(features1.descriptors, features2.descriptors, matches);
    auto matchEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = matchEnd - matchStart;
    double matchingTimeMs = elapsed.count();

    std::vector<float> distances;
    for (const auto &match : matches)
    {
        distances.push_back(match.distance);
    }
    return {matches, distances, matchingTimeMs};
}


