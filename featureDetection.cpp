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

    int screenWidth = 1920;
    int screenHeight = 1080;

    double scale = std::min(
        (double)screenWidth / image.cols,
        (double)screenHeight / image.rows
    );
;
    cv::resize(image, image, cv::Size(), scale, scale);

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

void doFeatureDetection(){
    ImageFeatures features1_1, features1_2, features2_1, features2_2, features3_1, features3_2;
    ImageFeatures features1_1_orb, features1_2_orb, features2_1_orb, features2_2_orb, features3_1_orb, features3_2_orb;
    cv::Mat image1_1, image1_2, image2_1, image2_2, image3_1, image3_2;

    // Load images
    image1_1 = load_image("../images/1_1.jpg");
    image1_2 = load_image("../images/1_2.jpg");

    image2_1 = load_image("../images/1_1.jpg");
    image2_2 = load_image("../images/1_2.jpg");

    image3_1 = load_image("../images/1_1.jpg");
    image3_2 = load_image("../images/1_2.jpg");

    // Do feature extraction using SIFT
    features1_1 = extract_features(image1_1);
    features1_2 = extract_features(image1_2);

    features2_1 = extract_features(image2_1);
    features2_2 = extract_features(image2_2);

    features3_1 = extract_features(image3_1);
    features3_2 = extract_features(image3_2);

    // Do feature extraction using ORB
    features1_1_orb = extract_features(image1_1, FeatureDetectorMethod::ORB);
    features1_2_orb = extract_features(image1_2, FeatureDetectorMethod::ORB);

    features2_1_orb = extract_features(image2_1, FeatureDetectorMethod::ORB);
    features2_2_orb = extract_features(image2_2, FeatureDetectorMethod::ORB);

    features3_1_orb = extract_features(image3_1, FeatureDetectorMethod::ORB);
    features3_2_orb = extract_features(image3_2, FeatureDetectorMethod::ORB);
    
    // Save images with keypoints drawn
    cv::imwrite("../outputs/image1_1_keypoints.jpg", features1_1.imageWithKeypoints);
    cv::imwrite("../outputs/image2_1_keypoints.jpg", features1_2.imageWithKeypoints);

    cv::imwrite("../outputs/image1_2_keypoints.jpg", features2_1.imageWithKeypoints);
    cv::imwrite("../outputs/image2_2_keypoints.jpg", features2_2.imageWithKeypoints);

    cv::imwrite("../outputs/image1_3_keypoints.jpg", features3_1.imageWithKeypoints);
    cv::imwrite("../outputs/image2_3_keypoints.jpg", features3_2.imageWithKeypoints);

    cv::imwrite("../outputs/image1_1_keypoints_orb.jpg", features1_1_orb.imageWithKeypoints);
    cv::imwrite("../outputs/image2_1_keypoints_orb.jpg", features1_2_orb.imageWithKeypoints);

    cv::imwrite("../outputs/image1_2_keypoints_orb.jpg", features2_1_orb.imageWithKeypoints);
    cv::imwrite("../outputs/image2_2_keypoints_orb.jpg", features2_2_orb.imageWithKeypoints);

    cv::imwrite("../outputs/image1_3_keypoints_orb.jpg", features3_1_orb.imageWithKeypoints);
    cv::imwrite("../outputs/image2_3_keypoints_orb.jpg", features3_2_orb.imageWithKeypoints);

    // Plot bar chart of #keypoint per image (pair) by extraction method
    double numKeypoints1 = features1_1.keypoints.size() + features1_2.keypoints.size();
    double numKeypoints2 = features2_1.keypoints.size() + features2_2.keypoints.size();
    double numKeypoints3 = features3_1.keypoints.size() + features3_2.keypoints.size();

    double numKeypoints1_orb = features1_1_orb.keypoints.size() + features1_2_orb.keypoints.size();
    double numKeypoints2_orb = features2_1_orb.keypoints.size() + features2_2_orb.keypoints.size();
    double numKeypoints3_orb = features3_1_orb.keypoints.size() + features3_2_orb.keypoints.size();

    std::vector<std::vector<double>> numKeypointByMethod = {{numKeypoints1, numKeypoints1_orb}, {numKeypoints2, numKeypoints2_orb}, {numKeypoints3, numKeypoints3_orb}};
    matplot::bar(numKeypointByMethod);
    matplot::ylabel("# Keypoints");
    matplot::gca()->x_axis().ticklabels({"SIFT", "ORB"});
    matplot::legend({"Image Pair 1", "Image Pair 2", "Image Pair 3"});
    matplot::title("Number of Keypoints Detected by Feature Extraction Method");
    matplot::save("../plots/num_keypoints.jpg");

    // Match features using BFMatcher
    FeatureMatches matches1 = match_features(features1_1, features1_2);
    FeatureMatches matches2 = match_features(features2_1, features2_2);
    FeatureMatches matches3 = match_features(features3_1, features3_2);

    FeatureMatches matches1_orb = match_features(features1_1_orb, features1_2_orb);
    FeatureMatches matches2_orb = match_features(features2_1_orb, features2_2_orb);
    FeatureMatches matches3_orb = match_features(features3_1_orb, features3_2_orb);

    // Plot bar chart of matching time per image (pair) by extraction method
    std::vector<std::vector<double>> matchingTimeByMethod = {{matches1.matchingTimeMs, matches1_orb.matchingTimeMs}, {matches2.matchingTimeMs, matches2_orb.matchingTimeMs}, {matches3.matchingTimeMs, matches3_orb.matchingTimeMs}};
    matplot::bar(matchingTimeByMethod);
    matplot::ylabel("Matching Time (ms)");
    matplot::gca()->x_axis().ticklabels({"SIFT", "ORB"});
    matplot::legend({"Image Pair 1", "Image Pair 2", "Image Pair 3"});
    matplot::title("Feature Matching Time by Extraction Method");
    matplot::save("../plots/matching_time.jpg");

    // Plot histograms of match distances by extraction method for each image
    matplot::title("Histogram of Match Distances by Extraction Method");
    matplot::figure();
    matplot::subplot(3, 1, 1);
    matplot::hist(matches1.distances, 30);
    matplot::hold(matplot::on);
    matplot::hist(matches1_orb.distances, 30);
    matplot::ylabel("Frequency");
    matplot::subplot(3, 1, 2);
    matplot::hist(matches2.distances, 30);
    matplot::hold(matplot::on);
    matplot::hist(matches2_orb.distances, 30);
    matplot::ylabel("Frequency");
    matplot::subplot(3, 1, 3);
    matplot::hist(matches3.distances, 30);
    matplot::hold(matplot::on);
    matplot::hist(matches3_orb.distances, 30);
    matplot::ylabel("Frequency");
    matplot::save("../plots/match_distances.jpg");
    matplot::show();

    //TODO: add title / x-Axis label + maybe normalize?


}

