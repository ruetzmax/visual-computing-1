#include <iostream>
#include <opencv2/opencv.hpp>
#include <matplot/matplot.h>

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

ImageFeatures extract_features(const cv::Mat &image)
{
    ImageFeatures features;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(image, cv::noArray(), features.keypoints, features.descriptors);
    cv::drawKeypoints(image, features.keypoints, features.imageWithKeypoints);
    return features;
}

int main()
{
    ImageFeatures features1, features2;
    cv::Mat image1, image2;

    image1 = load_image("../images/1_1.jpg");
    image2 = load_image("../images/1_2.jpg");

    features1 = extract_features(image1);
    features2 = extract_features(image2);

    cv::imwrite("../outputs/image1_keypoints.jpg", features1.imageWithKeypoints);
    cv::imwrite("../outputs/image2_keypoints.jpg", features2.imageWithKeypoints);

    std::cout << "Image 1: " << features1.keypoints.size() << " keypoints detected." << std::endl;
    std::cout << "Image 2: " << features2.keypoints.size() << " keypoints detected." << std::endl;

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;

    auto matchStart = std::chrono::high_resolution_clock::now();
    matcher.match(features1.descriptors, features2.descriptors, matches);
    auto matchEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = matchEnd - matchStart;
    std::cout << "Matching took " << elapsed.count() << " ms" << std::endl;

    std::vector<float> distances;
    for (const auto &match : matches)
    {
        distances.push_back(match.distance);
    }

    matplot::hist(distances);
    matplot::save("../plots/match_distances_1.jpg");

    // cv::imshow("Image Matches", img_matches);

    // cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Image Display", img_matches);

    // cv::waitKey(0);

    // cv::destroyAllWindows();

    return 0;
}