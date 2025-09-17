#include <opencv2/opencv.hpp>
#include <featureDetection.h>

struct HomographyEstimation
{
    cv::Mat H;
    int numInliers;
    float estimationTimeMs;
    float alignmentError;
};

HomographyEstimation estimateHomography(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, FeatureMatches matches, float threshold)
{
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
    for (size_t i = 0; i < points1.size(); ++i)
    {
        double dx = projectedPoints[i].x - points2[i].x;
        double dy = projectedPoints[i].y - points2[i].y;
        totalError += abs(dx) + abs(dy);
    }
    result.alignmentError = totalError / points1.size();

    return result;
}

enum class StitchingMethod
{
    OVERLAY,
    FEATHERING
};

// blend linearly only in overlap region
float d1(int x, int overlapStart, int overlapEnd)
{
    if (x < overlapStart || x > overlapEnd)
        return 1.0f;
    else
        return 1.0f - static_cast<float>(x - overlapStart) / (overlapEnd - overlapStart);
};

float d2(int x, int overlapStart, int overlapEnd)
{
    if (x < overlapStart || x > overlapEnd)
        return 1.0f;
    else
        return static_cast<float>(x - overlapStart) / (overlapEnd - overlapStart);
};

cv::Mat stitchImages(cv::Mat image1, cv::Mat image2, cv::Mat H, StitchingMethod method = StitchingMethod::OVERLAY)
{

    // Create (warped) images of same size
    cv::Mat stitchedImage;
    int h1, h2, w1, w2;
    h1 = image1.rows;
    w1 = image1.cols;
    h2 = image2.rows;
    w2 = image2.cols;

    cv::warpPerspective(image2, stitchedImage, H, cv::Size(w1 + w2, std::max(h1, h2)));

    switch (method)
    {
    case StitchingMethod::OVERLAY:
    {
        for (int y = 0; y < h1; y++)
        {
            for (int x = 0; x < w1; x++)
            {
                stitchedImage.at<cv::Vec3b>(y, x) = image1.at<cv::Vec3b>(y, x);
            }
        }
        break;
    }
    case StitchingMethod::FEATHERING:
    {
        cv::Mat image1Expanded = cv::Mat::zeros(stitchedImage.rows, stitchedImage.cols, image1.type());
        {
            for (int y = 0; y < h1; y++)
            {
                for (int x = 0; x < w1; x++)
                {
                    image1Expanded.at<cv::Vec3b>(y, x) = image1.at<cv::Vec3b>(y, x);
                }
            }
        }

        // Create overlap mask
        cv::Mat overlapMask = cv::Mat::zeros(stitchedImage.size(), CV_8UC1);
        for (int y = 0; y < stitchedImage.rows; ++y)
        {
            for (int x = 0; x < stitchedImage.cols; ++x)
            {
                bool nonZero1 = false, nonZero2 = false;
                if (x < w1 && y < h1)
                {
                    nonZero1 = cv::norm(image1.at<cv::Vec3b>(y, x)) > 0;
                }
                nonZero2 = cv::norm(stitchedImage.at<cv::Vec3b>(y, x)) > 0;
                if (nonZero1 && nonZero2)
                {
                    overlapMask.at<uchar>(y, x) = 255;
                }
            }
        }

        // Find the bounding box of the overlap region
        int minX = stitchedImage.cols, maxX = 0;
        for (int y = 0; y < stitchedImage.rows; ++y)
        {
            for (int x = 0; x < stitchedImage.cols; ++x)
            {
                if (overlapMask.at<uchar>(y, x) == 255)
                {
                    if (x < minX)
                        minX = x;
                    if (x > maxX)
                        maxX = x;
                }
            }
        }

        // Blend images
        for (int y = 0; y < stitchedImage.rows; y++)
        {
            for (int x = 0; x < stitchedImage.cols; x++)
            {
                stitchedImage.at<cv::Vec3b>(y, x) = image1Expanded.at<cv::Vec3b>(y, x) * d1(x, minX, maxX) + stitchedImage.at<cv::Vec3b>(y, x) * d2(x, minX, maxX);
            }
        }
        break;
    }
    default:
    {
        throw std::invalid_argument("Unknown stitching method");
    }

    }
    
    return stitchedImage;
};