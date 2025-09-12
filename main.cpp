#include <featureDetection.h>
#include <warping.h>
#include <matplot/matplot.h>



int main()
{
    ImageFeatures features1_1, features1_2, features2_1, features2_2, features3_1, features3_2;
    ImageFeatures features1_1_orb, features1_2_orb, features2_1_orb, features2_2_orb, features3_1_orb, features3_2_orb;
    cv::Mat image1_1, image1_2, image2_1, image2_2, image3_1, image3_2;

    // Load images
    image1_1 = load_image("../images/1_1.jpg");
    image1_2 = load_image("../images/1_2.jpg");

    image2_1 = load_image("../images/2_1.jpg");
    image2_2 = load_image("../images/2_2.jpg");

    image3_1 = load_image("../images/3_1.jpg");
    image3_2 = load_image("../images/3_2.jpg");

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
    matplot::title("Feature Matching Time by Extraction Method");
    matplot::save("../plots/matching_time.jpg");

    // Plot histograms of match distances by extraction method for each image
    // Scale SIFT and ORB distances to be comparable by using PDF
    //Equally pad the ORB bins to match the SIFT bins
    matplot::figure();
    matplot::subplot(3, 1, 1);
    auto siftHist1 = matplot::hist(matches1.distances, matplot::histogram::normalization::probability);
    matplot::hold(matplot::on);
    auto orbHist1 = matplot::hist(matches1_orb.distances, matplot::histogram::normalization::probability);
    orbHist1->bin_edges(siftHist1->bin_edges());
    matplot::ylabel("Frequency");
    matplot::subplot(3, 1, 2);
    auto siftHist2 = matplot::hist(matches2.distances, matplot::histogram::normalization::probability);
    matplot::hold(matplot::on);
    auto orbHist2 = matplot::hist(matches2_orb.distances, matplot::histogram::normalization::probability);
    orbHist2->bin_edges(siftHist2->bin_edges());
    matplot::ylabel("Frequency");
    matplot::xlabel("Match Distance");
    matplot::subplot(3, 1, 3);
    auto siftHist3 = matplot::hist(matches3.distances, matplot::histogram::normalization::probability);
    matplot::hold(matplot::on);
    auto orbHist3 = matplot::hist(matches3_orb.distances, matplot::histogram::normalization::probability);
    orbHist3->bin_edges(siftHist3->bin_edges());
    matplot::ylabel("Frequency");
    matplot::title("Histogram of Match Distances by Extraction Method (Normalized)");
    matplot::save("../plots/match_distances.jpg");
    //TODO: fix bin sizes

    // Estimate homography for each image pair, with varying reprojection thresholds
    HomographyEstimation homography1_3 = estimateHomography(features1_1.keypoints, features1_2.keypoints, matches1, 3.0);
    HomographyEstimation homography2_3 = estimateHomography(features2_1.keypoints, features2_2.keypoints, matches2, 3.0);
    HomographyEstimation homography3_3 = estimateHomography(features3_1.keypoints, features3_2.keypoints, matches3, 3.0);

    HomographyEstimation homography1_5 = estimateHomography(features1_1.keypoints, features1_2.keypoints, matches1, 5.0);
    HomographyEstimation homography2_5 = estimateHomography(features2_1.keypoints, features2_2.keypoints, matches2, 5.0);
    HomographyEstimation homography3_5 = estimateHomography(features3_1.keypoints, features3_2.keypoints, matches3, 5.0);

    HomographyEstimation homography1_10 = estimateHomography(features1_1.keypoints, features1_2.keypoints, matches1, 10.0);
    HomographyEstimation homography2_10 = estimateHomography(features2_1.keypoints, features2_2.keypoints, matches2, 10.0);
    HomographyEstimation homography3_10 = estimateHomography(features3_1.keypoints, features3_2.keypoints, matches3, 10.0);

    // Plot bar char of numbers of inliers by reprojection threshold for each image pair for each threshold
    std::vector<std::vector<int>> numInliersByThreshold = {{homography1_3.numInliers, homography1_5.numInliers, homography1_10.numInliers}, 
                                                            {homography2_3.numInliers, homography2_5.numInliers, homography2_10.numInliers}, 
                                                            {homography3_3.numInliers, homography3_5.numInliers, homography3_10.numInliers}};
    matplot::figure();
    matplot::bar(numInliersByThreshold);
    matplot::ylabel("# Inliers");
    matplot::xlabel("Reprojection Threshold");
    matplot::gca()->x_axis().ticklabels({"3.0", "5.0", "10.0"});
    matplot::title("Number of Inliers by Reprojection Threshold");
    matplot::save("../plots/num_inliers.jpg");

    // Plot bar chart of estimation time by reprojection threshold for each image pair for each threshold
    std::vector<std::vector<double>> estimationTimeByThreshold = {{homography1_3.estimationTimeMs, homography1_5.estimationTimeMs, homography1_10.estimationTimeMs}, 
                                                                  {homography2_3.estimationTimeMs, homography2_5.estimationTimeMs, homography2_10.estimationTimeMs}, 
                                                                  {homography3_3.estimationTimeMs, homography3_5.estimationTimeMs, homography3_10.estimationTimeMs}};
    matplot::figure();
    matplot::bar(estimationTimeByThreshold);
    matplot::ylabel("Estimation Time (ms)");
    matplot::xlabel("Reprojection Threshold");
    matplot::gca()->x_axis().ticklabels({"3.0", "5.0", "10.0"});
    matplot::title("Homography Estimation Time by Reprojection Threshold");
    matplot::save("../plots/estimation_time.jpg");


    //TODO: stitch quality by threshold
    //TODO: stitch quality by stitching method

    return 0;
}