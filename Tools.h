#pragma once

//First specify the computational environment. Other options are: AC_USE_GPU
#define AC_USE_CPU

//Then specify the library of use. Other options are: AC_WITH_VIENNACL (GPU), AC_WITH_CUDA (GPU)
#define AC_WITH_EIGEN
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <thread>
#include <cmath>
#include <csignal>
#include <mutex>
#include <chrono>

#include <Eigen/Dense>
#include "AlphaCore.h"

using namespace std;
using namespace cv;

void cap(float& f, float cap) {
    if (f > cap) {
        f = cap;
    }
}

void placeImage(cv::Mat &largeImage,  cv::Mat smallImage, int centerX, int centerY) {
    // Calculate the top-left position based on the center position
    int x = centerX - smallImage.cols / 2;
    int y = centerY - smallImage.rows / 2;

    // Ensure the small image can fit inside the large image at the given position
    if (x < 0 || y < 0 || x + smallImage.cols > largeImage.cols || y + smallImage.rows > largeImage.rows) {
        std::cerr << "The small image does not fit within the large image at the specified center position." << std::endl;
        return;
    }

    // Define the region of interest (ROI) in the large image
    cv::Rect roi(x, y, smallImage.cols, smallImage.rows);
    cv::Mat destinationROI = largeImage(roi);

    // Copy the small image into the ROI
    smallImage.copyTo(destinationROI);
}


float mapf(float s, float a1, float a2, float b1, float b2) {
    float output;
    output = (b1 + (((s - a1) * (b2 - b1)) / (a2 - a1)));
    return output;
}


Mat convertToGrayscale(const Mat &image) {
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    return grayImage;
}

vector<float> convertTo1DVector(const Mat &grayImage) {
    // Ensure the image is grayscale
    CV_Assert(grayImage.type() == CV_8UC1);

    // Create a 1D vector with the size of the total number of pixels
    vector<float> result;
    result.reserve(grayImage.total());

    // Iterate through the image and add pixel values to the vector
    for (int i = 0; i < grayImage.rows; ++i) {
        for (int j = 0; j < grayImage.cols; ++j) {
            result.push_back(mapf(static_cast<float>(grayImage.at<uchar>(i, j)), 0, 255, 0, 1));
        }
    }

    return result;
}

void addTextToImage(cv::Mat& image, const std::string& text,
    const cv::Point& location, int fontFace,
    double fontScale, const cv::Scalar& color,
    int thickness = 2, int lineType = cv::LINE_AA) {
    cv::putText(image, text, location, fontFace, fontScale, color, thickness, lineType);
}

void replaceQuadrant(cv::Mat& image1, const cv::Mat& image2, int quadrant) {
    // Check that the two images have the same size and type
    if (image1.size() != image2.size() || image1.type() != image2.type()) {
        std::cerr << "Images must have the same size and type." << std::endl;
        return;
    }

    // Determine the center of the image
    int centerX = image1.cols / 2;
    int centerY = image1.rows / 2;

    // Define the ROI (Region of Interest) based on the specified quadrant
    cv::Rect roi;
    switch (quadrant) {
        case 1: // Top-left quadrant
            roi = cv::Rect(0, 0, centerX, centerY);
            break;
        case 2: // Top-right quadrant
            roi = cv::Rect(centerX, 0, centerX, centerY);
            break;
        case 3: // Bottom-left quadrant
            roi = cv::Rect(0, centerY, centerX, centerY);
            break;
        case 4: // Bottom-right quadrant
            roi = cv::Rect(centerX, centerY, centerX, centerY);
            break;
        default:
            std::cerr << "Invalid quadrant specified. Must be 1, 2, 3, or 4." << std::endl;
            return;
    }

    // Replace the specified quadrant in image1 with the corresponding pixels from image2
    image2(roi).copyTo(image1(roi));
}

void drawCross(cv::Mat& image, const cv::Point& center, int length,
               const cv::Scalar& color, int thickness = 2, int lineType = cv::LINE_AA) {
    // Calculate the start and end points for the horizontal and vertical lines
    cv::Point horizontalStart(center.x - length / 2, center.y);
    cv::Point horizontalEnd(center.x + length / 2, center.y);
    
    cv::Point verticalStart(center.x, center.y - length / 2);
    cv::Point verticalEnd(center.x, center.y + length / 2);
    
    // Draw the horizontal line
    cv::line(image, horizontalStart, horizontalEnd, color, thickness, lineType);
    
    // Draw the vertical line
    cv::line(image, verticalStart, verticalEnd, color, thickness, lineType);
}
