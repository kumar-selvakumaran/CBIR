// Purpose : Contains the declearations of all the functions used.

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <map>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

#ifndef UTILS_H
#define UTILS_H

// #include<utils.h>


template<typename T>
void myPrintVec(std::vector<T> &vec) {
    std::cout << "\t[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\t" << std::endl;
}

template <typename T>
std::vector<size_t> sortIndices(std::vector<T> &array, bool descending = false) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(),
                    indices.end(),
                    [&array](size_t val1, size_t val2){return array[val1] < array[val2];}
                    );
    if(descending){
        reverse(indices.begin(), indices.end());
    }
    return indices;
}


std::string myMatType(cv::Mat &src);
void printmat(cv::Mat &src, int vizdim);
int earliestDecPos(double num);

cv::Mat makeHist(cv::Mat &src, int numBins);

// int magnitude(Mat &sx, Mat &sy, Mat &dst)

void displayImage(cv::Mat &img);

cv::Mat myThresh(cv::Mat &img, int postThresh, int negThresh);

std::pair<double, double> myNormMat(cv::Mat &src, cv::Mat &dst);

void myNormMatInv(cv::Mat &src, cv::Mat &dst, std::pair<double, double> pars);

std::pair<double, double> myNormVec(std::vector<double> &src, std::vector<double> &dst);

void emphEdges(cv::Mat &src, cv::Mat &dst);

void getEdgeImage(cv::Mat &src, cv::Mat &dst);

// void slide(cv::Mat &src, cv::Mat &filter, cv::);

#endif // UTILS_H

