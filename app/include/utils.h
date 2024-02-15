/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This is header used for utils.cpp which contains functions for miscellaneous purposes. 
*/

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

/**
 * The function "myPrintVec" prints the elements of a vector to the standard output.
 * 
 * @param vec A reference to a std::vector object containing elements of type T.
 */
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

/**
 * The template function "sortIndices" sorts the indices of a vector based on the corresponding
 * values in the vector in ascending order by default or descending order if specified.
 * 
 * @tparam T The template parameter `T` represents the data type of the elements in the vector.
 * 
 * @param array A reference to the vector whose indices are to be sorted.
 * 
 * @param descending A boolean value indicating the sorting order. 
 * If `false` (default), sorting is done in ascending order. 
 * If `true`, sorting is done in descending order.
 * 
 * @return A vector of size_t representing the sorted indices of the input vector.
 */
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

void vizHist(cv::Mat hist, int numBins);

void displayImage(cv::Mat &img, bool normalize, bool onlyPositive);

cv::Mat myThresh(cv::Mat &img, int postThresh, int negThresh);

std::pair<double, double> myNormMat(cv::Mat &src, cv::Mat &dst);

void myNormMatInv(cv::Mat &src, cv::Mat &dst, std::pair<double, double> pars);

std::pair<double, double> myNormVec(std::vector<double> &src, std::vector<double> &dst);

void drawEdges(cv::Mat &src, cv::Mat &dst);

void getEdgeImage(cv::Mat &src, cv::Mat &dst);

cv::Mat myThreshWeighted(cv::Mat &img, int postThresh, int negThresh, double weightPercent);

cv::Mat getNoiseImage(int width, int height, int minVal, int maxVal);

#endif // UTILS_H

