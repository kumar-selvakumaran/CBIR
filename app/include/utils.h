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

#include<utils.h>


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
std::vector<size_t> sortIndices(std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(),
                    indices.end(),
                    [&array](size_t val1, size_t val2){return array[val1] < array[val2];}
                    );
    return indices;
}


std::vector<float> myMatToVec(cv::Mat &src);
std::string myMatType(cv::Mat &src);
void printmat(cv::Mat &src, int vizdim);

#endif // UTILS_H
