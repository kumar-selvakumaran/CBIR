/*
This progrogam stores the different feature extraction functions
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>

// #include <featureExtraction.h> // YET TO MAKE
// #include <faceDetect.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>


#include <utils.h>

std::vector<float> myMatToVec(cv::Mat &src){
    std::vector<float> vecMat;
    if (src.isContinuous()) {
        // array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
        vecMat.assign((float*)src.data, (float*)src.data + src.total()*src.channels());
    } else {
        for (int i = 0; i < src.rows; ++i) {
            vecMat.insert(vecMat.end(), src.ptr<float>(i), src.ptr<float>(i)+src.cols*src.channels());
        }
    }

    return vecMat;
}

std::string myMatType(cv::Mat &src) {
    int type = src.type();
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void printmat(cv::Mat &src, int vizdim)
{
    int xMax = std::min(90, src.rows);
    int xMin = xMax - std::min(vizdim, src.rows);
    int yMax = std::min(90, src.cols);
    int yMin = yMax - std::min(vizdim, src.cols);
    cv::Mat vizslice(src(cv::Range(xMin, xMax), cv::Range(yMin, yMax)));
    std::cout << "\nmatrix chunk : \n" << format(vizslice, cv::Formatter::FMT_NUMPY );
    std::cout << "\nchannels : " << src.channels() << " type : " << myMatType(src) << "\n";
    
}



