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

#include <featureutils.h>
#include <distanceutils.h>
#include <utils.h>

#ifndef HOG_H
#define HOG_H


typedef cv::Vec<double, 5 > Vec5d;

typedef cv::Vec<double, 7 > Vec7d;

typedef cv::Vec<double, 8 > Vec8d;

typedef cv::Vec<double, 9 > Vec9d;

typedef cv::Vec<double, 10 >Vec10d;

class hog{
    private:
        int blurrKernelSize;
        int threshPositive;
        int threshNegative;
        int globalIntensityBins;
        int globalOrientationBins;


    public:
        hog(int blurrKernelSize,
            int threshPositive,
            int theshNegative,
            int globalIntensityBins,
            int globalOrientationBins);
        
        
        cv::Mat computeGrad(cv::Mat &src, bool isX);
        cv::Mat computeMagnitude(cv::Mat &gradX, cv::Mat &gradY);
        cv::Mat computeOrientation(cv::Mat &gradX, cv::Mat &gradY);
        cv::Mat computeGlobalHogV1(cv::Mat &src);   
        cv::Mat computeGlobalHog(cv::Mat &src);
        
};


#endif // HOG_H

