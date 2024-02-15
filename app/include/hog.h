/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This is header used for hog.cpp which contains or imports all the class and functions required to implement hog. 
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

#include <featureutils.h>
#include <distanceutils.h>
#include <utils.h>

#ifndef HOG_H
#define HOG_H


typedef cv::Vec<double, 5 > Vec5d;

/**
 * The class "hog" represents a Histogram of Oriented Gradients (HOG) feature extractor.
 * It provides methods to compute gradient, magnitude, orientation, and HOG features from an input image.
 */
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
        
};


#endif // HOG_H

