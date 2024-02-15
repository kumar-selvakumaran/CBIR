/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * purpose : This file is the header for featurutils.cpp which contains the utility classes and functions for feature extraction
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>

#ifndef FEATUREUTILS_H
#define FEATUREUTILS_H

// #include<featureutils.h>

typedef std::vector<std::vector<double>> (*featureMethod)(cv::Mat &src); 

/*
This class is contains all the required feature extraction methods, and auxillary data
*/
class FeatureExtractor{
    private:
        // Private member variable
        std::string imgdbdir;
        std::string csvOutPath;
        std::string featureName;
        featureMethod featureComputer;
        bool useStridedFeatures;
        
        bool checkPaths();

    public:
        
        FeatureExtractor(std::string inDir, std::string outPath, std::string featureMethodKey, bool useStridedFeatures);
    
        bool computeFeatures();
};

featureMethod getFeatureMethod(std::string featureMethodKey);

std::vector<std::vector<double>> baselineFeatures7x7(cv::Mat &src);

std::vector<std::vector<double>> histFeature(cv::Mat &src);

std::vector<std::vector<double>> upperLowerCropsHist(cv::Mat &src);

std::vector<std::vector<double>> globalHog(cv::Mat &src);

std::vector<std::vector<double>> globalHogandColour(cv::Mat &src);

void readDnnFeatures();

std::vector<std::vector<double>> slidingExtraction (cv::Mat &src, featureMethod featureSlide, int kernelSize);

#endif // FEATUREUTILS_H
