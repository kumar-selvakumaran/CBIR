/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This is header used for distanceutils.cpp which contains or imports all the functions and classes required for distance computation. 
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <map>

#include <featureutils.h>

#ifndef DISTANCEUTILS_H
#define DISTANCEUTILS_H

// #include<distanceutils.h>

typedef double (*distanceMethod)(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2); 

typedef double (*rawDistanceMethod)(cv::Mat mat1, cv::Mat mat2); 


/**
 * The class "DistanceFinder" implements functionality to compute distances between a target image and
 * a set of feature vectors, as well as finding similar or dissimilar images based on these distances.
 */
class DistanceFinder{
    private:
        // Private member variable
        std::string featurePath;
        std::string targetPath;
        std::string distanceName;
        distanceMethod distanceComputer;
        featureMethod targetFeatureComputer;
        std::string targetFeatureName;
        std::vector<double> distances;
        std::map<std::string, std::vector<std::vector<double>>> featureMap;
        std::vector<double> distancesSorted;
        std::vector<std::string> imPathsDistSorted;
        bool pathOpened(std::string dirname);
        
    public:
    
        DistanceFinder(std::string featurePath, std::string targetPath, std::string distanceMethodKey, std::string targetFeaturekey);

        bool computeDistances();
        
        bool loadFeatures();

        bool getSimilarImages(int numImages, std::string mode);
        bool getDisSimilarImages(int numImages, std::string mode);

};

double rawEuclideanDistance(cv::Mat mat1, cv::Mat mat2);

double rawHistogramIntersection(cv::Mat hist1, cv::Mat hist2);

double simpeEuclideanDistance(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2);

double HistogramIntersection(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2);

double upperLowerCropHistIntersect(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2);

distanceMethod getDistanceMethod(std::string distanceMethodKey);

double stridedDistanceComputer(std::vector<std::vector<double>> target, std::vector<std::vector<double>> vec2, rawDistanceMethod distanceGetter, bool maximize);

bool toMinOrMax(std::string &distanceKey);

#endif // DISTANCEUTILS_H
