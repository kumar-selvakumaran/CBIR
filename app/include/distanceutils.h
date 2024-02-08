// Purpose : Contains the declearations of all the functions used.

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <map>

#ifndef DISTANCEUTILS_H
#define DISTANCEUTILS_H

// #include<distanceutils.h>

typedef double (*distanceMethod)(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2); 

/*
Given a directory on the command line, scans through the directory for image
files.
Prints out the full path name for each file. This can be used as an argument to
fopen or to cv::imread.
*/


/*
The class takes is constructed with the distance method and the directory of the 
feature csv file.
It will compute distances among the different feature vectors from the csv file.

- members variables:
    - featurePath : Path to csv files containing the feature vectors .
    - distanceMethod : Function pointer corresponding to the distance metric to be used.
    - TargePath: Path to the target image.

- member functions:
    - private pathOpened(std::string dirPath[] ) -> boolean 
        returns if path opened successfully or not.

    - public computeDistances() -> boolean
        calls dirOpened(), and if successfully,  iterates over the feature vectors and 
        computes the distances between the target image's vector and resto of the vectors.

    - public getSimilarImages(int numImages) -> std::vector<std::string> finds the top K similar

*/

class DistanceFinder{
    private:
        // Private member variable
        std::string featurePath;
        std::string targetPath;
        std::string distanceName;
        distanceMethod distanceComputer;
        std::vector<double> distances;
        std::map<std::string, std::vector<std::vector<double>>> featureMap;
        std::vector<double> distancesSorted;
        std::vector<std::string> imPathsDistSorted;
        bool pathOpened(std::string dirname);
        
    public:
    
        // Constructor
        DistanceFinder(std::string featurePath, std::string targetPath, std::string distanceMethodKey);
        // Destructor
        // ~DistanceFinder();

        // Member function declarations
        bool computeDistances();
        
        // load features into a hash map to avoid looping to find target
        bool loadFeatures();

        // gets similar images by finding 'numImages' closest feature vectors, and saving
        //corresponding images in ../bin/similarImages/
        bool getSimilarImages(int numImages, std::string mode);

};


double euclideanDistance(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2);

double HistogramIntersection(std::vector<std::vector<double>> vec1, std::vector<std::vector<double>> vec2);

distanceMethod getDistanceMethod(std::string distanceMethodKey);

bool toMinOrMax(std::string &distanceKey);

#endif // DISTANCEUTILS_H
