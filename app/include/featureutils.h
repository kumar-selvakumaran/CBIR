// Purpose : Contains the declearations of all the functions used.

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
Given a directory on the command line, scans through the directory for image
files.
Prints out the full path name for each file. This can be used as an argument to
fopen or to cv::imread.
*/


/*
The class takes is constructed with the extraction method and the directory of the 
image database, and the savepath of the csv file.
It will compute features of the images in the database and store the feature vectors
in a csv file at the location specifiesd

- members variables:
    - imdbDir : path the directory of the image database.
    - csvOutPath : Path the the csv output file.

- member functions:
        - private dirOpened(std::string dirPath ) -> boolean 
        returns if dir path opened successfully or not

    - private saveCsv(const std::vector<std::vector<float>> encodings) -> boolean
        Takes the feature array as input and saves it as a csv. Returns True, if saved
        successfully.
    - public computeFeatures() -> boolean
        calls dirOpened(), and if successful,  iterates over the image database and 
        finds the feature embeddings of the different images. then calls saveCsv with the 
        feature embeddings.
*/
class FeatureExtractor{
    private:
        // Private member variable
        std::string imgdbdir;
        std::string csvOutPath;
        featureMethod featureComputer;
        
        bool checkPaths();

        // void featuresToCsv(const std::vector<std::vector<float>>& data);

    public:
        // Constructor
        FeatureExtractor(std::string inDir, std::string outPath, std::string featureMethodKey);
        // Destructor
        // ~FeatureExtractor();

        // Member function declarations
        bool computeFeatures();
};

featureMethod getFeatureMethod(std::string featureMethodKey);

std::vector<std::vector<double>> baselineFeatures7x7(cv::Mat &src);
std::vector<std::vector<double>> histFeature(cv::Mat &src);
std::vector<std::vector<double>> upperLowerQuartersHist(cv::Mat &src);

#endif // FEATUREUTILS_H
