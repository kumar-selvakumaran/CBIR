/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This is the feature extraction prograram which takes as in put the  feature 
 * extraction method and then creates a database of features,
 * each feature corresponding to a single image. 
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

// #include <faceDetect.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include<featureutils.h>

int main(int argc, char *argv[]) {

    std::string dataDir;
    std::string outputPath;
    std::string featureType;

    // check for sufficient arguments
    if( argc < 2) {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }

    dataDir = argv[1];
    featureType = argv[2];
    outputPath = argv[3];
    
    
    FeatureExtractor featureExtractor{FeatureExtractor(dataDir,  outputPath, featureType)};

    featureExtractor.computeFeatures();
    return(0);
}
