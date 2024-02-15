/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * the driver file for distance computation and displaying the most similar images
*/

/*
This is the feature extraction prograram which takes as in put the  feature 
extraction method and then creates a database of features,
each feature corresponding to a single image. 
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
#include<distanceutils.h>

//#### only for testing addtest/subtracttest
#include<featureutils.h>
//##########################################


int main(int argc, char *argv[]) {
    
    std::string featurePath;
    std::string distanceType;
    std::string featureType;
    std::string targetPath;
    std::string dataDir{"../data/olympus/olympus/"};
    std::string stride;
    bool useStride = false;
    int numImages;


    // check for sufficient arguments
    if( argc < 2) {
        printf("usage: %s <feature path (string)> <Distance type (string)> <target path (string)> <number of similar images (int)>\n", argv[0]);
        exit(-1);
    }
    
    featurePath = argv[1];
    distanceType = argv[2];
    featureType = argv[3];
    targetPath = argv[4];
    numImages = atoi(argv[5]);

    try{
        stride = argv[4];
        useStride = true;
    }
    // get the arguements path
    while(featurePath!=""){
        std::cout << "\n\nREAD THE ARGUEMENTS : \t";

        if(useStride == true){
            // it is the only distance metric supported for strided features
            distanceType = "stridedEuclideanDistance";
        }
        DistanceFinder dfObject{DistanceFinder(featurePath,  targetPath, distanceType, featureType)};

        std::cout << "\n\nINITIALZIED THE OBJECTS : \t";
        
        dfObject.computeDistances();

        std::cout << "\n\nCOMPUTED THE DISTANCES: \t";

        dfObject.getSimilarImages(numImages, "show");
        
        std::cout << "\n\nenter path of new image query : \t";
        std::string imgName;
        std::cin >> imgName;
        if(imgName=="")
            break;
        else{
            std::cout <<"\n\n how many similar images should be found? : \t";
            std::string numstr;
            std::cin >> numstr;
            numImages = atoi(numstr.c_str());
        }
        targetPath = dataDir + imgName;
    }
    return(0);
}
