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
    std::string targetPath;
    int numImages;
    char control = ' ';
    std::string buffer;
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;
    // check for sufficient arguments
    if( argc < 2) {
        printf("usage: %s <feature path (string)> <feature type (string)> <target path (string)> <number of similar images (int)>\n", argv[0]);
        exit(-1);
    }
    // get the arguements path
    while(featurePath!=""){
        featurePath = argv[1];
        distanceType = argv[2];
        targetPath = argv[3];
        numImages = atoi(argv[4]);
        

        // distanceMethod funcptr{&subtracttest};

        // distanceMethod funcptr{getfunc[distanceType]}

        // DistanceFinder dfObject{DistanceFinder(featurePath,  targetPath, funcptr)};

        // dfObject.computeDistances();

        // dfObject.getSimilarImages(numImages);
        
        std::cout << "\n\nenter path of new image query : \t";
        std::cin >> featurePath;
        if(featurePath=="")
            break;
        else{
            std::cout <<"\n\n how many similar images should be found? : \t";
            std::cin >> numImages;
        }
    }
    return(0);
}
