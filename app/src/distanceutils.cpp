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

#include<distanceutils.h>

bool DistanceFinder::pathOpened(std::string dirpath){
    std::ifstream file(dirpath.c_str());
    if (!(file.is_open())) {
        std::cerr << "Error opening file: " << dirpath << std::endl;
        return false;
    }
    return true;
}

DistanceFinder::DistanceFinder(
    std::string featurePath,
    std::string targetPath,
    distanceMethod distanceComputer
    ){
        bool status = pathOpened(featurePath);

        if(!status){
            printf("Cannot open features %s\n", featurePath);
            return;
        }

        this->featurePath = featurePath;
        this->targetPath = targetPath;
        this->distanceComputer = distanceComputer;   
}

/*
read the feature vectors csv file, from this->featurePath, and
compute distances to the vector corresponding to this->targetPath
save distances in a disances.csv 
*/

bool DistanceFinder::loadFeatures(){   
    std::ifstream featurecsv(featurePath);
    std::string line;
    int lineNum = 0;
    std::string imPath;
    while(std::getline(featurecsv, line)){
        std::vector<float> featureVec;
        std::stringstream ss(line);
        std::string element;

        if(lineNum%2!=0){
            while(std::getline(ss, element, ',')){
                float value;
                std::istringstream(element) >> value;
                featureVec.push_back(value);
            }
            cv::Mat featureMat(featureVec);
            featureMap[imPath] = featureMat;
        }
        else{
            imPath = line;
        }
        lineNum+=1;
        //##############
        // break;
        //##############
    
    }

    return true;
}

bool DistanceFinder::computeDistances(){

    return true;
}


bool DistanceFinder::getSimilarImages(int numImages){
    return true;
}

std::vector<float> euclideanDistance(cv::Mat &src, cv::Mat &target){
    
}


