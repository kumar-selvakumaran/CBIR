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
#include <map>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include<featureutils.h>
#include<utils.h>
#include<hog.h>


bool FeatureExtractor::checkPaths(){
    DIR *dirp;
    dirp = opendir(imgdbdir.c_str());
    
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", imgdbdir);
        return false;
    }

    std::ofstream outputCsv(csvOutPath.c_str());

    // Check if file is open
    if (!outputCsv.is_open()) {
        std::cerr << "Error opening file: " << csvOutPath << std::endl;
        return false;
    }

    return true;
     
}

FeatureExtractor::FeatureExtractor(
    std::string inDir,
    std::string outPath,
    std::string featureMethodKey
){
    this->imgdbdir = inDir;     
    this->csvOutPath = outPath;
    this->featureComputer = getFeatureMethod(featureMethodKey); 

    bool status = checkPaths();

    if(!status)
        return;
}

bool FeatureExtractor::computeFeatures(){

    std::string buffer;
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;
    int progress = 0;

    
    dirp = opendir(imgdbdir.c_str());

    std::ofstream outputCsv(csvOutPath.c_str());

    while( (dp = readdir(dirp)) != NULL ) {
        // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif") ) {
            // build the overall filename
            buffer = imgdbdir;
            // std::cout<< " \n " << buffer << "\n" ;
            buffer += dp->d_name;

            //#################################################
            if(progress %10 ==0){
                std::cout<< " \n full buffer " << buffer << "\n";
            } 
            progress++;
            //#################################################

            //read the image
            cv::Mat dbim = cv::imread(buffer, cv::IMREAD_COLOR);
            // cv::Mat dbim{100, 100, CV_32FC3, cv::Scalar(1,1,1)};

            //compute the feature vector
            std::vector<std::vector<double>> features;

            features = featureComputer(dbim);
            //store feature in the feature db

            outputCsv << buffer << "\n";
            
            for(size_t i = 0; i < features.size() ; i++) {
                if(i > 0){
                    outputCsv << ",<SEP>,";
                }

                cv::Mat temp(features[i]);
                std::vector<double> feature = temp;  

                for (size_t j = 0; j < feature.size(); ++j) {
                    outputCsv << feature[j];
                    if (j < feature.size() - 1) {
                        outputCsv << ",";
                    }
                }
            }
            
            outputCsv << "\n";

            // cv::namedWindow("testim", 1);
            // cv::imshow("testim", testim); 
            // cv::waitKey(0);
            //####################
            // featureMap[buffer] = features;
            //#####################
        }
    }

    outputCsv.close();

    printf("Terminating\n");

    return true;
}

/*
Test functions to check passing function pointers as parameters in
constructor.
*/

featureMethod getFeatureMethod(std::string featureMethodKey){
    featureMethod featureComputer;
    if(featureMethodKey == "Baseline"){
        featureComputer = &baselineFeatures7x7; 
    } else if (featureMethodKey == "Histogram") {
        featureComputer = &histFeature;
    } else if (featureMethodKey == "upperLowerCropsHist"){
        featureComputer = &upperLowerCropsHist;
    } else if (featureMethodKey == "globalHog"){
        featureComputer = &globalHog;
    }
 
    else {
        std::cout << "\n FEATURE METHOD INPUTTED INCORRECTLY OR NOT AT ALL \n";
        featureComputer = &baselineFeatures7x7;
    }
    
    return featureComputer;
}

std::vector<std::vector<double>> baselineFeatures7x7(cv::Mat &src){

    src.convertTo(src, CV_32FC3);
    // cv::Mat testim(100, 100, CV_32FC3, cv::Scalar(1, 1, 1));

    int middleRowStart = (src.rows/2) - 3;
    int middleColStart = (src.cols/2) - 3;
    int middleRowEnd = middleRowStart + 7;
    int middleColEnd = middleColStart + 7;

    cv::Mat middleSlice;

    src(
        cv::Range(middleRowStart, middleRowEnd),
        cv::Range(middleColStart, middleColEnd)
        ).copyTo(middleSlice);

    std::vector<std::vector<double>> features;
    features.push_back(middleSlice.reshape(1,1));
    
    return features;
}

std::vector<std::vector<double>> histFeature(cv::Mat &src){

    cv::Mat hist;

    hist = makeHist(src, 8);    
    
    std::vector<std::vector<double>> features;
    features.push_back(hist.reshape(1,1));

    // std::cout <<"\n RETURNING HISTOGRAM\n";
    return features;

}

/*
This feature is a super-naive way to identify the kind of landscape.
It hopes to help compute outdoor images of a similar type. Eg:
blue sky + greenery, blue sky + infrastucture, yellow-red sky + infrastructure.
given an indoor image, it returns images of a similar backround

This feaature returns the upper and lower Crop of the image
*/
std::vector<std::vector<double>> upperLowerCropsHist(cv::Mat &src){
    
    cv::Mat histUpperCrop;
    cv::Mat histLowerCrop;

    int upperCropRowStart = 0;
    int upperCropColStart = 0;
    int upperCropRowEnd = (int)(src.rows/4);
    int upperCropColEnd = src.cols;

    cv::Mat upperCrop;

    src(
        cv::Range(upperCropRowStart, upperCropRowEnd),
        cv::Range(upperCropColStart, upperCropColEnd)
        ).copyTo(upperCrop);

    histUpperCrop = makeHist(upperCrop, 8); 

    int lowerCropRowStart = (int)(3*(src.rows/4));
    int lowerCropColStart = 0;
    int lowerCropRowEnd = src.rows;
    int lowerCropColEnd = src.cols;

    cv::Mat lowerCrop;

    src(
        cv::Range(lowerCropRowStart, lowerCropRowEnd),
        cv::Range(lowerCropColStart, lowerCropColEnd)
        ).copyTo(lowerCrop);


    histLowerCrop = makeHist(lowerCrop, 8); 

    std::vector<std::vector<double>> features;
    features.push_back(histUpperCrop.reshape(1,1));
    features.push_back(histLowerCrop.reshape(1,1));

    return features;

}

std::vector<std::vector<double>> globalHog(cv::Mat &src){

    cv::Mat hist;

    hog hogComputer(5, 160, -160, 16, 5);

    hist = hogComputer.computeGlobalHogV1(src);   
    // hist = hogComputer.computeGlobalHog(src);   
    
    std::vector<std::vector<double>> features;
    features.push_back(hist.reshape(1,1));

    return features;
}