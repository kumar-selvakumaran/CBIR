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

// void FeatureExtractor::featuresToCsv(const std::vector<std::vector<float>>& data) {
//     // Open a file stream for writing
//     std::ofstream outputCsv(csvOutPath);

//     // Iterate over the 2D vector
//     for (const auto& row : data) {
//         for (size_t i = 0; i < row.size(); ++i) {
//             outputCsv << row[i]; // Write the current element
//             if (i < row.size() - 1) {
//                 outputCsv << ","; // Add comma for separation except for the last element
//             }
//         }
//         outputCsv << "\n"; // End of row
//     }
//     // Close the file
//     outputCsv.close();
// }

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
            // std::cout<< " \n full buffer " << buffer << "\n" ;


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
    } else if (featureMethodKey == "upperLowerQuartersHist"){
        featureComputer = &upperLowerQuartersHist;
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

This feaature returns the upper and lower quarter of the image
*/
std::vector<std::vector<double>> upperLowerQuartersHist(cv::Mat &src){
    
    cv::Mat histUpperQuarter;
    cv::Mat histLowerQuarter;

    int upperQuarterRowStart = 0;
    int upperQuarterColStart = 0;
    int upperQuarterRowEnd = (int)(src.rows/4);
    int upperQuarterColEnd = src.cols;

    cv::Mat upperQuarter;

    src(
        cv::Range(upperQuarterRowStart, upperQuarterRowEnd),
        cv::Range(upperQuarterColStart, upperQuarterColEnd)
        ).copyTo(upperQuarter);

    histUpperQuarter = makeHist(upperQuarter, 8); 

    int lowerQuarterRowStart = (int)(3*(src.rows/4));
    int lowerQuarterColStart = 0;
    int lowerQuarterRowEnd = src.rows;
    int lowerQuarterColEnd = src.cols;

    cv::Mat lowerQuarter;

    src(
        cv::Range(lowerQuarterRowStart, lowerQuarterRowEnd),
        cv::Range(lowerQuarterColStart, lowerQuarterColEnd)
        ).copyTo(lowerQuarter);


    histLowerQuarter = makeHist(lowerQuarter, 8); 

    std::vector<std::vector<double>> features;
    features.push_back(histUpperQuarter.reshape(1,1));
    features.push_back(histLowerQuarter.reshape(1,1));

    return features;

}