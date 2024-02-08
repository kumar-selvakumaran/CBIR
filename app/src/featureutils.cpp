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

void FeatureExtractor::featuresToCsv(const std::vector<std::vector<float>>& data) {
    // Open a file stream for writing
    std::ofstream outputCsv(csvOutPath);

    // Iterate over the 2D vector
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outputCsv << row[i]; // Write the current element
            if (i < row.size() - 1) {
                outputCsv << ","; // Add comma for separation except for the last element
            }
        }
        outputCsv << "\n"; // End of row
    }
    // Close the file
    outputCsv.close();
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
    
    dirp = opendir(imgdbdir.c_str());

    std::ofstream outputCsv(csvOutPath.c_str());

    // //########################
    // std::ofstream maptest("maptest");
    // std::map<std::string, std::vector<float>> featureMap;
    //########################

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
            std::vector<float> features;

            features = featureComputer(dbim);
            //store feature in the feature db

            outputCsv << buffer << "\n";

            for (size_t i = 0; i < features.size(); ++i) {
                outputCsv << features[i];
                if (i < features.size() - 1) {
                    outputCsv << ",";
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
    }
 
    else {
        std::cout << "\n FEATURE METHOD INPUTTED INCORRECTLY OR NOT AT ALL \n";
        featureComputer = &baselineFeatures7x7;
    }
    
    return featureComputer;
}

std::vector<float> baselineFeatures7x7(cv::Mat &src){

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

    std::vector<float> features = middleSlice.reshape(1,1);
    
    return features;
}

std::vector<float> histFeature(cv::Mat &src){
    cv::Mat hist;
    // NUMBER OF BINS
    int histsize = 8;
    // int max = 0;
    hist = cv::Mat::zeros( cv::Size( histsize, histsize ), CV_32FC1 );
    // std::cout << "\n computing histogram \n";
    // max = 0;

    for( int i=0;i<src.rows;i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++) {

            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;
            float r = R / divisor;
            float g = G / divisor;

            int rindex = (int)( r * (histsize - 1) + 0.5 );
            int gindex = (int)( g * (histsize - 1) + 0.5 );

            hist.at<float>(rindex, gindex)++;

        //   float newvalue = hist.at<float>(rindex, gindex);
        //   max = newvalue > max ? newvalue : max;
        }
    }

    hist /= (src.rows * src.cols);

    std::vector<float> features = hist.reshape(1,1);

    // std::cout <<"\n RETURNING HISTOGRAM\n";
    return features;

}

