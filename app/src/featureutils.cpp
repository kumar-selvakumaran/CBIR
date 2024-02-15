/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This file contains the utility classes and functions for feature extraction.
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>

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


/**
 * The function checks if a directory can be opened and if a file can be created and opened for
 * writing.
 * 
 * @return a boolean value.
 */
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
/**
 * The FeatureExtractor constructor initializes the object with input and output directories, a feature
 * method key, and a flag for using strided features. The feature method key is used to initialize a feau
 * 
 * @param inDir The input directory where the images are located.
 * @param outPath The 'outPath' parameter is a string that represents the path where the output CSV
 * file will be saved.
 * @param featureMethodKey The 'featureMethodKey' parameter is a string that represents the key or
 * identifier for the feature extraction method to be used. It is used to retrieve the appropriate
 * feature extraction method from the 'getFeatureMethod' function.
 * @param useStridedFeatures A boolean flag indicating whether to use strided features or not.
 * 
 * @return If the 'status' variable is 'false', then nothing is being returned. If the 'status'
 * variable is 'true', then nothing is being returned either.
 */
FeatureExtractor::FeatureExtractor(
    std::string inDir,
    std::string outPath,
    std::string featureMethodKey,
    bool useStridedFeatures
){
    this->imgdbdir = inDir;     
    this->csvOutPath = outPath;
    this->featureComputer = getFeatureMethod(featureMethodKey); 
    this->featureName = featureMethodKey;
    this->useStridedFeatures = useStridedFeatures;


    bool status = checkPaths();

    if(!status)
        return;
}

/**
 * The computeFeatures function reads images from a directory, computes their feature vectors using a
 * the feaature extractor specified by the feaure method key passed to the constructor, and stores the 
 * feature vectors in a CSV file. It also uses a strided feature extraction if prompted. If each image has
 * multiple features, then they are written on the like corresponding to that particular image and are 
 * seperated by a <SEP>.
 * 
 * @return a boolean value, which is always true.
 */
bool FeatureExtractor::computeFeatures(){

    std::string buffer;
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;
    int progress = 0;

    if(featureName == "Resnet"){
        readDnnFeatures();
        return true;
    }
    
    dirp = opendir(imgdbdir.c_str());

    std::ofstream outputCsv(csvOutPath.c_str());

    while( (dp = readdir(dirp)) != NULL ) {
        if( strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif") ) {

            buffer = imgdbdir;
            buffer += dp->d_name;

            if(progress %10 ==0){
                std::cout<< " \n full buffer " << buffer << "\n";
            } 
            progress++;

            cv::Mat dbim = cv::imread(buffer, cv::IMREAD_COLOR);

            std::vector<std::vector<double>> features;

            if (useStridedFeatures == false){
                features = featureComputer(dbim);
            }
            else if(useStridedFeatures == true){
                int kernelSize = dbim.size().width / 3;
                features = slidingExtraction(dbim, featureComputer, kernelSize);
            }
            
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
        }
    }

    outputCsv.close();

    printf("Terminating\n");

    return true;
}

/**
 * The function "getFeatureMethod" returns a custom datatype which is function pointer which is used as a 
 * template for all feature extraction functions.
if the key is not recognized, it returns the baseline feature method.
 * 
 * @param featureMethodKey The parameter 'featureMethodKey' is a string that represents the key or name
 * of a feature method.
 * 
 * @return a pointer of featureMethod object.
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
    } else if (featureMethodKey == "globalHogandColour"){
        featureComputer = &globalHogandColour;
    }
 
    else {
        std::cout << "\n FEATURE METHOD INPUTTED INCORRECTLY OR NOT AT ALL \n";
        featureComputer = &baselineFeatures7x7;
    }
    
    return featureComputer;
}



/**
 * Task 1:
 * 
 * The function "baselineFeatures7x7" extracts features from the middle 7x7 region of the input image.
 * 
 * @param src The parameter 'src' is the input image from which features are to be extracted. It should 
 * be of type 'cv::Mat'.
 * 
 * @return a vector of vectors of type double containing the extracted features. Each inner vector 
 * represents a row of features.
 */
std::vector<std::vector<double>> baselineFeatures7x7(cv::Mat &src){

    src.convertTo(src, CV_32FC3);

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


/**
 * The function "histFeature" computes the histogram features from the input image.
 * 
 * @param src The parameter 'src' is the input image from which histogram features are computed. It 
 * should be of type 'cv::Mat'.
 * 
 * @return a vector of vectors of type double containing the computed histogram features. Each inner 
 * vector represents a row of histogram features.
 */
std::vector<std::vector<double>> histFeature(cv::Mat &src){

    cv::Mat hist;

    hist = makeHist(src, 16);    
    
    std::vector<std::vector<double>> features;
    features.push_back(hist.reshape(1,1));

    return features;

}

/**
 * The function "upperLowerCropsHist" extracts features representing the upper and lower crops of 
 * the input image and returns histograms of these crops.
 * 
 * 
 * This feature is a super-naive way to identify the kind of landscape. It hopes to help compute outdoor
 * images of a similar type. Eg: blue sky + greenery, blue sky + infrastucture, yellow-red
 * sky + infrastructure. given an indoor image, it returns images of a similar backround.
 * 
 * @param src The parameter 'src' is the input image from which features are to be extracted. It should 
 * be of type 'cv::Mat'.
 * 
 * @return a vector of vectors of type double containing histograms representing the upper and lower 
 * crops of the input image. Each inner vector represents a row of histogram values.
 *  */
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

/**
 * The function "globalHog" computes the Histogram of Oriented Gradients (HOG) features globally from 
 * the input image.
 * 
 * @param src The parameter 'src' is the input image from which HOG features are to be computed. It 
 * should be of type 'cv::Mat'.
 * 
 * @return a vector of vectors of type double containing the computed HOG features. It returns a single vector,
 * it is "unsqueezed" to have an extra dimention to ensure uniformity in feature extraction methods.
 */
std::vector<std::vector<double>> globalHog(cv::Mat &src){

    cv::Mat hist;

    int chromBins = 8;
    int orBins = 5;

    hog hogComputer(5, 2, -2, chromBins, orBins);

    hist = hogComputer.computeGlobalHogV1(src);    
    
    std::vector<std::vector<double>> features;
    features.push_back(hist.reshape(1,1));

    return features;
}

/**
 * The function "globalHogandColour" computes the Histogram of Oriented Gradients (HOG) features 
 * globally and color histogram features from the input image. The texture features are concatenated
 * with the colour histogram features to make a single vector.
 * 
 * @param src The parameter 'src' is the input image from which HOG and color histogram features are 
 * to be computed. It should be of type 'cv::Mat'.
 * 
 * @return a vector of vectors of type double containing the computed HOG and color histogram features. 
 * Each inner vector represents a row of features. The first row corresponds to HOG features, and the 
 * second row corresponds to color histogram features.
 */
std::vector<std::vector<double>> globalHogandColour(cv::Mat &src){
    cv::Mat histHog;
    cv::Mat histCol;

    int chromBins = 2;
    int orBins = 5;

    hog hogComputer(5, 140, -140, chromBins, orBins);

    histHog = hogComputer.computeGlobalHogV1(src);   
    
    histCol = makeHist(src, chromBins);

    std::vector<std::vector<double>> features;

    features.push_back(histHog.reshape(1,1));
    features.push_back(histCol.reshape(1,1));

    return features;
}

/**
 * The function "readDnnFeatures" reads the DNN features from the CSV file containing paths to images 
 * and their corresponding features, and writes the paths to another CSV file named "features.csv" located 
 * in the "../data/" directory. This is done to ensure uniformity, so that it is readable by every distance 
 * metric function.
 * 
 * This function assumes that the input CSV file containing features is formatted as follows:
 * - Each line contains a comma-separated list of values.
 * - The first value in each line is the filename of an image.
 * - The remaining values represent the features extracted from that image.
 * 
 * The output CSV file "features.csv" contains the paths to the images read from the input CSV file.
 *
 * This function does not return any value.
 */
void readDnnFeatures(){
    std::string buffer;
    int i;
    std::string fileName;
    std::string csvOutPath{"../data/features.csv"};
    std::string imgdbdir{"../data/olympus/olympus/"};

    std::string nnPath{"../data/ResNet18_olym.csv"};

    std::ofstream outputCsv(csvOutPath.c_str());
    
    std::ifstream featurecsv(nnPath);

    std::string line;
    std::string imPath;

    while(std::getline(featurecsv, line)){
        std::stringstream ss(line);
        std::string element;
        std::vector<double> feature;
        std::vector<std::vector<double>> features;
        buffer = imgdbdir;
        std::getline(ss, fileName, ',');

        buffer += fileName;

        outputCsv << buffer << "\n";

        bool isFirst = true;
        while(std::getline(ss, element, ',')){
            if(element == "<SEP>"){
                outputCsv << element;
                outputCsv << ",";
            } else {
                double value;
                std::istringstream(element) >> value;
                if(isFirst != true){
                    outputCsv << ",";
                }
                isFirst = false;
                outputCsv << value;   
            }
        }
        outputCsv << "\n";
    }
}

/**
 * The function "slidingExtraction" performs sliding window feature extraction on the input image.
 * There is not overlap between success windows, there is no padding used. if the window is less than
 * the kernel's dimentions then the remain part of the image is taken as it is, to compute the features.
 * 
 * @param src The parameter 'src' is the input image on which sliding window feature extraction is 
 * performed. It should be of type 'cv::Mat'.
 * 
 * @param featureSlide The parameter 'featureSlide' is a function pointer representing the feature 
 * extraction method to be applied to each sliding window.
 * 
 * @param kernelSize The parameter 'kernelSize' specifies the size of the sliding window kernel.
 * 
 * @return a vector of vectors of type double containing the extracted features. Each inner vector 
 * represents a set of features extracted from a sliding window region of the input image.
 */
std::vector<std::vector<double>> slidingExtraction (cv::Mat &src, featureMethod featureSlide, int kernelSize){
    cv::Mat imCrop;
    std::vector<std::vector<double>> features;

    int width = src.size().width;
    int height = src.size().height;
    
    int numStridesHor = std::ceil(double(width) / (double)kernelSize);
    int numStridesVert = std::ceil(double(height) / (double)kernelSize);
    // std::cout << "\n kernel size : " << kernelSize << " rows : " << numStridesHor << " cols : " << numStridesVert << "\n";
    for(int rStride = 0 ; rStride < numStridesVert ; rStride++){
        for(int cStride = 0 ; cStride < numStridesVert ; cStride++){
            int xmin = rStride * kernelSize;
            int ymin = cStride * kernelSize;
            int strideWidth = std::min(width - xmin, kernelSize);
            int strideHeight = std::min(height - ymin, kernelSize);
            // std::cout << "crop : xmin : "<< xmin << " ymin : " << ymin << " strideWith : ";
            // std::cout << strideWidth << " strideHeight : " << strideHeight << "\n";
            cv::Rect crop(xmin, ymin, strideWidth, strideHeight);
            src(crop).copyTo(imCrop);
            std::vector<std::vector<double>> cropFeatures;
            cropFeatures = featureSlide(imCrop);

            for(size_t i = 0 ; i < cropFeatures.size() ; i++){
                features.push_back(cropFeatures[i]);
             }   

         }
    }
    return features;
}
