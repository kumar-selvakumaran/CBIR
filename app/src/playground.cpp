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

#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <map>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <algorithm>

//#### only for testing addtest/subtracttest
#include <featureutils.h>
#include <distanceutils.h>
#include <utils.h>
#include <hog.h>


int main(int argc, char *argv[]) {

    std::string featurePath{"../data/features.csv"};
    
    // std::string dataDir{"../data/olympus/olympus/"};
    std::string dataDir{"../data/olympus/olympus/"};
    std::string targetPath{"../data/olympus/objects/pic.0373.jpg"};


    // // //____________ENTER FEATURE TYPE_____________________________________________
    std::string featureType{"globalHogandColour"};
    bool useStridedFeatures = false;

    //___________________________________________________________________________

    // std::cout << "\n going to initialize feature extractor \n";

    // FeatureExtractor featureExtractor{FeatureExtractor(dataDir,  featurePath, featureType, useStridedFeatures)};

    // std::cout << "\n initialized feature extractor, going to compute the features \n";
    // featureExtractor.computeFeatures();

    // std::cout << "\n FINISHED COMPUTING THE FEATURES \n";


    //___________ENTER DISTANCE TYPE and NUM IMAGES TO DISTPLAY__________________
    std::string distanceType{"stridedEuclideanDistance"};
    int numImages = 8;
    //___________________________________________________________________________

    DistanceFinder dfObject{DistanceFinder(featurePath,  targetPath, distanceType, featureType)};
    
    std::cout << "\n going to compute distances \n";

    dfObject.loadFeatures();

    dfObject.computeDistances();

    std::cout<<"\n computed distances \n";

    std::cout<<"\n getting similar images\n";

    dfObject.getSimilarImages(numImages, "show");

    std::cout<<"\n done getting similar images\n";


    //##################### PLAY GROUND ##############################

    
    // std::string imPath1{"../data/olympus/olympus/pic.0930.jpg"};

    /*
    alter loadFeatures() : make feature map std::map<std::string , std::vector<std::vector<double>>>
    - alterDistance functions to take in std::vector<std::vector<double>> as input, .
            vector of float vectors is preferred over 2d mat, because, it supports heterogenous sized elements,
            meaning, features of different sizes. probably with the use of pointers and non-contiguous elements(inner arrays).

    - distance fuctions that 
    */
    
    // std::ifstream featurecsv(featurePath);
    // std::string line;
    // std::getline(featurecsv, line);
    // std::getline(featurecsv, line);
    // std::cout << "\n" << line << "\n";
    // std::string test;
    // std::stringstream testss(line);
    // std::getline(testss, test, ',');
    // double element;
    // std::istringstream(test) >> element;
    // if(test == "0.00108337"){
    //     std::cout << "\nstringstream element comparable\n";
    // }
    // std::cout << "\n" << typeid(test).name() << " test " << test << "\n";
    // std::cout << "\n" << typeid(element).name() << " element " << element << "\n";



    // std::stringstream ss{"123.02,1231239,1.23,0.123214,<SEP>,234e-24,123123.123,5567,4334.0"};
    // std::string line;
    // std::vector<float> feature;
    // std::vector<std::vector<float>>features;
    // std::string element;
    
    // // if(lineNum%2!=0){

    // while(std::getline(ss, element, ',')){
    //     if(element == "<SEP>"){
    //         features.push_back(feature);
    //         feature = std::vector<float>();
    //     } else {
    //     float value;
    //     std::istringstream(element) >> value;
    //     feature.push_back(value);
    //     }
    // }
    // features.push_back(feature);

    // std::cout <<" \n intermediate 2d vector : ";
    // for(size_t i =  0 ; i < features.size() ; i++){
    //     myPrintVec(features[i]);
    // }

    // std::cout << "\n features size : " << features.size() << "\n";
    // for(size_t i = 0; i < features.size() ; i++) {
    //     if(i > 0){
    //         std::cout << ",<SEP>,";
    //     }
    //     cv::Mat temp(features[i]);
    //     std::vector<float> feature = temp;
    //     for (size_t j = 0; j < feature.size(); ++j) {
    //         std::cout << feature[j];
    //         if (j < feature.size() - 1) {
    //             std::cout << ",";
    //         }
    //     }

    // }
    
    // std::cout<<"\n";

    // // while(std::getline(featurescsv, line)){
    // //     std::stringstream ss(line);
    // //     std::string element;
    // //     std::vector<double> featureVec;
    // //     if(lineNum)
    // // }

    // double initer[32] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8};
    // cv::Mat testmat(8, 4, CV_64FC1, initer);
    // printmat(testmat, 10);

    // cv::Mat src = testmat;

    // cv::Mat histUpperCrop;
    // cv::Mat histLowerCrop;

    // int upperCropRowStart = 0;
    // int upperCropColStart = 0;
    // int upperCropRowEnd = (int)(src.rows/4);
    // int upperCropColEnd = src.cols;

    // cv::Mat upperCrop;

    // std::cout<<"\ncropping the upper Crop\n";

    // src(
    //     cv::Range(upperCropRowStart, upperCropRowEnd),
    //     cv::Range(upperCropColStart, upperCropColEnd)
    //     ).copyTo(upperCrop);

    // std::cout << "\ncropped the upper Crop, making hist \n";
    // printmat(upperCrop, 10);
    // histUpperCrop = makeHist(upperCrop, 4); 

    // std::cout << "\n made upper Crop hist, cropping lower Crop \n";

    // int lowerCropRowStart = (int)(3*(src.rows/4));
    // int lowerCropColStart = 0;
    // int lowerCropRowEnd = src.rows;
    // int lowerCropColEnd = src.cols;

    // cv::Mat lowerCrop;

    // src(
    //     cv::Range(lowerCropRowStart, lowerCropRowEnd),
    //     cv::Range(lowerCropColStart, lowerCropColEnd)
    //     ).copyTo(lowerCrop);

    // std::cout << "\ncropped lower Crop , making lower Crop hist\n";

    // printmat(lowerCrop, 10);

    // histLowerCrop = makeHist(lowerCrop, 4); 

    // std::cout << "\n made lower Crop hist, making features \n";

    // std::vector<std::vector<double>> features;
    // features.push_back(histUpperCrop.reshape(1,1));
    // features.push_back(histLowerCrop.reshape(1,1));

    // for( size_t i = 0 ; i < features.size() ; i++){
    //     myPrintVec(features[i]);
    // }

    // /*output has 0s becaause toy matrix is 1d aka only blue channel.*/

    // cv::Mat testmat = cv::imread(targetPath, cv::IMREAD_COLOR);

    
    // std::string buffer;
    // FILE *fp;
    // DIR *dirp;
    // struct dirent *dp;
    // int i;
    // dataDir = "../data/olympus/objects/";
    // dirp = opendir(dataDir.c_str());;
    // cv::Mat targetMat;

    // std::vector<std::string> imPaths;

    // while( (dp = readdir(dirp)) != NULL ) {
    //     // check if the file is an image
    //     if( strstr(dp->d_name, ".jpg") ||
    //     strstr(dp->d_name, ".png") ||
    //     strstr(dp->d_name, ".ppm") ||
    //     strstr(dp->d_name, ".tif") ) {
    //         // build the overall filename
    //         buffer = dataDir;
    //         // std::cout<< " \n " << buffer << "\n" ;
    //         buffer += dp->d_name;
    //         imPaths.push_back(buffer);
    //     }         
    // }


    // std::vector<size_t> indices(imPaths.size());
    // std::iota(indices.begin(), indices.end(), 0);
    // std::random_shuffle( indices.begin(), indices.end() );
    
    // targetPath =  "../data/olympus/objects/pen.jpg";

    // for(size_t i = 0 ; i < indices.size() ; i++){

    //     /* `targetPath = imPaths[indices[i]]` is assigning the `i`th image path from the `imPaths`
    //     vector to the `targetPath` variable. The `indices` vector is used to shuffle the indices of
    //     the `imPaths` vector, so each time the loop runs, a different image path is assigned to
    //     `targetPath`. This allows for random selection of images from the `imPaths` vector. */
    //     std::string featureImPath = imPaths[indices[i]];
    //     // targetPath =  "../data/olympus/olympus/pic.0598.jpg";

    //     std::cout<< "impath : " <<featureImPath << "\n";

    //     cv::Mat src;
    //     src = cv::imread(featureImPath, cv::IMREAD_COLOR); 

    //     hog hogComputer(5, 160, -160, 8, 5);

    //     getEdgeImage(src, src);

    //     std::string saveName = featureImPath.substr(0,16) + "processed/";    
    //     saveName += featureImPath.substr(24, (featureImPath.length() - 23 -  4)) + "edge.jpg";  

    //     cv::imwrite(saveName, src);

    //     std::cout << "\nsave name :" << saveName << "\n ";
    


    //     //############
    //     printmat(src, 3);
    //     cv::namedWindow("target image");
    //     cv::imshow("target image", src);
    //     cv::waitKey(0);

        // cv::Mat temp;

        // featureMethod featureSlide;
        // featureSlide = &histFeature;
        // std::vector<std::vector<double>> features;

        // int kernelSize = src.size().width / 3;

        // features =  slidingExtraction(src, featureSlide, kernelSize);

        // targetMat = cv::imread(targetPath, cv::IMREAD_COLOR);
        // std::vector<std::vector<double>> targetFeatures;
        // targetFeatures = featureSlide(targetMat);

        // rawDistanceMethod distancegetter = &rawEuclideanDistance;
        // double distance = stridedDistanceComputer(targetFeatures, features, distancegetter, false);
        
        // std::cout << "\ndistance found = " << distance  << "\n";    
        // displayImage(temp);
    // }
    //    ________________making grad orient hist__________________
    return 0;
    // }
}