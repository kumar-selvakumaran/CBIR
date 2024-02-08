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

//#### only for testing addtest/subtracttest
#include <featureutils.h>
#include <distanceutils.h>
#include <utils.h>


int main(int argc, char *argv[]) {

    std::string featurePath{"../data/features.csv"};
    
    std::string dataDir{"../data/olympus/olympus/"};
    std::string targetPath{"../data/olympus/olympus/pic.0164.jpg"};

    
    

    //____________ENTER FEATURE TYPE_____________________________________________
    // std::string featureType{"upperLowerQuartersHist"};
    // //___________________________________________________________________________

    // std::cout << "\n going to initialize feature extractor \n";

    // FeatureExtractor featureExtractor{FeatureExtractor(dataDir,  featurePath, featureType)};

    // std::cout << "\n initialized feature extractor, going to compute the features \n";
    // featureExtractor.computeFeatures();

    // std::cout << "\n FINISHED COMPUTING THE FEATURES \n";


    // //___________ENTER DISTANCE TYPE and NUM IMAGES TO DISTPLAY__________________
    std::string distanceType{"HistogramIntersection"};
    int numImages = 5;
    //___________________________________________________________________________

    DistanceFinder dfObject{DistanceFinder(featurePath,  targetPath, distanceType)};
    
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

    // cv::Mat histUpperQuarter;
    // cv::Mat histLowerQuarter;

    // int upperQuarterRowStart = 0;
    // int upperQuarterColStart = 0;
    // int upperQuarterRowEnd = (int)(src.rows/4);
    // int upperQuarterColEnd = src.cols;

    // cv::Mat upperQuarter;

    // std::cout<<"\ncropping the upper quarter\n";

    // src(
    //     cv::Range(upperQuarterRowStart, upperQuarterRowEnd),
    //     cv::Range(upperQuarterColStart, upperQuarterColEnd)
    //     ).copyTo(upperQuarter);

    // std::cout << "\ncropped the upper quarter, making hist \n";
    // printmat(upperQuarter, 10);
    // histUpperQuarter = makeHist(upperQuarter, 4); 

    // std::cout << "\n made upper quarter hist, cropping lower quarter \n";

    // int lowerQuarterRowStart = (int)(3*(src.rows/4));
    // int lowerQuarterColStart = 0;
    // int lowerQuarterRowEnd = src.rows;
    // int lowerQuarterColEnd = src.cols;

    // cv::Mat lowerQuarter;

    // src(
    //     cv::Range(lowerQuarterRowStart, lowerQuarterRowEnd),
    //     cv::Range(lowerQuarterColStart, lowerQuarterColEnd)
    //     ).copyTo(lowerQuarter);

    // std::cout << "\ncropped lower quarter , making lower quarter hist\n";

    // printmat(lowerQuarter, 10);

    // histLowerQuarter = makeHist(lowerQuarter, 4); 

    // std::cout << "\n made lower quarter hist, making features \n";

    // std::vector<std::vector<double>> features;
    // features.push_back(histUpperQuarter.reshape(1,1));
    // features.push_back(histLowerQuarter.reshape(1,1));

    // for( size_t i = 0 ; i < features.size() ; i++){
    //     myPrintVec(features[i]);
    // }

    // /*output has 0s becaause toy matrix is 1d aka only blue channel.*/

    return 0;
}