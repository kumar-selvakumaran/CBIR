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

    
    

    // //____________ENTER FEATURE TYPE_____________________________________________
    // std::string featureType{"Histogram"};
    // //___________________________________________________________________________

    // std::cout << "\n going to initialize feature extractor \n";

    // FeatureExtractor featureExtractor{FeatureExtractor(dataDir,  featurePath, featureType)};

    // std::cout << "\n initialized feature extractor, going to compute the features \n";
    // featureExtractor.computeFeatures();

    // std::cout << "\n FINISHED COMPUTING THE FEATURES \n";



    // //___________ENTER DISTANCE TYPE and NUM IMAGES TO DISTPLAY__________________
    std::string distanceType{"EuclideanDistance"};
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







    // while(std::getline(featurescsv, line)){
    //     std::stringstream ss(line);
    //     std::string element;
    //     std::vector<double> featureVec;
    //     if(lineNum)
    // }

    return 0;
}