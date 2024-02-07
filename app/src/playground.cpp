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
    std::map<std::string , cv::Mat>featureMap;
    std::ifstream featurecsv(featurePath);
    std::string line;
    int lineNum = 0;
    std::string imPath1{"../data/olympus/olympus/pic.0002.jpg"};
    std::string imPath2{"../data/olympus/olympus/pic.0003.jpg"};
    std::string targetPath{"../data/olympus/olympus/pic.1016.jpg"};

    // cv::Mat a = cv::imread(targetPath, cv::IMREAD_COLOR);
    cv::Mat b = cv::imread(imPath2, cv::IMREAD_COLOR);
    
    int numImages = 5;

    featureMethod middlecrop{&baselineFeatures7x7};
    distanceMethod eudist{&euclideanDistance};

    std::vector<float> imVec2 = middlecrop(b);

    cv::Mat vecMat2(imVec2, CV_32F);

    std::string distanceType{"Eucliean Distance"};

    DistanceFinder dfObject{DistanceFinder(featurePath,  targetPath, distanceType)};
    
    std::cout << "\n going to compute distances \n";

    dfObject.loadFeatures();

    dfObject.computeDistances();

    std::cout<<"\n computed distances \n";

    std::cout<<"\n getting similar images\n";

    dfObject.getSimilarImages(numImages, "show");

    std::cout<<"\n done getting similar images\n";

    // std::cout << "\n" << funcptr(vecMat1, vecMat2) << "\n";

    
    return 0;
}