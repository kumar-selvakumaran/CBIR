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
#include <map>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

//#### only for testing addtest/subtracttest
#include<featureutils.h>

int main(int argc, char *argv[]) {

    std::string featurePath{"../data/features.csv"};
    std::map<std::string , cv::Mat>featureMap;
    std::ifstream featurecsv(featurePath);
    std::string line;
    int lineNum = 0;
    std::string imPath;
    cv.
    return 0;
}