/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * A playground file which executes the whole pipeline, and where tests , prototyping can be done.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <map>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <random>
#include <algorithm>

#include <featureutils.h>
#include <distanceutils.h>
#include <utils.h>
#include <hog.h>


int main(int argc, char *argv[]) {


    std::string featurePath{"../data/features.csv"};
    
    // std::string dataDir{"../data/olympus/olympus/"};
    std::string dataDir{"../data/olympus/olympus/"};
    std::string targetPath{"/app/data/sidewalk.jpg"};
    // std::string targetPath{"/app/data/olympus/olympus/pic.1009.jpg"};


    // // //____________ENTER FEATURE TYPE_____________________________________________
    std::string featureType{"Histogram"};
    bool useStridedFeatures = false;
    
    cv::Mat tempMat, hist;


    tempMat = cv::imread(targetPath, cv::IMREAD_COLOR);
    // cv::GaussianBlur(tempMat, tempMat, cv::Size(5,5), 3, 3, 4);

    cv::Mat dst;
    int numBins = tempMat.rows;


    hist =  makeHist(tempMat, numBins);
    
    cv::namedWindow("temp");
    cv::imshow("temp", tempMat);
    cv::waitKey(0);

    // vizHist(hist, 8);

    // cv::Mat dst;

    std::cout << "\nhist rows : " << hist.rows << "\n hist columns : " << hist.cols<<"\n";
    for(int vizi = 0 ; vizi < 5 ; vizi++){
        dst.create( hist.size(), CV_8UC3 );
        for(int i=0;i<hist.rows;i++) {
            cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);
            float *hptr = hist.ptr<float>(i);
            for(int j=0;j<hist.cols;j++) {
                if( i + j > hist.rows ) {
                    ptr[j] = cv::Vec3b( 200, 120, 60 ); // default color
                    continue;
                }
                float rcolor = (float)i / numBins;
                float gcolor = (float)j / numBins;
                float bcolor = 1 - (rcolor + gcolor);
                ptr[j][0] = hptr[j] > 0 ? hptr[j] * 128 + 128 * bcolor : 0 ;
                ptr[j][1] = hptr[j] > 0 ? hptr[j] * 128 + 128 * gcolor : 0 ;
                ptr[j][2] = hptr[j] > 0 ? hptr[j] * 128 + 128 * rcolor : 0 ;
            }
        }
        cv::resize (dst, dst, cv::Size(500,500));
        displayImage(dst, false, false);
    }


    // std::cout << "\n going to initialize feature extractor \n";

    // FeatureExtractor featureExtractor{FeatureExtractor(dataDir,  featurePath, featureType, useStridedFeatures)};

    // std::cout << "\n initialized feature extractor, going to compute the features \n";
    // featureExtractor.computeFeatures();

    // std::cout << "\n FINISHED COMPUTING THE FEATURES \n";
    // //___________________________________________________________________________

    // // reading dnn features

    // // readDnnFeatures();

    // //___________ENTER DISTANCE TYPE and NUM IMAGES TO DISTPLAY__________________
    // std::string distanceType{"HistogramIntersection"};
    // int numImages = 8;
    // //___________________________________________________________________________

    // DistanceFinder dfObject{DistanceFinder(featurePath,  targetPath, distanceType, featureType)};
    
    // std::cout << "\n going to compute distances \n";

    // dfObject.loadFeatures();

    // dfObject.computeDistances();

    // std::cout<<"\n computed distances \n";

    // std::cout<<"\n getting similar images\n";

    // dfObject.getSimilarImages(numImages, "show");
    // // dfObject.getDisSimilarImages(numImages, "show");

    // std::cout<<"\n done getting similar images\n";

    return 0;
}