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
#include <sstream>
#include <vector>

#include <utils.h>
#include <hog.h>

std::string myMatType(cv::Mat &src) {
    int type = src.type();
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void printmat(cv::Mat &src, int vizdim)
{
    int xMax = std::min(90, src.rows);
    int xMin = xMax - std::min(vizdim, src.rows);
    int yMax = std::min(90, src.cols);
    int yMin = yMax - std::min(vizdim, src.cols);
    cv::Mat vizslice(src(cv::Range(xMin, xMax), cv::Range(yMin, yMax)));
    std::cout << "\nmatrix chunk : \n" << format(vizslice, cv::Formatter::FMT_NUMPY );
    double minVal;
    double maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);
    std::cout << "\nchannels : " << src.channels() << " type : " << myMatType(src); 
    std::cout << " Min val : " << minVal << " Max val : " << maxVal;
    std::cout << " rows : " << src.rows << " cols : " << src.cols << "\n";

}

std::pair<double, double> myNormMat(cv::Mat &src, cv::Mat &dst){
    double minVal;
    double maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);
    dst = src.clone();
    if(maxVal != minVal){
        dst = (dst - minVal) / (maxVal - minVal);
    }    
    std::pair<double, double> pars(minVal, maxVal);
    return pars;
}

std::pair<double, double> myNormVec(std::vector<double> &src, std::vector<double> &dst){
    double minVal;
    double maxVal;

    cv::Mat matSrc(src);
    cv::minMaxLoc(matSrc, &minVal, &maxVal);

    if(maxVal != minVal){
        matSrc = (matSrc - minVal) / (maxVal - minVal);
    }   

    dst = matSrc;
    std::pair<double, double> pars(minVal, maxVal);
    return pars;
}

void myNormMatInv(cv::Mat &src, cv::Mat &dst, std::pair<double, double> pars){
    double minVal = pars.first;
    double maxVal = pars.second;
    dst = src.clone();
    dst = (dst * (maxVal - minVal)) + minVal;
}

cv::Mat makeHist(cv::Mat &src, int numBins){
    cv::Mat hist;
    // NUMBER OF BINS
    int histSize = std::max(src.rows, src.cols);
    histSize = std::min(numBins, histSize);

    hist = cv::Mat::zeros( cv::Size( histSize, histSize ), CV_64FC1 );

    for( int i=0;i<src.rows;i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++) {

            double B = ptr[j][0];
            double G = ptr[j][1];
            double R = ptr[j][2];

            double divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;
            double r = R / divisor;
            double g = G / divisor;

            int rindex = (int)( r * (histSize - 1) + 0.5 );
            int gindex = (int)( g * (histSize - 1) + 0.5 );

            hist.at<double>(rindex, gindex)++;
        }
    }

    hist /= (src.rows * src.cols);
    return hist;
}

int earliestDecPos(double num){
    if(std::floor(num) > 0){
        std::cout << "\n TRYNG TO FIND THE EARLIEST DECIMAL PLACE FOR NUM > 1 (NOT INTENDED WHILE DEFINITION)\n";
        return 0;
    }
    double decPart  = num - std::floor(num);
    int reqpos = 0; 
    double diff = 0;
    while(diff==0){
        double decPart10 = decPart * 10;
        diff = (int)decPart10 - (int)decPart;
        decPart = decPart10;
        reqpos++;
    }

    return reqpos;
}

void displayImage(cv::Mat &img){
    cv::Mat vizim = img.clone();
    vizim.convertTo(vizim, CV_8U);
    cv::namedWindow("viz image");
    cv::imshow("viz image", vizim);
    cv::waitKey(0);
}

cv::Mat myThresh(cv::Mat &img, int postThresh, int negThresh){
    cv::Mat neg, pos;
    pos = (img >= postThresh) / 255;
    neg = (img <= negThresh) / 255;

    cv::Mat out;  
    out = pos + neg;

    out.convertTo(out, CV_64F);

    cv::multiply(out, img, out);

    return out;
}

void emphEdges(cv::Mat &src, cv::Mat &dst){
    cv::Mat gradX;
    cv::Mat gradY;
    cv::Mat magMat;
    cv::Mat orMat;
    cv::Mat channels[3];
    cv::Mat test;

    hog hogComputer(5, 160, -160, 8, 5);
    gradX = hogComputer.computeGrad(src, true);
    gradY = hogComputer.computeGrad(src, false);
    magMat = hogComputer.computeMagnitude(gradX, gradY);
    orMat = hogComputer.computeOrientation(gradX, gradY);
    test = (magMat >= 20);
    test.convertTo(test, CV_64F);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
                cv::Size(2, 2),
                cv::Point(1, 1));

    cv::erode( test, test, element);
    cv::dilate( test, test, element );

    cv::split(test, channels);
    
    test = cv::max(channels[0], channels[1]);
    test = cv::max(channels[2], test);  

    test = test/255;
    test = 1 - test;
    
    channels[0] = test;
    channels[1] = test;
    channels[2] = test;
    
    cv::merge(channels, 3, test);
    
    test.convertTo(test, CV_64F);
    
    src.convertTo(src, CV_64F);
    
    cv::multiply(test, src, test);

    dst = test;
}   


void getEdgeImage(cv::Mat &src, cv::Mat &dst){

    cv::Mat gradX;
    cv::Mat gradY;
    cv::Mat magMat;
    cv::Mat orMat;
    cv::Mat channels[3];
    cv::Mat test;

    hog hogComputer(5, 160, -160, 8, 5);
    gradX = hogComputer.computeGrad(src, true);
    gradY = hogComputer.computeGrad(src, false);
    magMat = hogComputer.computeMagnitude(gradX, gradY);
    orMat = hogComputer.computeOrientation(gradX, gradY);
    test = (magMat >= 20);
    test.convertTo(test, CV_64F);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
                cv::Size(2, 2),
                cv::Point(1, 1));

    cv::erode( test, test, element);
    cv::dilate( test, test, element, cv::Point(-1,-1), 2);
    cv::erode( test, test, element);
    cv::dilate( test, test, element, cv::Point(-1,-1), 2);
    cv::erode( test, test, element, cv::Point(-1,-1), 2);


    cv::split(test, channels);
    
    test = cv::max(channels[0], channels[1]);
    test = cv::max(channels[2], test);  
    
    channels[0] = test;
    channels[1] = test;
    channels[2] = test;
    
    cv::merge(channels, 3, test);
    dst = test;
}   