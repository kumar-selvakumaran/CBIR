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

// std::vector<float> myMatToVec(cv::Mat &src){
//     std::vector<float> vecMat;
//     if (src.isContinuous()) {
//         // array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
//         vecMat.assign((float*)src.data, (float*)src.data + src.total()*src.channels());
//     } else {
//         for (int i = 0; i < src.rows; ++i) {
//             vecMat.insert(vecMat.end(), src.ptr<float>(i), src.ptr<float>(i)+src.cols*src.channels());
//         }
//     }

//     return vecMat;
// }

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
    std::cout << "\nchannels : " << src.channels() << " type : " << myMatType(src) << "\n";
    
}

// int kmeans( std::vector<cv::Vec3b> &data,
//     std::vector<cv::Vec3b> &means,
//     int *labels,
//     int K,
//     int maxIterations,
//     int stopThresh ) {
//     // error checking
//     if( K > data.size() ) {
//         printf("error: K must be less than the number of data points\n");
//         return(-1);
//     }
//     // clear the means vector
//     means.clear();
//     // initialize the K mean values
//     // use comb sampling to select K values
//     int delta = data.size() / K;
//     int istep = rand() % (data.size() % K);
//     for(int i=0;i<K;i++) {
//         int index = (istep + i*delta) % data.size();
//         means.push_back( data[index] );
//     }
//     // have K initial means
//     // loop the E-M steps
//     for(int i=0;i<maxIterations;i++) {
//         // classify each data point using SSD
//         for(int j=0;j<data.size();j++) {
//         int minssd = SSD( means[0], data[j] );
//         int minidx = 0;
//         for(int k=1;k<K;k++) {
//             int tssd = SSD( means[k], data[j] );
//             if( tssd < minssd ) {
//                 minssd = tssd;
//                 minidx = k;
//             }
//         }
//         labels[j] = minidx;
//         }
//         // calculate the new means
//         std::vector<cv::Vec4i> tmeans(means.size(), cv::Vec4i(0, 0, 0, 0) ); //
//         initialize with zeros
//         for(int j=0;j<data.size();j++) {
//             tmeans[ labels[j] ][0] += data[j][0];
//             tmeans[ labels[j] ][1] += data[j][1];
//             tmeans[ labels[j] ][2] += data[j][2];
//             tmeans[ labels[j] ][3] ++; // counter
//         }
//         int sum = 0;
//         for(int k=0;k<tmeans.size();k++) {
//             tmeans[k][0] /= tmeans[k][3];
//             tmeans[k][1] /= tmeans[k][3];
//             tmeans[k][2] /= tmeans[k][3];
//             // compute the SSD between the new and old means
//             sum += SSD( tmeans[k], means[k] );
//             means[k][0] = tmeans[k][0]; // update the mean
//             means[k][1] = tmeans[k][1]; // update the mean
//             means[k][2] = tmeans[k][2]; // update the mean
//         }
//         // check if we can stop early
//         if( sum <= stopThresh ) {
//             break;
//         }
//     }
//     // the labels and updated means are the final values
//     return(0);
// }

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


