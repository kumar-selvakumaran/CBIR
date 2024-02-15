/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This file contains the functions and classes required to compute the HOG histogram.
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>

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
#include <featureutils.h>
#include <distanceutils.h>
#include <hog.h>

/**
 * The constructor "hog" initializes an object of the hog class with specified parameters.
 * 
 * @param blurrKernelSize An integer representing the size of the blur kernel. it is used to reduce background
 * texture if it is abundant and keep the distance metric from being overly driven by background texture
 * 
 * @param threshPositive An integer representing the threshold for positive values. Gradients above this threshold
 * are kept for further processing
 * 
 * @param threshNegative An integer representing the threshold for negative values. Gradients below this threshold
 * are kept for further processing
 * 
 * @param globalIntensityBins An integer representing the number of bins for intensity binned by the chromotacity histogram.
 * 
 * @param globalOrientationBins An integer representing the number of bins for orientation binned by the histogram of orientations.
 * 
 * Upon construction, the constructor assigns the provided parameters to corresponding member variables
 * of the hog object.
 */

hog::hog(int blurrKernelSize,
        int threshPositive,
        int threshNegative,
        int globalIntensityBins,
        int globalOrientationBins){
    
    this->blurrKernelSize = blurrKernelSize;
    this->threshPositive = threshPositive;
    this->threshNegative = threshNegative;
    this->globalIntensityBins = globalIntensityBins;
    this->globalOrientationBins = globalOrientationBins;
}

/**
 * This function applies median blur to the input image, converts it to 64-bit floating-point format,
 * and then computes gradients using the specified filters. The filters can be modified based on the desired
 * gradient computation method (e.g., Sobel, Laws wave S5, Gaussian). The function also performs thresholding
 * on the computed gradients using predefined thresholds ('threshPositive' and 'threshNegative').
 * 
 * The function "computeGrad" computes gradients in the provided image using specified filters.
 * 
 * @param src The input image on which gradients are to be computed.
 * 
 * @param isX A boolean value indicating whether to compute gradients along the x-axis ('true') or y-axis ('false').
 * 
 * @return A cv::Mat object representing the computed gradients.
 * 
 */
cv::Mat hog::computeGrad(cv::Mat &src, bool isX){
    cv::Mat dst;
    dst = src.clone();
    
    cv::Mat grad;

    dst.convertTo(dst, CV_8U);
    cv::medianBlur(dst, dst, 5);

    dst.convertTo(dst, CV_64F);

    //sobel
    float filter1f[5] = {1, 2, 0, -2, -1};

    //laws wave S5
    // float filter1f[5] = {1, -2, 0, 2, -1};

    //laws spot S5
    // float filter1f[5] = {-1, 0, 2, 0, -1};


    //gaussian 
    float filter2f[5] = {1, 2, 4, 2, 1};

    
    //laws spot S5
    // float filter2f[5] = {1, -2, 0, 2, -1};


    cv::Mat filterRow(1, 5, CV_64F, filter1f);
    cv::Mat filterCol(1, 5, CV_64F, filter2f);

    if(isX == false){
        cv::swap(filterCol, filterRow);
    }

    dst.convertTo(dst, CV_16S);
    grad.convertTo(grad, CV_16S);
    cv::sepFilter2D(dst, grad, dst.depth(), filterRow, filterCol);
    dst.convertTo(dst, CV_64F);
    grad.convertTo(grad, CV_64F);

    grad =  myThresh(grad, threshPositive, threshNegative);

    grad.convertTo(grad, CV_64F);

    return grad;
}  

/**
 * This function computes the magnitude of gradients using the provided x and y gradient components.
 * 
 * The function "computeMagnitude" computes the magnitude of gradients given the x and y gradient components.
 * 
 * @param gradX A reference to a cv::Mat object representing the x-component of gradients.
 * 
 * @param gradY A reference to a cv::Mat object representing the y-component of gradients.
 * 
 * @return A cv::Mat object representing the computed magnitude of gradients.
 */
cv::Mat hog::computeMagnitude(cv::Mat &gradX, cv::Mat &gradY){
    cv::Mat temp1;
    cv::Mat temp2;
    cv::Mat magMat;
    cv::pow(gradX, 2, temp1);
    cv::pow(gradY, 2, temp2);

    cv::sqrt(temp1 + temp2, magMat);
    magMat += cv::Scalar(1, 1, 1);

    return magMat;
}

/**
 * The function "computeOrientation" computes orientation matrices from provided gradients along the x-axis and y-axis.
 * 
 * @param gradX A cv::Mat object representing the gradients computed along the x-axis.
 * 
 * @param gradY A cv::Mat object representing the gradients computed along the y-axis.
 * 
 * @return A cv::Mat object representing the computed orientation matrix.
 * 
 * This function first computes the magnitude matrix using the provided gradients.
 * Then, it calculates the orientation matrix by dividing the sum of gradients along the x-axis and y-axis
 * by the magnitude matrix. This operation helps in obtaining the orientation information from the gradients.
 * 
 * Additionally, the function normalizes the orientation matrix to a specified range using the "myNormMatInv" function,
 * which ensures that the values fall within the range [0, 255].
 */
cv::Mat hog::computeOrientation(cv::Mat &gradX, cv::Mat &gradY){
    cv::Mat orMat;

    cv::Mat magMat;
    magMat = computeMagnitude(gradX, gradY);
    
    orMat = (gradX + gradY) / (magMat);
    
    std::pair<double, double> parsold = myNormMat(orMat, orMat);

    std::pair<double, double> parsNew(0, 255);
    myNormMatInv(orMat, orMat, parsNew);


    return orMat;
}

/**
 * The function "computeGlobalHogV1" computes Histogram of Oriented Gradients (HOG) features
 * for the provided image using specified parameters.
 * 
 * @param src The input image for which HOG features are to be computed.
 * 
 * @return A cv::Mat object representing the computed HOG features.
 * 
 * This function first computes gradients along both x and y axes using the "computeGrad" function,
 * then calculates magnitude and orientation matrices. It then constructs a histogram of oriented gradients
 * using the computed magnitude and orientation matrices. The histogram is devised based on specified
 * global intensity and orientation bins. The magnitude is splitted into bins based on how close that magnitude value
 * is to a bin.
 */
cv::Mat hog::computeGlobalHogV1(cv::Mat &src){

    cv::Mat gradX;
    gradX = computeGrad(src, true);
    cv::Mat gradY;
    gradY = computeGrad(src, false);


    cv::Mat magMat;
    magMat = computeMagnitude(gradX, gradY);

    cv::Mat orMat;
    orMat = computeOrientation(gradX, gradY);   

    int dimArray[3] = {globalIntensityBins,
                        globalIntensityBins,
                        globalOrientationBins};

    cv::Mat hist = cv::Mat::zeros(3, dimArray, CV_64FC1);


    double maxVal, minVal;
    
    cv::minMaxLoc(magMat, &minVal, &maxVal);
    int intensityBinSize = static_cast<int>(maxVal / globalIntensityBins);
    intensityBinSize = std::max(1, intensityBinSize);

    cv::minMaxLoc(orMat, &minVal, &maxVal);
    int orientationBinSize = static_cast<int>(maxVal / globalOrientationBins);
    orientationBinSize = std::max(1, orientationBinSize);

    int numElements = magMat.size().width * magMat.size().height;
    int numRows = magMat.size().height;
    int numCols = magMat.size().width;

    cv::minMaxLoc(magMat, &minVal, &maxVal);
    if((magMat.isContinuous() == true) && (orMat.isContinuous() == true) && (hist.isContinuous() == true)){
        for (int elInd = 0 ; elInd < numElements ; elInd ++){
            int rowInd = (int)elInd / numCols;
            int colInd = (int)elInd % numCols;

            cv::Vec<double, 3> currMag = magMat.at<cv::Vec<double, 3>>(rowInd, colInd); //(1 was added to avoid dividebyZero error)
            cv::Vec<double, 3> currOr = orMat.at<cv::Vec<double, 3>>(rowInd, colInd);

            double bMag = currMag[0] - 1;
            double gMag = currMag[1] - 1;
            double rMag = currMag[2] - 1;

            double bOr = currOr[0];
            double gOr = currOr[1];
            double rOr = currOr[2];

            double divisor = bMag + gMag + rMag;
            divisor = divisor > 0.0 ? divisor : 1.0;
            
            double chromR = rMag / divisor;
            double chromG = gMag / divisor;

            int rBin = (int) (chromR * (globalIntensityBins - 1) + 0.5);
            int gBin = (int) (chromG * (globalIntensityBins - 1) + 0.5);

            int orBinLowR = static_cast<int>(rOr/orientationBinSize);
            int orBinHighR = std::min(globalOrientationBins - 1, orBinLowR + 1);
            double lowRatioR;
            if(orBinLowR != globalOrientationBins - 1){
                lowRatioR = (rOr - (orBinLowR*orientationBinSize)) / ((orBinHighR*orientationBinSize) - (orBinLowR*orientationBinSize));
            } else {
                lowRatioR = 0.5;
            }

            // ----------- SPLITTING MAGNITUDE ---------------
            double lowComponentR = lowRatioR * rMag;          
            double highComponentR = (1 - lowRatioR) * rMag;

            //------------NO SPLITTING MAGNITUDE -------------
            // double lowComponentR = rMag;          
            // double highComponentR = 0;           

            const int constGlobalOrientationBins{globalOrientationBins}; 

            if(globalOrientationBins == 5){
                hist.at<Vec5d>(rBin, gBin)[orBinLowR] += lowComponentR;
                hist.at<Vec5d>(rBin, gBin)[orBinHighR] += highComponentR;
            } else if (globalOrientationBins == 2){
                hist.at<cv::Vec2d>(rBin, gBin)[orBinLowR] += lowComponentR;
                hist.at<cv::Vec2d>(rBin, gBin)[orBinHighR] += highComponentR;
            }
                


            int orBinLowG = static_cast<int>(gOr/orientationBinSize);
            int orBinHighG = std::min(globalOrientationBins - 1, orBinLowG + 1);
            double lowRatioG;
            if(orBinLowG != globalOrientationBins - 1){
                lowRatioG = (gOr - (orBinLowG*orientationBinSize)) / ((orBinHighG*orientationBinSize) - (orBinLowG*orientationBinSize));
            } else {
                lowRatioG = 0.5;
            }

            // --------------SPLITTING MAGNITUDE -------------
            double lowComponentG = lowRatioG * gMag;          
            double highComponentG = (1 - lowRatioG) * gMag;

            //---------------NO SPLITTING MAGNITUDE -----------
            // double lowComponentG = gMag;          
            // double highComponentG = 0;



            if(globalOrientationBins == 5){
                hist.at<Vec5d>(rBin, gBin)[orBinLowG] += lowComponentG;
                hist.at<Vec5d>(rBin, gBin)[orBinHighG] += highComponentG;
            } else if (globalOrientationBins == 2){
                hist.at<cv::Vec2d>(rBin, gBin)[orBinLowG] += lowComponentG;
                hist.at<cv::Vec2d>(rBin, gBin)[orBinHighG] += highComponentG;
            }

        }    
    }

    return hist;

}




