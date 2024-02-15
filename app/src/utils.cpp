/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This is the file which contains utility functions used for miscellaneous purposes. 
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
#include <random>

#include <utils.h>
#include <hog.h>

/**
 * The function "myMatType" returns a string representation of the type of a given OpenCV Mat object.
 * 
 * @param src The input cv::Mat object whose type needs to be determined.
 * 
 * @return A string representing the type of the input Mat object.
 */
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

/**
 * The function "printmat" prints a specified chunk of a matrix along with additional information.
 * 
 * @param src The input matrix whose chunk needs to be printed.
 * 
 * @param vizdim An integer representing the dimension of the chunk to be visualized.
 */
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

/**
 * The function "myNormMat" normalizes the values in the input matrix to the range [0, 1].
 * 
 * @param src The input matrix whose values are to be normalized.
 * 
 * @param dst The output matrix where the normalized values will be stored.
 * 
 * @return A std::pair<double, double> containing the minimum and maximum values of the input matrix before normalization.
 */
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

/**
 * The function "myNormVec" normalizes the values in the source vector to a specified range.
 * 
 * @param src The source vector containing the values to be normalized.
 * 
 * @param dst A reference to a vector where the normalized values will be stored.
 * 
 * @return A std::pair<double, double> containing the minimum and maximum values of the source vector
 * before normalization.
 */
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

/**
 * Scales the matrix using the min and max values supplied.
 * 
 * @param src The input matrix to be normalized.
 * 
 * @param dst The output matrix where the normalized result will be stored.
 * 
 * @param pars A pair of doubles representing the normalization range (minimum and maximum values).
 * 
 */
void myNormMatInv(cv::Mat &src, cv::Mat &dst, std::pair<double, double> pars){
    double minVal = pars.first;
    double maxVal = pars.second;
    dst = src.clone();
    dst = (dst * (maxVal - minVal)) + minVal;
}

/**
 * The function "makeHist" computes a RG chromotacity histogram from the provided input image.
 * 
 * @param src The input image for which the histogram is to be computed.
 * 
 * @param numBins An integer representing the number of bins for histogram computation.
 * 
 * @return A cv::Mat object representing the computed histogram.
 */
cv::Mat makeHist(cv::Mat &src, int numBins){
    cv::Mat hist;
    // NUMBER OF BINS
    int histSize = std::max(src.rows, src.cols);
    histSize = std::min(numBins, histSize);

    hist = cv::Mat::zeros( cv::Size( histSize, histSize ), CV_32FC1 );

    for( int i=0;i<src.rows;i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++) {

            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;
            float r = R / divisor;
            float g = G / divisor;

            int rindex = (int)( r * (histSize - 1) + 0.5 );
            int gindex = (int)( g * (histSize - 1) + 0.5 );

            hist.at<float>(rindex, gindex)++;
        }
    }

    hist /= (src.rows * src.cols);
    return hist;
}

/** 
 * This function generates a visualization of the provided histogram using colors to represent intensity values.
 * 
 * @param hist The input histogram represented as a cv::Mat.
 * @param numBins The number of bins used in the histogram.
 */
void vizHist(cv::Mat hist, int numBins){
    cv::Mat dst;
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
    

/**
 * The function "earliestDecPos" finds the position of the earliest decimal place in a given number.
 * 
 * @param num The input double value for which the earliest decimal place position needs to be found.
 * 
 * @return An integer representing the position of the earliest decimal place in the provided number.
 */

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

/**
 * This function is used to display an image. It "saturate_casts" the values to uchar (0-255) by default, or normalizes them
 * if specified using 'normalize'. The values are normalized with min value = 0 , if specified using only Positive.
 * 
 * The function "displayImage" displays the provided image on the screen.
 * 
 * @param img The input image to be displayed.
 * 
 * @param normalize A boolean value indicating whether to normalize the pixel values of the image (`true`) or not (`false`).
 * 
 * @param onlyPositive A boolean value indicating whether to consider only positive pixel values during normalization (`true`) or both positive and negative values (`false`).
 */
void displayImage(cv::Mat &img, bool normalize, bool onlyPositive){
    cv::Mat vizim = img.clone();
    cv::Mat matSrc(img);
    
    if(normalize == true){
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);

        if(onlyPositive == true){
            int height = img.size().height;
            int width = img.size().width;
            cv::Mat iMat(height, width, CV_64FC3, cv::Scalar(1,1,1));
            img = img + (iMat*minVal);
        }
        
        if(maxVal != minVal){
            matSrc = (img - minVal) / (maxVal - minVal);
            matSrc = matSrc * 255;
            vizim = matSrc; 
        }
    }

    vizim.convertTo(vizim, CV_8U);
    printmat(vizim, 3);
    cv::namedWindow("viz image");
    cv::imshow("viz image", vizim);
    cv::waitKey(0);
}

/**
 * The function "myThresh" applies thresholding to the input image. * This function computes a thresholded image by
 * combining values greater than or equal to `postThresh values less than or equal to `negThresh`
 * to produce the final output.
 * 
 * @param img The input image to be thresholded.
 * 
 * @param postThresh An integer representing the threshold value for positive values.
 * 
 * @param negThresh An integer representing the threshold value for negative values.
 * 
 * @return A cv::Mat object representing the thresholded image.
 * 

 */

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

/**
 * The function "drawEdges" computes and draws edges on the input image. This function computes gradients
 * along the x and y axes using the hog class's "computeGrad" method. It then calculates the magnitude and
 * orientation of gradients using the "computeMagnitude" and "computeOrientation" methods, respectively.
 * Edges are detected by thresholding the magnitude of gradients and applying morphological operations
 * (erosion and dilation) for refinement.
 * 
 * The resulting edge mask is used to extract edges from the original image by setting non-edge pixels to zero.
 * The edges are drawn on the output image, which is returned in the "dst" parameter.
 * 
 * @param src The input image on which edges are to be drawn.
 * 
 * @param dst The output image where the edges will be drawn.
 */
void drawEdges(cv::Mat &src, cv::Mat &dst){
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

/**
 * This function computes gradients along the x and y axes using the "computeGrad" method
 * from the "hog" class. It then calculates the magnitude and orientation of gradients,
 * and thresholds the magnitude values to identify potential edge pixels. Morphological
 * operations (erosion and dilation) are applied to refine the edge map. Finally, the function
 * generates an edge image by setting pixels with non-zero magnitude values to white.
 * 
 * The function "getEdgeImage" generates an edge image from the provided input image.
 * 
 * @param src The input image from which the edge image is to be generated.
 * 
 * @param dst The output edge image generated by the function.
 */

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

/**
 * The function "getNoiseImage" generates a noise image with specified dimensions and intensity range using
 * values from a uniform distrubution specified by 'minVal', and 'maxVal'.
 * 
 * @param width An integer representing the width of the noise image.
 * 
 * @param height An integer representing the height of the noise image.
 * 
 * @param minVal An integer representing the minimum intensity value for noise pixels.
 * 
 * @param maxVal An integer representing the maximum intensity value for noise pixels.
 * 
 * @return A cv::Mat object representing the generated noise image.
 * 
 */

cv::Mat getNoiseImage(int width, int height, int minVal, int maxVal) {
    cv::Mat noiseImage = cv::Mat::zeros(height, width, CV_64FC3);
    
    // Define random device and generator
    std::random_device rd;
    std::mt19937 generator(rd());

    // Define uniform distribution with provided min and max values
    std::uniform_int_distribution<> dist(minVal, maxVal);

    // Iterate over each pixel and channel to fill with Gaussian noise
    for(int y = 0; y < noiseImage.rows; y++) {
        for(int x = 0; x < noiseImage.cols; x++) {
            for(int c = 0; c < 3; c++) { 
                double noise = dist(generator);
                noiseImage.at<cv::Vec3d>(y, x)[c] = noise;
            }
        }
    }
    return noiseImage;
}

/**
 * The function "myThreshWeighted" applies thresholding to the input image based on specified thresholds, and multiples
 * the resulting values with percentatge of the max value as weight. The percentage is specfied by 'weightPercent'
 * 
 * @param img The input image to which thresholding is applied.
 * 
 * @param postThresh An integer representing the threshold for positive values.
 * 
 * @param negThresh An integer representing the threshold for negative values.
 * 
 * @param weightPercent A double value representing the percentage of the maximum pixel value to use as a weight.
 * 
 * @return A cv::Mat object representing the thresholded image with weighted values.
 */
cv::Mat myThreshWeighted(cv::Mat &img, int postThresh, int negThresh, double weightPercent){
    cv::Mat neg, pos;
    pos = (img >= postThresh) / 255;
    neg = (img <= negThresh) / 255;

    cv::Mat out;  
    out = pos + neg;

    out.convertTo(out, CV_64F);

    double minVal, maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);
    
    // out = out * (weightPercent*maxVal);
    cv::multiply(out, img, out);
    displayImage(out, false, false);

    return out;
}