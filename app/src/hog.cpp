/*
This progrogam stores the different feature extraction functions
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

cv::Mat hog::computeGrad(cv::Mat &src, bool isX){
    cv::Mat dst;
    dst = src.clone();
    
    cv::Mat grad;

    dst.convertTo(dst, CV_8U);
    cv::medianBlur(dst, dst, 5);

    dst.convertTo(dst, CV_64F);

    float sobelf[5] = {1, 2, 0, -2, -1};
    float gaussianf[5] = {1, 2, 4, 2, 1};

    cv::Mat filterRow(1, 5, CV_64F, sobelf);
    cv::Mat filterCol(1, 5, CV_64F, gaussianf);

    if(isX == false){
        cv::swap(filterCol, filterRow);
    }

    cv::sepFilter2D(dst, grad, dst.depth(), filterRow, filterCol);

    grad =  myThresh(grad, threshPositive, threshNegative);
    grad.convertTo(grad, CV_64F);

    return grad;
}  

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

/*
orientation values range from 0 - 255, 127 corresponds to 0
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
            // double lowComponentR = lowRatioR * rMag;          
            // double highComponentR = (1 - lowRatioR) * rMag;

            //------------NO SPLITTING MAGNITUDE -------------
            double lowComponentR = rMag;          
            double highComponentR = 0;           

            const int constGlobalOrientationBins{globalOrientationBins}; 


            hist.at<Vec5d>(rBin, gBin)[orBinLowR] += lowComponentR;
            hist.at<Vec5d>(rBin, gBin)[orBinHighR] += highComponentR;


            int orBinLowG = static_cast<int>(gOr/orientationBinSize);
            int orBinHighG = std::min(globalOrientationBins - 1, orBinLowG + 1);
            double lowRatioG;
            if(orBinLowG != globalOrientationBins - 1){
                lowRatioG = (gOr - (orBinLowG*orientationBinSize)) / ((orBinHighG*orientationBinSize) - (orBinLowG*orientationBinSize));
            } else {
                lowRatioG = 0.5;
            }

            // --------------SPLITTING MAGNITUDE -------------
            // double lowComponentG = lowRatioG * gMag;          
            // double highComponentG = (1 - lowRatioG) * gMag;

            //---------------NO SPLITTING MAGNITUDE -----------
            double lowComponentG = gMag;          
            double highComponentG = 0;

            hist.at<Vec5d>(rBin, gBin)[orBinLowG] += lowComponentG;
            hist.at<Vec5d>(rBin, gBin)[orBinHighG] += highComponentG;

        }    
    }

    return hist;

}

cv::Mat hog::computeGlobalHog(cv::Mat &src){
    // hog hogComputer(5, 160, -160, 8, 5);

    //############
    printmat(src, 3);
    displayImage(src);
    //############

    cv::Mat gradX;
    gradX = computeGrad(src, true);
    cv::Mat gradY;
    gradY = computeGrad(src, false);

    //############
    printmat(gradX, 3);
    displayImage(gradX);
    //############

    //############
    printmat(gradY, 3);
    displayImage(gradY);
    //############
    
    cv::Mat magMat;
    magMat = computeMagnitude(gradX, gradY);

    cv::Mat orMat;
    orMat = computeOrientation(gradX, gradY);   

    

    int dimArray[3] = {globalIntensityBins,
                        globalIntensityBins,
                        globalOrientationBins};

    // int dimArray[3] = {3,2,4}
    
    cv::Mat hist = cv::Mat::zeros(3, dimArray, CV_64FC1);



    std::cout << " histogram size: " << hist.size() << " " << hist.dims << "\n";

    std::cout << "\nmagmat dets : " << magMat.size() << " " << magMat.dims << " continuous ? : " << magMat.isContinuous();
    std::cout << " width : " << magMat.size().width << " height : " << magMat.size().height << "\n";
    printmat(magMat, 3);
    displayImage(magMat);

    std::cout << "\n orientation dets" << orMat.size() << "  " << orMat.dims << "continuous ? : " << orMat.isContinuous();
    std::cout << " width : " << orMat.size().width << " height : " << orMat.size().height << "\n"; 
    printmat(orMat, 3);
    displayImage(orMat);

    double maxVal, minVal;
    
    cv::minMaxLoc(magMat, &minVal, &maxVal);
    int intensityBinSize = static_cast<int>(maxVal / globalIntensityBins);
    intensityBinSize = std::max(1, intensityBinSize);
    //###########
    std::cout << "\nmax val : " << maxVal;
    std::cout << "\n globalIntensityBins : " << globalIntensityBins << "\n";
    //###########

    cv::minMaxLoc(orMat, &minVal, &maxVal);
    int orientationBinSize = static_cast<int>(maxVal / globalOrientationBins);
    orientationBinSize = std::max(1, orientationBinSize);
    //###########
    std::cout << "\nmaxVal : " << maxVal << "min val : " << minVal;
    std::cout << "\n globalOrientationBins: " << globalOrientationBins << "\n";
    //###########

    //###########
    std::cout << "\nintensisity bin size : " << intensityBinSize;
    std::cout << "\n orientation bin size : " << orientationBinSize << "\n";
    //###########
    int numElements = magMat.size().width * magMat.size().height;
    int numRows = magMat.size().height;
    int numCols = magMat.size().width;

    /*
    - iterating over magMat and alloting the current magnitude to respective orientation bins
    - allotion policy : if orbin_i < orientation(x) < orbin_i+1,
    for each color (green and red)
        floorbin_ratio = (orientation(elind)  - orbin_i) / (orbin_i+1 - orbin_i)
        // magbin will be the chromaticity by doing rowBin r/r+g+b, and colBin g/r+g+b        
        hist[magbin, orbin_i] += floorbin_ratio * magnitude(x)
        hist[magbin, orbin_i+1] += (1 - floorbin_ratio) * magnitude(x)
    
    */

    /*
        // QUESTION : WE ARE STORING ONLY GREEN AND RED MAGNITIUDE, WONT WE 
            LOSE BLUE'S MAGNITUDE?
            ANSWER :   rbin and gbin gives us the ratio. 
            corresponding blue's ratio wrt green * green's magnitude
                or
            corresponding blue's ration wrt red * red's magnitiude should
            give us blue's magnitude 
    */
    /*
        cv::Vec indices will read invalid indices, whereas cv::Point wont.
        check your indexing with cv::Point, and then use cv::Vec. cv::Point
        is not used for final implementation because it cant represent more than
        3 channels. 
    */

    //###########
    // int dimArr[3] = {3,2,3};
    // magMat = cv::Mat::zeros(3, dimArr, CV_64F);
    // numElements = magMat.size().width * magMat.size().height;
    // numRows = magMat.size().height;
    // numCols = magMat.size().width;
    //############

    // cv::Vec3f* magPtr = magMat.ptr<cv::Vec3f>(0);
    // cv::Point3f* magPtr = magMat.ptr<cv::Point3f>(0);
    // double* orPtr = orMat.ptr<double>(0);
    // double* histPtr = hist.ptr<double>(0);

    cv::minMaxLoc(magMat, &minVal, &maxVal);
    if((magMat.isContinuous() == true) && (orMat.isContinuous() == true) && (hist.isContinuous() == true)){
        for (int elInd = 0 ; elInd < numElements ; elInd ++){
            int rowInd = (int)elInd / numCols;
            int colInd = (int)elInd % numCols;

            // cv::Vec<double, 3> pixel;
            // magMat.at<cv::Vec<double, 3>>(rowInd, colInd)[0] = 100;
            // pixel = magMat.at<cv::Vec<double, 3>>(rowInd, colInd);
            
            //###############################
            // cv::Vec<double, 3> histind;
            // histind[0] = rowInd;
            // histind[1] = colInd;

            // magMat.at<cv::Vec<double, 3>>(rowInd, colInd)[0] = 100;
            // pixel = magMat.at<cv::Vec<double, 3>>(rowInd, colInd);
            //###############################

            //###############################
            // std::cout<< "\n[ ";
            // bool carryon = true;
            // for(int i = 0  ; i < 3 ; i ++){
                
            //     // std::cout << "working : " << magMat.at<cv::Vec<double, 3>>(rowInd, colInd)[i]  << " simple's permuations \n";
            //     // int histind[] = {rowInd, colInd, i};
            //     // std::cout<< "rowInd, colInd, i " << magMat.at<double>(histind)<<", ";
            //     // int histind2[] = { colInd, rowInd, i};
            //     // std::cout<< "colInd, rowInd, i " << magMat.at<double>(histind2)<<", ";
            //     // int histind3[] = {rowInd, i, colInd};
            //     // std::cout<< "rowInd, i, colInd " << magMat.at<double>(histind3)<<", ";
            //     // int histind4[] = {i, rowInd, colInd};
            //     // std::cout<< "i, rowInd, colInd: " << magMat.at<double>(histind4)<<", ";
            //     // int histind1[] = {colInd, i, rowInd};
            //     // std::cout<< "colInd, i, rowInd : " << magMat.at<double>(histind1)<<", ";
            //     // std::cout << "\n";
            //     // int histind7[] = {i, colInd, rowInd};
            //     // std::cout<< "i, colInd, rowInd: " << magMat.at<double>(histind7)<<", ";
            //     // std::cout << "\n";
                
                
                
                
            //     if(!(magMat.at<cv::Vec<double, 3>>(rowInd, colInd)[i] == magMat.at<double>(histind))){
            //         std::cout << "\n\n   INDEXING NOT EQUAL AT : " << rowInd << " " << colInd << " " << i <<"\n\n";
            //         carryon = false;
            //         break;
            //     }
            //     cv::waitKey(100);
            // }
            // if(carryon != true){
            //     break;
            // }
            // std::cout << "]\n";
            //###############################


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
            
            double chromR = rMag / std::max(divisor, 1.0);
            double chromG = gMag / std::max(divisor, 1.0);

            int rBin = (int) (chromR * (globalIntensityBins - 1) + 0.5);
            int gBin = (int) (chromG * (globalIntensityBins - 1) + 0.5);

            // int magBin = static_cast<int>(currMag/intensityBinSize);
            int orBinLowR = static_cast<int>(rOr/orientationBinSize);
            int orBinHighR = std::min(globalOrientationBins - 1, orBinLowR + 1);
            double lowRatioR;
            if(orBinLowR != globalOrientationBins - 1){
                lowRatioR = (rOr - (orBinLowR*orientationBinSize)) / ((orBinHighR*orientationBinSize) - (orBinLowR*orientationBinSize));
            } else {
                lowRatioR = 0.5;
            }

            // ----------- SPLITTING MAGNITUDE ---------------
            // double lowComponentR = lowRatioR * rMag;          
            // double highComponentR = (1 - lowRatioR) * rMag;

            //------------NO SPLITTING MAGNITUDE -------------
            double lowComponentR = rMag;          
            double highComponentR = 0;           

            const int constGlobalOrientationBins{globalOrientationBins}; 

            // hist.at<cv::Vec<double, constGlobalOrientationBins>>(rBin, gBin)[orBinLowR] = lowComponentR;
            // hist.at<cv::Vec<double, constGlobalOrientationBins>>(rBin, gBin)[orBinHighR] = highComponentR;
            //_________________________________________________________________________________________
            //#####################
            // if((orBinHighR != 4)&&(orBinHighR != 4)){
            std::cout<<"\n\nat hist[" << rBin << ", " << gBin << ", " << orBinLowR <<"] : oldvalue : " << hist.at<Vec5d>(rBin, gBin)[orBinLowR];
            std::cout<<"\nat hist[" << rBin << ", " << gBin << ", " << orBinHighR <<"] : oldvalue : " << hist.at<Vec5d>(rBin, gBin)[orBinHighR];
            // }
            //#####################

            hist.at<Vec5d>(rBin, gBin)[orBinLowR] += lowComponentR;
            hist.at<Vec5d>(rBin, gBin)[orBinHighR] += highComponentR;

            //#####################
            // if((orBinHighR != 4)&&(orBinHighR != 4)){
            std::cout<<"\n\nat hist[" << rBin << ", " << gBin << ", " << orBinLowR <<"] : NEW VALUE : " << hist.at<Vec5d>(rBin, gBin)[orBinLowR];
            std::cout<<"\nat hist[" << rBin << ", " << gBin << ", " << orBinHighR <<"] : NEW VALUE : " << hist.at<Vec5d>(rBin, gBin)[orBinHighR];
            // }
            //#####################

            int orBinLowG = static_cast<int>(gOr/orientationBinSize);
            int orBinHighG = std::min(globalOrientationBins - 1, orBinLowG + 1);
            double lowRatioG;
            if(orBinLowG != globalOrientationBins - 1){
                lowRatioG = (gOr - (orBinLowG*orientationBinSize)) / ((orBinHighG*orientationBinSize) - (orBinLowG*orientationBinSize));
            } else {
                lowRatioG = 0.5;
            }

            // --------------SPLITTING MAGNITUDE -------------
            // double lowComponentG = lowRatioG * gMag;          
            // double highComponentG = (1 - lowRatioG) * gMag;

            //---------------NO SPLITTING MAGNITUDE -----------
            double lowComponentG = gMag;          
            double highComponentG = 0;

            // hist.at<cv::Vec<double, constGlobalOrientationBins>>(rBin, gBin)[orBinLowG] = lowComponentG;
            // hist.at<cv::Vec<double, constGlobalOrientationBins>>(rBin, gBin)[orBinHighG] = highComponentG;
            //_________________________________________________________________________________________
            //#####################
            // if((orBinHighR != 4)&&(orBinHighR != 4)){
            std::cout<<"\n\nat hist[" << rBin << ", " << gBin << ", " << orBinLowG <<"] : oldvalue : " << hist.at<Vec5d>(rBin, gBin)[orBinLowG];
            std::cout<<"\nat hist[" << rBin << ", " << gBin << ", " << orBinHighG <<"] : oldvalue : " << hist.at<Vec5d>(rBin, gBin)[orBinHighG];
            // }
            //#####################

            hist.at<Vec5d>(rBin, gBin)[orBinLowG] += lowComponentG;
            hist.at<Vec5d>(rBin, gBin)[orBinHighG] += highComponentG;

            //#####################
            // if((orBinHighR != 4)&&(orBinHighR != 4)){
            std::cout<<"\n\nat hist[" << rBin << ", " << gBin << ", " << orBinLowG <<"] : NEW VALUE : " << hist.at<Vec5d>(rBin, gBin)[orBinLowG];
            std::cout<<"\nat hist[" << rBin << ", " << gBin << ", " << orBinHighG <<"] : NEW VALUE : " << hist.at<Vec5d>(rBin, gBin)[orBinHighG];
            // }
            //#####################

            // std::cout << "\n" << typeid(magPtr[elInd]).name() << " " << magPtr[elInd][0] << " " << magPtr[elInd][1] << magPtr[elInd][2]; 
            // int histRowInd =  

            // ############
            // if(elInd > (170*200)){
            //     break;
            // }
            // cv::waitKey(100);
            // ############
        }    
    }
    
    int histels = dimArray[0] * dimArray[1] * dimArray[2];
    for (int  i = 0  ; i < histels ; i+=globalOrientationBins){
        int celInd = i/globalOrientationBins;
        int rowInd = static_cast<int>(celInd / globalIntensityBins);
        int colInd = static_cast<int>(celInd % globalIntensityBins);
        Vec5d binCell = hist.at<Vec5d>(rowInd, colInd);
        
        std::cout << "hist[" << rowInd<<", "<< colInd << "] = [";
        for(int j = 0 ; j < globalOrientationBins ; j++){
            std::cout << binCell[j] <<", ";
        }
        std::cout <<"]\n";
    }

    //############

    std::cout << "\n PARSED IMAGE SUCCESSFULLY \n";
    std::cout << " histogram size: " << hist.size() << " " << hist.dims << "\n";

    std::cout << "\nmagmat dets : " << magMat.size() << " " << magMat.dims << " continuous ? : " << magMat.isContinuous();
    std::cout << " width : " << magMat.size().width << " height : " << magMat.size().height << "\n";
    printmat(magMat, 3);
    displayImage(magMat);
    std::cout << "\n orientation dets" << orMat.size() << "  " << orMat.dims << "continuous ? : " << orMat.isContinuous();
    std::cout << " width : " << orMat.size().width << " height : " << orMat.size().height << "\n"; 
    printmat(orMat, 3);
    //#############

    // if(hist.isContinuous() == true){
    //     for (int elind = 0 ; elind < numElements ; elind += globalOrientationBins){
    //         for(int j = 0 ; j < 4 ; j++){    
    //             matPtr[i+j] = i*1.3;
    //         }   
    //     }
    // }

    return hist;            
}




