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
#include <cmath>


#include <distanceutils.h>
#include <utils.h>


bool DistanceFinder::pathOpened(std::string dirpath){
    std::ifstream file(dirpath.c_str());
    if (!(file.is_open())) {
        std::cerr << "Error opening file: " << dirpath << std::endl;
        return false;
    }
    return true;
}

DistanceFinder::DistanceFinder(
    std::string featurePath,
    std::string targetPath,
    std::string distanceMethodKey
    ){
        bool status = pathOpened(featurePath);

        if(!status){
            printf("Cannot open features %s\n", featurePath);
            return;
        }

        this->featurePath = featurePath;
        this->targetPath = targetPath; 
        this->distanceName = distanceMethodKey;
        this->distanceComputer = getDistanceMethod(this->distanceName);

        loadFeatures();
}

/*
read the feature vectors csv file, from this->featurePath, and
compute distances to the vector corresponding to this->targetPath
save distances in a disances.csv 
*/

bool DistanceFinder::loadFeatures(){   
    std::ifstream featurecsv(featurePath);
    std::string line;
    int lineNum = 0;
    std::string imPath;
    while(std::getline(featurecsv, line)){
        std::stringstream ss(line);
        std::string element;
        std::vector<double> feature;
        std::vector<std::vector<double>> features;
        if(lineNum%2!=0){
            while(std::getline(ss, element, ',')){
                if(element == "<SEP>"){
                    // myPrintVec(feature);
                    features.push_back(feature);
                    feature = std::vector<double>();
                } else {
                double value;
                std::istringstream(element) >> value;
                feature.push_back(value);
                }
            }
            features.push_back(feature);
            featureMap.insert(
                std::pair<std::string, std::vector<std::vector<double>>>(imPath, features)
            );
        }
        else{
            imPath = line;
        }
        
        lineNum+=1;
    
    }

    return true;
}


bool DistanceFinder::computeDistances(){
    std::vector<double> distances(featureMap.size());
    std::cout << "\n initialized a distances array of size " << featureMap.size() << "\n";
    
    std::vector<std::string> imPaths(featureMap.size());
    
    std::map<std::string, std::vector<std::vector<double>>>::iterator it = featureMap.begin();

    // cv::Mat targetMat(featureMap[targetPath]);

    // //#################
    // std::cout << "target path : \t " << targetPath <<"\n";
    // printmat(targetMat, 5);
    // //#################
    std::vector<std::vector<double>> targetVec(featureMap[targetPath]);

    int pos = 0;

    while(it != featureMap.end()){
        std::string imPath = it->first;
        // std::cout << "\nloaded impath : \t " << imPath <<" loading MAT\n";
        // cv::Mat featureMat(it->second);
        std::vector<std::vector<double>> featureVec = it->second;
        // std::cout << "\nloaded featurevec, computing distances\n";
        // printmat(featureMat, 4);
        // std::cout <<"\nerror not in printmat\n";
        double distance = distanceComputer(targetVec, featureVec);
        // std::cout << "\ncomputed distances\n";
        distances[pos] = distance;
        imPaths[pos] = imPath;

        pos++;
        it++;
    }
    
    bool maximize = toMinOrMax(distanceName);

    std::vector<size_t> sortedDistInds = sortIndices(distances, maximize);

    for(size_t i = 0 ; i < sortedDistInds.size() ; i++){
        size_t sortedPos = sortedDistInds[i];
        distancesSorted.push_back(distances[sortedPos]);
        imPathsDistSorted.push_back(imPaths[sortedPos]);
    }

    return true;
}


bool DistanceFinder::getSimilarImages(int numImages, std::string mode){
    cv::Mat viztarget = cv::imread(targetPath, cv::IMREAD_COLOR);
    cv::Mat vizimg;
    cv::namedWindow("target image");
    cv::namedWindow("similar images");

    cv::imshow("target image", viztarget);
    cv::waitKey(0);

    for(size_t i = 0 ; i < imPathsDistSorted.size() ; i++){
        std::string imPath = imPathsDistSorted[i+1]; 
        double distance = distancesSorted[i+1];
        std::cout << "\n distance : " << distance << "\n";
        if (mode == "show"){
            vizimg = cv::imread(imPath, cv::IMREAD_COLOR);
            std::cout << "\n SIMILAR IMAGE : " << i << " path : " << imPath << "\n";
            cv::imshow("similar images", vizimg);
            cv::waitKey(0);
        }    
        else if (mode == "save") {
            std::cout << "\n mode = 'save' in getSimilarImages is NOT IMPLEMENTED\n";
            return false;
        }
        
        if ( numImages==0 ){
            cv::destroyWindow("similar images");
            cv::destroyWindow("target image");
            break;
        }

        numImages--;
    }

    return true;
}

distanceMethod getDistanceMethod(std::string distanceMethodKey){
    distanceMethod distanceComputer;
    
    if (distanceMethodKey == "EuclideanDistance"){
        distanceComputer = &euclideanDistance;
    } else if(distanceMethodKey == "HistogramIntersection"){
        distanceComputer = &HistogramIntersection;
    } else if(distanceMethodKey == "upperLowerCropHistIntersect"){
        distanceComputer = &upperLowerCropHistIntersect;
    }

    else {
        std::cout << "\n DISTANCE METHOD INPUTTED INCORRECTLY OR NOT AT ALL \n";
        distanceComputer = &euclideanDistance;
    }

    return distanceComputer;
}

bool toMinOrMax(std::string &distanceKey){
    bool maximize;

    if (distanceKey == "EuclideanDistance"){
        maximize = false;
    } else if(distanceKey == "HistogramIntersection"){
        maximize = true;
    } else if(distanceKey == "upperLowerCropHistIntersect"){
        maximize = true;
    }

    else {
        std::cout << "\n DISTANCE METHOD INPUTTED INCORRECTLY OR NOT AT ALL \n";
        maximize = false;
    }

    return maximize;
}

double rawEuclideanDistance(cv::Mat mat1, cv::Mat mat2){
    cv::Mat temp1;

    cv::pow(mat1 - mat2, 2, temp1);
    
    cv::Scalar channelSums = cv::sum(temp1);
    double distance = channelSums[0] + channelSums[1] + channelSums[2] + channelSums[3];
    
    distance = sqrt(distance);
    
    return distance;
}

double rawHistogramIntersection(cv::Mat hist1, cv::Mat hist2){
    cv::Mat histIntersect = cv::min(hist1, hist2);

    cv::Scalar channelSums = cv::sum(histIntersect);
    double similarity = channelSums[0] + channelSums[1] + channelSums[2] + channelSums[3];
    
    return similarity;
}


double euclideanDistance(std::vector<std::vector<double>> vec1, 
                        std::vector<std::vector<double>> vec2){
    
    if(vec1.size() > 1){
        std::cout << "\n NUMBER OF FEATURES GREATER THAN 1 FOR EUCLIDEAN DISTANCE COMPUTATION";
        std::cout << "EUCLIDEAN DISTANCE FOR MULTIPLE FEATURES PER IMAGE IS NOT DEFINED\n";
    }

    cv::Mat mat1(vec1[0]);
    cv::Mat mat2(vec2[0]);
    
    double distance;

    distance = rawEuclideanDistance(mat1, mat2);
    
    return distance;
}

double HistogramIntersection(std::vector<std::vector<double>> vec1, 
                        std::vector<std::vector<double>> vec2){
    
    if(vec1.size() > 1){
        std::cout << "\n NUMBER OF FEATURES GREATER THAN 1 FOR EUCLIDEAN DISTANCE COMPUTATION";
        std::cout << "EUCLIDEAN DISTANCE FOR MULTIPLE FEATURES PER IMAGE IS NOT DEFINED\n";
    }
    
    cv::Mat hist1(vec1[0]);
    cv::Mat hist2(vec2[0]);
    
    double similarity = rawHistogramIntersection(hist1, hist2);

    return similarity;
}



/*
1. interesects will be in range 0-1. 
2. we want the value to be more than much higher if both upper and lower match
    than we just 1 of the Crops match, infact, we we want to discriminate 
    if even one of the Crops is a bad match.
3. we choose to make the intersects >0 (by multiplying by 100) as good matches are
    usually above 0.01. Finally, we take the product of the upper and lower intersects.
4. With reference to the 2nd point, if a match is less than 1% of maximum possible value aka < 0.01, we set
    the intersect to 0.9. so that the overall product is driven to be lower.
    We cap the intersect, so as to not discriminate too much. We consider 1 quart matches over polarly opposite images.
*/
double upperLowerCropHistIntersect(std::vector<std::vector<double>> vec1,
                                    std::vector<std::vector<double>> vec2){

    if(vec1.size() == 1){
        std::cout << "\n EXPECTED MORE THAN 1 FEATURE VECTOR \n(2 specificially, upper and lower histogram)\n";
    }

    cv::Scalar channelSums;
    
    cv::Mat upper1(vec1[0]);
    cv::Mat lower1(vec1[1]);
    cv::Mat upper2(vec2[0]);
    cv::Mat lower2(vec2[1]);

    double upperIntersectValue = rawHistogramIntersection(upper1, upper2);
    
    double lowerIntersectValue = rawHistogramIntersection(lower1, lower2);
    

    // if(upperIntersectValue>0.01){
    //     upperIntersectValue = upperIntersectValue * 100;
    // } else {
    //     upperIntersectValue = 0.9;
    // }

    // if(lowerIntersectValue>0.01){
    //     lowerIntersectValue = lowerIntersectValue * 100;
    // } else {
    //     lowerIntersectValue = 0.9;
    // }

    // double similarity = upperIntersectValue * lowerIntersectValue;

    //####################
    upperIntersectValue = upperIntersectValue * 100;
    lowerIntersectValue = lowerIntersectValue * 100;
    double similarity = upperIntersectValue * lowerIntersectValue;
    //####################
    return similarity;
}


