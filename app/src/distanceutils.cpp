/**
 * Names : Kumar Selvakumaran, Neel Adke,
 * date : 2/13/2024
 * Purpose : This is file contains or imports all the required classes and fuction required for distance computation. 
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
#include <float.h>


#include <distanceutils.h>
#include <utils.h>
#include <featureutils.h>

/**
 * The function "pathOpened" checks if a file exists at the specified directory path.
 * 
 * @param dirpath The parameter 'dirpath' is a string representing the directory path
 * where the existence of a file needs to be checked.
 * 
 * @return A boolean value indicating whether a file exists at the specified directory path.
 * Returns 'true' if the file exists and can be opened successfully, otherwise returns 'false'.
 */
bool DistanceFinder::pathOpened(std::string dirpath){
    std::ifstream file(dirpath.c_str());
    if (!(file.is_open())) {
        std::cerr << "Error opening file: " << dirpath << std::endl;
        return false;
    }
    return true;
}

/**
 * The constructor "DistanceFinder" initializes an object of the DistanceFinder class.
 * It takes several parameters to set up the object, including a path to the feature csv file,
 * keys to specify distance calculation method, and a key to specify the  target feature 
 * extraction method used to generate features for the target image.
 * 
 * @param featurePath A string representing the path to the feature csv file.
 * 
 * @param targetPath A string representing the path to the target image path.
 * 
 * @param distanceMethodKey A string representing the key or name of the distance calculation method.
 * 
 * @param targetFeaturekey A string representing the key or name of the target feature extraction method.
 * 
 * Upon construction, the constructor checks if the feature file can be opened successfully
 * using the "pathOpened" function. If it cannot be opened, an error message is printed, and the constructor returns.
 * 
 * After validating the feature file, the constructor initializes member variables of the DistanceFinder object:
 * - 'featurePath': Path to the feature file.
 * - 'targetPath': Path to the target file.
 * - 'distanceName': Name of the distance calculation method.
 * - 'distanceComputer': Pointer to the distance calculation function obtained using the 'distanceMethodKey' key.
 * - 'targetFeatureName': Name of the target feature extraction method.
 * - 'targetFeatureComputer': Pointer to the target feature extraction function obtained the 'targetFeatureName' key.
 * 
 * Finally, the constructor calls the "loadFeatures" function to load features from the feature csv file.
 */
DistanceFinder::DistanceFinder(
    std::string featurePath,
    std::string targetPath,
    std::string distanceMethodKey,
    std::string targetFeaturekey
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
        this->targetFeatureName = targetFeaturekey;
        this->targetFeatureComputer = getFeatureMethod(this->targetFeatureName);
        loadFeatures();
}

/**
 * The function "loadFeatures" is a member function of the DistanceFinder class, which loads features from a CSV file.
 * 
 * This function reads the feature CSV file specified by the member variable 'featurePath'.
 * It parses each line of the CSV file to extract image paths and corresponding feature vectors.
 * 
 * Features are expected to be formatted such that every odd-numbered line contains an image path,
 * and the subsequent even-numbered line contains comma-separated feature values.
 * 
 * @return A boolean value indicating the success of loading features.
 * Returns 'true' if the features are loaded successfully, otherwise returns 'false'.
 * 
 * Upon successful loading of features, the function populates the member variable 'featureMap'.
 * 'featureMap' is a map with image paths as keys and corresponding feature vectors as values.
 * 
 * Note: If the feature CSV file does not follow the expected format, the function may not load features correctly.
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

/**
 * The function "computeDistances" calculates distances between a target image and the set of feature vectors
 * stored in the feature map. It populates the member variable 'distances'with computed distances and sorts them in ascending or
 * descending order based on the specified distance calculation method.
 * 
 * @return A boolean value indicating the success of the computation. Returns 'true' if computation
 * is successful, otherwise returns 'false'.
 */
bool DistanceFinder::computeDistances(){
    std::vector<double> distances(featureMap.size());
    std::cout << "\n initialized a distances array of size " << featureMap.size() << "\n";
    
    std::vector<std::string> imPaths(featureMap.size());
    
    std::map<std::string, std::vector<std::vector<double>>>::iterator it = featureMap.begin();

    //------- READING FROM FEATURES.CSV ------
    // std::vector<std::vector<double>> targetVec = featureMap[targetPath] ;
    //----------------------------------------

    //------- RECOMPUTING IN RUNTIME ----------
    std::vector<std::vector<double>> targetVec;
    cv::Mat targetImage = cv::imread(targetPath, cv::IMREAD_COLOR);
    targetVec = targetFeatureComputer(targetImage);
    //-----------------------------------------

    int pos = 0;

    bool maximize = toMinOrMax(distanceName);
    
    while(it != featureMap.end()){
        double distance;
        std::string imPath = it->first;
        std::vector<std::vector<double>> featureVec = it->second;
        if(distanceName != "stridedEuclideanDistance"){
            distance = distanceComputer(targetVec, featureVec);
        }
        //######------SLIDING DISANCE COMPUTER------###########
        else {
            rawDistanceMethod distancegetter = &rawEuclideanDistance;
            distance = stridedDistanceComputer(targetVec, featureVec, distancegetter, maximize);
        }
        //######------SLIDING DISTANCE CMPUTER END---##########
        
        distances[pos] = distance;
        imPaths[pos] = imPath;

        pos++;
        it++;
    }

    std::vector<size_t> sortedDistInds = sortIndices(distances, maximize);

    for(size_t i = 0 ; i < sortedDistInds.size() ; i++){
        size_t sortedPos = sortedDistInds[i];
        distancesSorted.push_back(distances[sortedPos]);
        imPathsDistSorted.push_back(imPaths[sortedPos]);
    }

    return true;
}

/**
 * The function "getSimilarImages" displays or saves similar images to the target image based on the provided mode.
 * 
 * @param numImages An integer representing the number of similar images to display or save.
 * 
 * @param mode A string indicating the mode of operation. It can be either "show" to display similar images or "save"
 * to save them.
 * 
 * The function reads the target image from the specified path using OpenCV's "imread" function and displays it in a
 * named window titled "target image". It then iterates over the sorted list of image paths and corresponding distances
 * and displays or saves the specified number of similar images based on the provided mode.
 * 
 * If the mode is "show", similar images are displayed one by one in a named window titled "similar images" using OpenCV's
 * "imshow" function, and the user can navigate through them using keyboard inputs until the user closes the window.
 * 
 * If the mode is "save", similar images are saved to the "./images/" directory with filenames corresponding to their
 * index in the sorted list appended with ".jpg" extension using OpenCV's "imwrite" function.
 * 
 * After displaying or saving the specified number of similar images, if the number of images to display is zero,
 * the function closes the named windows "similar images" and "target image" using OpenCV's "destroyWindow" function
 * and returns 'true' to indicate successful execution.
 */
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
            vizimg = cv::imread(imPath, cv::IMREAD_COLOR);
            std::cout << "\n SIMILAR IMAGE : " << i << " path : " << imPath << "\n";
            std::string savename = "./images/" + std::to_string(i) + ".jpg";
            cv::imwrite(savename, vizimg);
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

bool DistanceFinder::getDisSimilarImages(int numImages, std::string mode){
    cv::Mat viztarget = cv::imread(targetPath, cv::IMREAD_COLOR);
    cv::Mat vizimg;
    cv::namedWindow("target image");
    cv::namedWindow("similar images");

    cv::imshow("target image", viztarget);
    cv::waitKey(0);

    for(size_t i = imPathsDistSorted.size() - 1  ; i > 0 ; i--){
        std::string imPath = imPathsDistSorted[i]; 
        double distance = distancesSorted[i];
        std::cout << "\n distance : " << distance << "\n";
        if (mode == "show"){
            vizimg = cv::imread(imPath, cv::IMREAD_COLOR);
            std::cout << "\n SIMILAR IMAGE : " << i << " path : " << imPath << "\n";
            cv::imshow("similar images", vizimg);
            cv::waitKey(0);

            //################################3
            std::string savename = "./images/" + std::to_string(i) + ".jpg";
            cv::imwrite(savename, vizimg);
            //#################################
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
        distanceComputer = &simpeEuclideanDistance;
    } else if(distanceMethodKey == "HistogramIntersection"){
        distanceComputer = &HistogramIntersection;
    } else if(distanceMethodKey == "upperLowerCropHistIntersect"){
        distanceComputer = &upperLowerCropHistIntersect;
    }

    else {
        std::cout << "\n DISTANCE METHOD INPUTTED INCORRECTLY OR NOT AT ALL \n";
        distanceComputer = &simpeEuclideanDistance;
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


double simpeEuclideanDistance(std::vector<std::vector<double>> vec1, 
                        std::vector<std::vector<double>> vec2){
    
    std::vector<double>features1;
    std::vector<double>features2;

    for(size_t i  = 0 ; i < vec1.size() ; i++){
        

        
        //---------------NORMALIZE-----------------
        // cv::Mat temp1(vec1[i]);
        // cv::Mat temp2(vec2[i]);

        // std::pair<double, double> minMaxVals;
        // minMaxVals = myNormMat(temp1, temp1);
        // minMaxVals = myNormMat(temp2, temp2);
        
        // temp1 = temp1 * 1000;
        // temp2 = temp2 * 1000;

        // std::vector<double> vecTemp1 = temp1;
        // std::vector<double> vecTemp2 = temp2;
        //-------------------------------------------
        //------------DONT NORMALIZE-----------------
        std::vector<double> vecTemp1 = vec1[i];
        std::vector<double> vecTemp2 = vec2[i];        

        features1.reserve(features1.size() + std::distance(vecTemp1.begin(), vecTemp1.end()));
        features1.insert(features1.end(), vecTemp1.begin(), vecTemp1.end());   

        features2.reserve(features2.size() + std::distance(vecTemp2.begin(), vecTemp2.end()));
        features2.insert(features2.end(), vecTemp2.begin(), vecTemp2.end());   
    }

    cv::Mat mat1(features1);
    cv::Mat mat2(features2);

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

double stridedDistanceComputer(std::vector<std::vector<double>> target,
                                std::vector<std::vector<double>> vec2,
                                rawDistanceMethod distanceGetter,
                                bool maximize){
    
    int numFeatures = target.size();
    int numCrops = vec2.size() / numFeatures;
    double optDistance;
    std::cout << "\n maximize : " << maximize << "\n"; 
    if(maximize == true){
        optDistance = 0;
    } else {
        optDistance = DBL_MAX;
    }
    std::vector<double> targetVec;
    

    for (int i = 0 ; i < numFeatures ; i++){
        targetVec.reserve(targetVec.size() + std::distance(target[i].begin(), target[i].end()));
        targetVec.insert(targetVec.end(), target[i].begin(), target[i].end());  
    }

    std::cout << "\n going to compute distances for each crop\n";
    for(int i = 0 ; i < numCrops ; i ++){
        std::vector<double> featureVec;
        std::cout << "\n crop num :" << i << "\n";
        
        for(int j = 0 ; j < numFeatures; j++){
            featureVec.reserve(featureVec.size() + std::distance(vec2[i+j].begin(), vec2[i+j].end()));
            featureVec.insert(featureVec.end(), vec2[i+j].begin(), vec2[i+j].end());   
        }
        
        std::cout << "\n made concatenated feature vector, computing distance now \n";

        cv::Mat mat1(targetVec);
        cv::Mat mat2(featureVec);

        double distance;

        distance = distanceGetter(mat1, mat2);

        std::cout << "\n computed distance for crop : " << distance << "\n";

        
        if(maximize == true){
            if(distance > optDistance){
                optDistance = distance;
            }
        } else {
            if(distance < optDistance){
                optDistance = distance;
            }
        }
            
    }

    std::cout << "\n final distance : "<< optDistance << "\n";
    return optDistance;   
}

