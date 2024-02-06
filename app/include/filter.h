// Group : Kumar Selvakumaran, Neel Adke 
// DATE : 1/26/2024
// Purpose : Contains the declearations of all the functions used.

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>

#ifndef FILTER_H
#define FILTER_H

#include<filter.h>

int greyscale(cv::Mat &src, cv::Mat &dst);
void betterBGRtoBW(cv::Mat &src, cv::Mat &dst);
void ContrastSquare(cv::Mat &src, cv::Mat &dst);
void ContrastSigmoid(cv::Mat &src, cv::Mat &dst);
void printmat(cv::Mat* src);
void Sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst );
int sobelY3x3(cv::Mat &src, cv::Mat &dst );
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
void findFaces(cv::Mat &frame);
void envblurr(cv::Mat &frame);
void bwenv(cv::Mat &frame);
void makeNegative(cv::Mat &frame);
void vizderivative(cv::Mat &src);
void makecartoon(cv::Mat &frame);
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

#endif
