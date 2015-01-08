#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

CascadeClassifier getFaceCascadeClassifier();
CascadeClassifier getEyesCascadeClassifier(void);
CascadeClassifier getMouthCascadeClassifier(void);
CascadeClassifier getNoseCascadeClassifier(void);
vector<Rect> detectFaces(CascadeClassifier face_classifier, Mat frame);
Rect selectBestFace(Mat frame, vector<Rect> faces);
Rect selectBestFace2(Mat frame, vector<Rect> faces);
void showAllFeatures(Mat frame, vector<Rect> faces);
vector<Rect> detectEye(CascadeClassifier eye_classifier, Mat frame ) ;
vector<Rect> detectMouth(CascadeClassifier mouth_classifier, Mat frame ) ;
vector<Rect> detectNose(CascadeClassifier nose_classifier, Mat frame ) ;
int detectFacesWebcam();
void showFaces(string file);
void showAllFaces(void);
bool showEyes(string filename,bool verbose);
void showAllEyes(bool verbose) ;
void showLeftRightEyes(string filename) ;
void showEyes(string filename);
