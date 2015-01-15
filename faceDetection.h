#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

CascadeClassifier getFaceCascadeClassifier(void);
CascadeClassifier getEyesCascadeClassifier(void);
CascadeClassifier getMouthCascadeClassifier(void);
CascadeClassifier getNoseCascadeClassifier(void);

Rect selectBestFace(Mat frame, vector<Rect> faces);

vector<Rect> detectFaces(CascadeClassifier face_classifier, Mat frame);
vector<Rect> detectEye(CascadeClassifier eye_classifier, Mat frame ) ;
vector<Rect> detectMouth(CascadeClassifier mouth_classifier, Mat frame ) ;
vector<Rect> detectNose(CascadeClassifier nose_classifier, Mat frame ) ;

void showAllFeatures(Mat frame, vector<Rect> faces);
void showFaces(string file);
void showAllFaces(void);
bool showEyes(string filename,bool verbose);
void showAllEyes(bool verbose) ;
void showLeftRightEyes(string filename) ;
void showEyes(string filename);

// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
bool enhanceDetection(Rect face, vector<KeyPoint> &keypoints_eyes, vector<KeyPoint> &keypoints_mouth, vector<KeyPoint> &keypoints_nose, int detectionType);

void extractCroppedDescriptor(int n, int m , bool verbose) ;
void displaySelectedFeaturesOnFaces(int method) ;
// selectionMethod 1 : information gain / 2 : chi²
void showSelectedFacesFeatures(int n, int m, int nb_feats,  string nb,int selectionMethod) ;
void showPCAfaces(int n, int m,int nb_components,int selectionMethod);
