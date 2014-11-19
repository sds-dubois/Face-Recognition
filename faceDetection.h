#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

CascadeClassifier getCascadeClassifier();
vector<Rect> detectFaces(CascadeClassifier face_classifier, Mat frame);
int detectFacesWebcam();
int showFaces();
