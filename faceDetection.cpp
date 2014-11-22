#include "faceDetection.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "boost/filesystem.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

CascadeClassifier getFaceCascadeClassifier()
{
    String face_cascade_name = "../lib/haarcascade_frontalface_alt.xml";
    CascadeClassifier face_cascade;
    face_cascade.load(face_cascade_name);
    return face_cascade;
}

vector<Rect> detectFaces(CascadeClassifier face_classifier, Mat frame )
{
    vector<Rect> faces;
    Mat frame_gray = frame.clone();
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_classifier.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    return faces;
}

int detectFacesWebcam(){
    String face_cascade_name = "../lib/haarcascade_frontalface_alt.xml";
    CascadeClassifier face_cascade;
    string window_name = "Capture - Face detection";

    CvCapture* capture;
    Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){
        printf("--(!)Error loading\n");
        return -1;
    }

    //-- 2. Read the video stream
    capture = cvCaptureFromCAM( -1 );
    if(capture)
    {
        while(true){
            frame = cvQueryFrame( capture );

            //-- 3. Apply the classifier to the frame
            if( !frame.empty() ){
                vector<Rect> faces = detectFaces(face_cascade, frame);
                for( size_t i = 0; i < faces.size(); i++ ){
                    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
                    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
                }
                imshow( window_name, frame );
            }
            else{
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            int c = waitKey(10);
            if( (char)c == 'c' ) {
                break;
            }
        }
    }
    return 0;
}

void showFaces(string file){
	CascadeClassifier face_classifier = getFaceCascadeClassifier();	
	Mat input = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
	
	vector<Rect> faces = detectFaces(face_classifier, input); 
	if(faces.size() != 0){
		rectangle(input,faces.front(),Scalar(0,0,255),1,8,0) ;
		imshow("face",input) ;
		waitKey() ;
	}
	else
		cout << "Aucun visage detecte" << endl ;
}

void showAllFaces(void){
	for (directory_iterator it1("../data/labeled"); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
				showFaces(p2.string()) ;
			}
		}
	}
}

void showAllEyes(void){
	for (directory_iterator it1("../data/labeled"); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
				showEyes(p2.string()) ;
			}
		}
	}
}

CascadeClassifier getEyeLeftCascadeClassifier()
{
    String eye_cascade_name = "../lib/haarcascade_lefteye_2splits.xml";
    CascadeClassifier eye_cascade;
    eye_cascade.load(eye_cascade_name);
    return eye_cascade;
}

CascadeClassifier getEyeRightCascadeClassifier()
{
    String eye_cascade_name = "../lib/haarcascade_righteye_2splits.xml";
    CascadeClassifier eye_cascade;
    eye_cascade.load(eye_cascade_name);
    return eye_cascade;
}

CascadeClassifier getEyesCascadeClassifier()
{
    String eye_cascade_name = "../lib/haarcascade_eye_tree_eyeglasses.xml";
    CascadeClassifier eye_cascade;
    eye_cascade.load(eye_cascade_name);
    return eye_cascade;
}

vector<Rect> detectEye(CascadeClassifier face_classifier, Mat frame )
{
    vector<Rect> eyes;
    Mat frame_gray = frame.clone();
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_classifier.detectMultiScale(frame_gray, eyes, 1.1, 3, 0,Size(30,30));

    return eyes;
}

void showLeftRightEyes(string filename){
	CascadeClassifier lefteye_classifier = getEyeLeftCascadeClassifier();	
	CascadeClassifier righteye_classifier = getEyeRightCascadeClassifier();	
	Mat input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	
	vector<Rect> lefteyes = detectEye(lefteye_classifier, input); 
	vector<Rect> righteyes = detectEye(righteye_classifier, input); 
	if(lefteyes.size() != 0){
		cout << "Nombre d'oeils gauches : " << lefteyes.size() << endl ;
		rectangle(input,lefteyes.front(),Scalar(0,255,0),1,8,0) ;
	}
	if(righteyes.size() != 0){
		cout << "Nombre d'oeils droits : " << righteyes.size() << endl ;
		rectangle(input,righteyes.front(),Scalar(0,0,255),1,8,0) ;
	}
	if(righteyes.size() == 0 && lefteyes.size() ==0)
		cout << "Aucun oeils detecte" << endl ;
	imshow("eyes",input) ;
	waitKey() ;
}

void showEyes(string filename){
	CascadeClassifier eye_classifier = getEyesCascadeClassifier();	
	Mat input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	
	vector<Rect> eyes = detectEye(eye_classifier, input); 
	if(eyes.size() != 0){
		cout << "Nombre de paires d'oeils : " << eyes.size() << endl ;
		for (int j=0;j<eyes.size();j++){
			rectangle(input,eyes[j],Scalar(0,255,0),1,8,0) ;
		}
	}
	else
		cout << "Aucun oeils detecte" << endl ;
	imshow("eyes",input) ;
	waitKey() ;
}