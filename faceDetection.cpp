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
				// Loading file
				showFaces(p2.string()) ;
			}
		}
	}
}
