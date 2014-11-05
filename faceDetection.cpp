 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
vector<Rect> detectAndDisplay( Mat frame );

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 CascadeClassifier face_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);

 /** @function main */
int main( int argc, const char** argv )
{
    CvCapture* capture;
    Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    //-- 2. Read the video stream
    capture = cvCaptureFromCAM( -1 );
    if( capture )
    {
        while( true ){
            frame = cvQueryFrame( capture );

            //-- 3. Apply the classifier to the frame
            if( !frame.empty() ){
                vector<Rect> faces = detectAndDisplay( frame );
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

/** @function detectAndDisplay */
vector<Rect> detectAndDisplay( Mat frame )
{
  vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  return faces;
}
