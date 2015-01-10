#include "faceDetection.h"
#include "tools.h"
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
    face_classifier.detectMultiScale(frame_gray, faces, 1.1, 4, 0);
    return faces;
}

/**
 * This function returns the most probable face among the given faces
 * The rule is the following : we return the face with the highest
 * number of features (eyes, nose, mouth) inside
 */
Rect selectBestFace(Mat frame, vector<Rect> faces){
    CascadeClassifier eyes_class = getEyesCascadeClassifier();
    CascadeClassifier nose_class = getNoseCascadeClassifier();
    CascadeClassifier mouth_class = getMouthCascadeClassifier();

    vector<int> nb_features(faces.size());

    for(int i=0;i<faces.size();i++){
        Mat reframedImg = frame(faces[i]);
        vector<Rect> features = detectEye(eyes_class, reframedImg);
        nb_features[i] = features.size();
        features = detectMouth(mouth_class, reframedImg);
        nb_features[i] += features.size();
        features = detectNose(nose_class, reframedImg);
        nb_features[i] += features.size();
    }
    int i_max=0;
    int val_max = 0;
    for(int i=0;i<faces.size();i++){
        if(nb_features[i] >= val_max){
            val_max = nb_features[i];
            i_max = i;
        }
    }
    return faces[i_max];
}

/**
 * This function returns the most probable face among the given faces
 * The rule is the following : we return the face with the highest
 * number of zone (eyes, nose, mouth) inside
 */
Rect selectBestFace2(Mat frame, vector<Rect> faces){
    CascadeClassifier eyes_class = getEyesCascadeClassifier();
    CascadeClassifier nose_class = getNoseCascadeClassifier();
    CascadeClassifier mouth_class = getMouthCascadeClassifier();

    vector<int> nb_features(faces.size());

    for(int i=0;i<faces.size();i++){
        Mat reframedImg = frame(faces[i]);
        vector<Rect> features = detectEye(eyes_class, reframedImg);
        nb_features[i] = min<int>(1,features.size());
        features = detectMouth(mouth_class, reframedImg);
        nb_features[i] += min<int>(1,features.size());
        features = detectNose(nose_class, reframedImg);
        nb_features[i] += min<int>(1,features.size());
    }
    int i_max=0;
    int val_max = 0;
    for(int i=0;i<faces.size();i++){
        if(nb_features[i] > val_max){
            val_max = nb_features[i];
            i_max = i;
        }
    }
    return faces[i_max];
}

/**
 * Show all features in an image : faces, eyes, mouthes, noses
 */
void showAllFeatures(Mat frame, vector<Rect> faces){
    CascadeClassifier eyes_class = getEyesCascadeClassifier();
    CascadeClassifier nose_class = getNoseCascadeClassifier();
    CascadeClassifier mouth_class = getMouthCascadeClassifier();
    Mat frame_gray = frame.clone();
    equalizeHist( frame_gray, frame_gray );

    cerr << faces.size() << " faces detected " << endl;
    for(int i=0;i<faces.size();i++){
        rectangle(frame_gray, faces[i], Scalar(255, 255, 255), 1, 8, 0);
    }
    imshow("AllFeatures", frame_gray);
    //waitKey();

    vector<Rect> eyes = detectEye(eyes_class, frame);
    for(int i=0;i<eyes.size();i++){
        rectangle(frame_gray, eyes[i], Scalar(255, 255, 255), 1, 8, 0);
    }
    cerr << eyes.size() << " eyes detected" << endl;
    imshow("AllFeatures", frame_gray);
    //waitKey();

    vector<Rect> noses = detectNose(nose_class, frame);
    for(int i=0;i<noses.size();i++){
        rectangle(frame_gray, noses[i], Scalar(255, 255, 255), 1, 8, 0);
    }
    cerr << noses.size() << " noses" << endl;
    imshow("AllFeatures", frame_gray);
    //waitKey();

    vector<Rect> mouthes = detectMouth(mouth_class, frame);
    for(int i=0;i<mouthes.size();i++){
        rectangle(frame_gray, mouthes[i], Scalar(255, 255, 255), 1, 8, 0);
    }
    cerr << mouthes.size() << " mouthes" << endl;
    imshow("AllFeatures", frame_gray);

    waitKey();
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

void showAllEyes(bool verbose){
	int tot = 0 ;
	int good = 0 ;
	for (directory_iterator it1("../data/yale_face_db/labeled"); it1 != directory_iterator() && tot <200 ; it1++){
		path p = it1->path() ;
		for(directory_iterator it2(p); it2 != directory_iterator() && tot < 200 ; it2 ++){
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
				tot ++ ;
				if(showEyes(p2.string(),verbose))
					good ++;
			}
		}
	}
	cout << "Résultat : " << good << " / " << tot << endl ;
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
    String eye_cascade_name = "../lib/haarcascade_eye.xml";
    CascadeClassifier eye_cascade;
    eye_cascade.load(eye_cascade_name);
    return eye_cascade;
}

vector<Rect> detectEye(CascadeClassifier eye_classifier, Mat frame )
{
    vector<Rect> eyes;
    Mat frame_gray = frame.clone();
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    eye_classifier.detectMultiScale(frame_gray, eyes, 1.1, 5, 0,Size(30,30));

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
		cout << "Aucun oeil detecte" << endl ;
	imshow("eyes",input) ;
	waitKey() ;
}

bool showEyes(string filename,bool verbose){
	CascadeClassifier eye_classifier = getEyesCascadeClassifier();	
	Mat input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	bool res = false ;
	vector<Rect> eyes = detectEye(eye_classifier, input); 
	if(eyes.size() != 0){
		cout << "Nombre d'oeils : " << eyes.size() << endl ;
		if(eyes.size() > 1)
			res = true ;
		if(verbose){
			for (int j=0;j<eyes.size();j++){
				rectangle(input,eyes[j],Scalar(0,255,0),1,8,0) ;
			}
			imshow("eyes",input) ;
			waitKey() ;
		}
	}
	else
		cout << "Aucun oeil detecte" << endl ;

	return res ;
}

CascadeClassifier getMouthCascadeClassifier()
{
    String mouth_cascade_name = "../lib/haarcascade_mcs_mouth.xml";
    CascadeClassifier mouth_cascade;
    mouth_cascade.load(mouth_cascade_name);
    return mouth_cascade;
}

vector<Rect> detectMouth(CascadeClassifier mouth_classifier, Mat frame )
{
    vector<Rect> mouth;
    Mat frame_gray = frame.clone();
    equalizeHist( frame_gray, frame_gray );

    mouth_classifier.detectMultiScale(frame_gray, mouth, 1.2, 6, 0,Size(30,30));

    return mouth;
}

CascadeClassifier getNoseCascadeClassifier()
{
    String nose_cascade_name = "../lib/haarcascade_mcs_nose.xml";
    CascadeClassifier nose_cascade;
    nose_cascade.load(nose_cascade_name);
    return nose_cascade;
}


vector<Rect> detectNose(CascadeClassifier nose_classifier, Mat frame )
{
    vector<Rect> nose;
    Mat frame_gray = frame.clone();
    equalizeHist( frame_gray, frame_gray );

    nose_classifier.detectMultiScale(frame_gray, nose, 1.1, 3, 0,Size(30,30));

    return nose;
}

void extractCroppedDescriptor(int n, int m , bool verbose){
	// n = 168 m = 192
	String dir_data = "../data/cropped" ;
	Mat descriptors;
	vector<int> classes ;

	int classe = 0 ;
	//Images to extract feature descriptors and build the vocabulary
	for (directory_iterator it1(dir_data); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		for(directory_iterator it2(p); it2 != directory_iterator(); it2 ++){
			cout << it2->path() << endl ;
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
				// Loading file
				Mat input = imread(p2.string(), CV_LOAD_IMAGE_GRAYSCALE);
				cout << input.type() << endl ;
				Mat inputbis ;
				resize(input,inputbis,Size(n,m)) ;
				if(verbose){
					imshow("input",input) ;
					imshow("resized",inputbis) ;
					waitKey() ;
				}
				cout << input.size() << endl ;
				int l = inputbis.rows *inputbis.cols ;
				cout << l << endl ;
				Mat rowDes = Mat::zeros(1,l,CV_8U) ;
				//cout << rowDes.size()  << endl ;
				//cout << input << endl << endl ;
				for (int y=0;y<inputbis.rows;y++){
					for(int x=0;x< inputbis.cols;x++){
						rowDes.at<uchar>(0,y*inputbis.cols +x) = inputbis.at<uchar>(y,x) ;
					}
				}
				//cout << rowDes << endl << endl ;
				descriptors.push_back(rowDes) ;
				classes.push_back(classe) ;
			}		
		}
		classe ++ ;
	}

	FileStorage f(("../allFeatures/cropped/face_descriptors.yml"), FileStorage::WRITE);
	f << "descriptors" << descriptors;
	f << "classes" << classes;
	f.release();

	ofstream fout("../allFeatures/cropped/face_descriptors.csv");

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<descriptors.rows; i++)
    {
		fout<<classes[i] <<",";
        for(int j=0; j<descriptors.cols; j++)
        {
            fout<<( (int) descriptors.at<uchar>(i,j) )<<",";
        }
        fout<<endl;
    }

    fout.close();
}


void showPCAfaces(int n, int m,int nb_components,int rankedFeatures[]){
	Mat descriptors ;
	FileStorage f(("../allFeatures/cropped/face_descriptors.yml"), FileStorage::READ);
	f["descriptors"] >> descriptors;
	f.release();

	Mat descriptorsFloat ;
	descriptors.convertTo(descriptorsFloat,CV_32FC1) ;

	PCA facePCA(descriptorsFloat, Mat(), CV_PCA_DATA_AS_ROW, nb_components);
	
	Mat vec0 = descriptorsFloat.row(0).clone() ;	
	Mat vec0bis ;
	Mat short_face ;
	facePCA.project( vec0,short_face) ;
	facePCA.backProject(short_face,vec0bis) ;
	
	cout << facePCA.eigenvectors.size() << endl ;
	Mat vec1 = facePCA.eigenvectors.row(0).clone() ;
	Mat vec1short = Mat::zeros(1,n*m,CV_32FC1) ;

	Mat vecTest = Mat::zeros(1,n*m,CV_32FC1) ;
	for(int x = 0 ; x < 20000 ; x++){
		vecTest.at<float>(0,rankedFeatures[x]) = vec0.at<float>(0,rankedFeatures[x]) ;
		vec1short.at<float>(0,rankedFeatures[x]) = vec1.at<float>(0,rankedFeatures[x]) ;
	}
	Mat vec1bis ;
	facePCA.project( vec1short,vec1bis) ;
	facePCA.backProject(vec1bis,vec1short) ;

	Mat vbis ;
	facePCA.project( vecTest,vbis) ;
	facePCA.backProject(vbis,vecTest) ;

	Mat face0 =  Mat(m,n,CV_32FC1); ;	
	Mat face0bis =  Mat(m,n,CV_32FC1); 
	Mat face1 =  Mat(m,n,CV_32FC1); 
	Mat faceTest = Mat(m,n,CV_32FC1); 
	/*for (int i=0 ; i< m*n; i++){
		int x = i/m ;
		int y = i-x*m ;
		face0bis.at<uchar>(y,x) = vec0bis.at<uchar>(0,i) ;
		face0.at<uchar>(y,x) = vec0.at<uchar>(0,i) ;
	}*/
	for (int y=0;y<face0.rows;y++){
		for(int x=0;x< face0.cols;x++){
			face0.at<float>(y,x) = vec0.at<float>(0,y*n + x) ;
			face0bis.at<float>(y,x) = vec0bis.at<float>(0,y*n + x) ;
			face1.at<float>(y,x) = vec1short.at<float>(0,y*n + x) ;
			faceTest.at<float>(y,x) = vecTest.at<float>(0,y*n + x) ;
		}
	}

	Mat face0converted , face0bisU , face1U , testU;
	face0.convertTo(face0converted,CV_8U) ;
	face0bis.convertTo(face0bisU,CV_8U) ;
	face1.convertTo(face1U,CV_8U) ;
	faceTest.convertTo(testU,CV_8U) ;
	
	imshow("face1",face1U) ;
	imshow("test",testU) ;
	imshow("face0",face0converted) ;
	imshow("face0 bis",face0bisU) ;
	waitKey() ;
}


void showSelectedFacesFeatures(int n, int m, int nb_feats, string nb, int rankedFeatures[]){
	Mat face = Mat(m,n,CV_32FC1);
	int strength = nb_feats ;
	cout << nb_feats << endl ;

	int j = 0 ;
	for (int i=0 ; i< n*m; i++){
		int feat = rankedFeatures[i] ;
		int y = feat/n ;
		int x = feat-y*n ;
		face.at<float>(y,x) = strength ;
		//cout << x << " " << y << endl ;
		j++ ;
		/*
		if(j>800){
			strength ++ ;
			j=0 ;		
		}*/
		if(strength >0)
			strength -- ;
	}
	
	ofstream fout("../allFeatures/cropped/face.csv");

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<face.rows; i++)
    {
        for(int j=0; j<face.cols; j++)
        {
            fout<<( (int) face.at<float>(i,j) )<<",";
        }
        fout<<endl;
    }

    fout.close();


	Mat A ;
	face.convertTo(A, CV_8U, 255.0/nb_feats, 0.);

	//cout << A << endl ;
	imshow("ranked features "+ nb,A) ;
	waitKey() ;
}