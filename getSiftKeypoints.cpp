#include "getSiftKeypoints.h"
#include "featureDetection.h"
#include "faceDetection.h"
#include "tools.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include <math.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <set>


using namespace std;
using namespace cv;
using namespace boost::filesystem ;


vector<KeyPoint> getSiftOnMouth(Mat input, Rect searchZone, CascadeClassifier mouth_classifier,Ptr<FeatureDetector> detector,float alpha,bool verbose){
	Mat reframedImg = input(searchZone);
	Mat img_with_sift ;
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
	vector<Rect> mouths = detectMouth(mouth_classifier, reframedImg);
	if(mouths.size() ==0){
		cout << "Erreur : aucune bouche trouvee" << endl ;
	}
	else{
		if(mouths.size() >1)
			cout << "Attention : plus d'une bouche trouvee" << endl ;
		Rect mouthZone = mouths[0] ;
        addNumberToFile("../stats/mouth_x.csv", (float)mouthZone.x/searchZone.width);
        addNumberToFile("../stats/mouth_width.csv", (float)mouthZone.width/searchZone.width);
        addNumberToFile("../stats/mouth_y.csv", (float)mouthZone.y/searchZone.height);
        addNumberToFile("../stats/mouth_height.csv", (float)mouthZone.height/searchZone.height);
		mouthZone.x += searchZone.x ;
		mouthZone.y += searchZone.y ;
		//Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U);
		//mask(mouthZone) = 1;
		if(verbose)
			rectangle(img_with_sift,mouthZone,Scalar(0,255,0),1,8,0) ;
		//compute the descriptors for each keypoint and put it in a single Mat object
		Point_<float> c1 = Point_<float>(mouthZone.x+0.5*mouthZone.size().width,mouthZone.y+0.5*mouthZone.size().height);
		if(verbose)
			cout << "Alpha = " << alpha << endl ;
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(mouthZone.size().width+mouthZone.size().height),alpha));
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			rectangle(img_with_sift,mouthZone,Scalar(0,255,0),1,8,0) ;
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	return keypoints_best ;
}

vector<KeyPoint> getSiftOnNose(Mat input, Rect searchZone, CascadeClassifier nose_classifier,Ptr<FeatureDetector> detector,float alpha,bool verbose){
	Mat img_with_sift ;
	Mat reframedImg = input(searchZone);
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
	vector<Rect> noses = detectMouth(nose_classifier, reframedImg);
	if(noses.size() ==0){
		cout << "Erreur : aucun nez trouve" << endl ;
	}
	else{
		if(noses.size() >1)
			cout << "Attention : plus d'un nez trouve" << endl ;
		Rect nose = noses[0] ;
        addNumberToFile("../stats/nose_x.csv", (float)nose.x/searchZone.width);
        addNumberToFile("../stats/nose_width.csv", (float)nose.width/searchZone.width);
        addNumberToFile("../stats/nose_y.csv", (float)nose.y/searchZone.height);
        addNumberToFile("../stats/nose_height.csv", (float)nose.height/searchZone.height);
		nose.x += searchZone.x ;
		nose.y += searchZone.y ;
		//Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U);
		//mask(nose) = 1;
		if(verbose)
			rectangle(img_with_sift,nose,Scalar(0,255,0),1,8,0) ;
		//compute the descriptors for each keypoint and put it in a single Mat object
		Point_<float> c1 = Point_<float>(nose.x+0.5*nose.size().width,nose.y+0.5*nose.size().height);
		if(verbose)
			cout << "Alpha = " << alpha << endl ;
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(nose.size().width+nose.size().height),alpha));
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			rectangle(img_with_sift,nose,Scalar(0,255,0),1,8,0) ;
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	return keypoints_best ;
}

vector<KeyPoint> getSiftOnEyes1(Mat input,CascadeClassifier eyes_classifier,Ptr<FeatureDetector> detector,bool verbose){
	Mat img_with_sift ;
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
    vector<Rect> eyes = detectEye(eyes_classifier, input);
	if(eyes.size() == 2){
		Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U);
		for (int k=0;k<2;k++){
			mask(eyes[k]) = 1;
			if(verbose)
				rectangle(img_with_sift,eyes[k],Scalar(0,255,0),1,8,0) ;
		}
		//compute the descriptors for each keypoint and put it in a single Mat object
		vector<KeyPoint> keypoints ;
		detector->detect(input, keypoints,mask);
		int count = 0 ;
		int s = keypoints.size() ;
		sort(keypoints.begin(),keypoints.end(),waytosort);
		for(int t = 0; t <s && t < 10; t++){
			keypoints_best.push_back(keypoints[t]) ;
			count ++ ;
			if(verbose){
				cout << keypoints[t].response << " - " << t  << endl ;
				cout << keypoints[t].angle << " - " << keypoints[t].size << endl ;
				drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
				imshow("Best Keypoints",img_with_sift) ;
				waitKey() ;
			}
		}
		if(verbose)
			cout << "nbr keypoints : " << count << " - " << keypoints_best.size() << " - " << s << endl ;
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Best Keypoints",img_with_sift) ;
			drawKeypoints(input,keypoints,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			for (int k=0;k<2;k++){
				rectangle(img_with_sift,eyes[k],Scalar(0,255,0),1,8,0) ;
			}
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	else
		cout << "Error in SIFT detection " << endl ;

	return keypoints_best ;
}

vector<KeyPoint> getSiftOnEyes2(Mat input,Rect searchZone,CascadeClassifier eyes_classifier,Ptr<FeatureDetector> detector, float& alpha,bool verbose){
	Mat reframedImg = input(searchZone);
	Mat img_with_sift ;
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
    vector<Rect> eyes = detectEye(eyes_classifier, reframedImg);
	if(eyes.size() >= 2){
		if(eyes.size() >2)
			cout << "Attention : plus de deux yeux trouvees" << endl;
		Rect eyeZone1 = eyes[0] ;
        addNumberToFile("../stats/eye_x.csv", (float)eyeZone1.x / searchZone.width);
        addNumberToFile("../stats/eye_width.csv", (float)eyeZone1.width / searchZone.width);
        addNumberToFile("../stats/eye_y.csv", (float)eyeZone1.y / searchZone.height);
        addNumberToFile("../stats/eye_height.csv", (float)eyeZone1.height / searchZone.height);
		eyeZone1.x += searchZone.x ;
		eyeZone1.y += searchZone.y ;
        Rect eyeZone2 = eyes[1] ;
        addNumberToFile("../stats/eye_x.csv", (float)eyeZone2.x / searchZone.width);
        addNumberToFile("../stats/eye_width.csv", (float)eyeZone2.width / searchZone.width);
        addNumberToFile("../stats/eye_y.csv", (float)eyeZone2.y / searchZone.height);
        addNumberToFile("../stats/eye_height.csv", (float)eyeZone2.height / searchZone.height);
		eyeZone2.x += searchZone.x ;
		eyeZone2.y += searchZone.y ;
		if(verbose){
			rectangle(img_with_sift,eyeZone1,Scalar(0,255,0),1,8,0) ;
			rectangle(img_with_sift,eyeZone2,Scalar(0,255,0),1,8,0) ;
		}
		Point_<float> c1 = Point_<float>(eyeZone1.x+0.5*eyeZone1.size().width,eyeZone1.y+0.5*eyeZone1.size().height);
		Point_<float> c2 = Point_<float>(eyeZone2.x+0.5*eyeZone2.size().width,eyeZone2.y+0.5*eyeZone2.size().height);
		alpha = (atan((c1.y-c2.y)/(c1.x-c2.x)))*180/3 ;
		if(verbose)
			cout << "Alpha = " << alpha << endl ;
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(eyeZone1.size().width+eyeZone1.size().height),alpha));
		keypoints_best.push_back(KeyPoint(c2.x,c2.y,0.5*(eyeZone2.size().width+eyeZone2.size().height),alpha));
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	else if(eyes.size() > 0){
		Rect eyeZone1 = eyes[0] ;
        addNumberToFile("../stats/eye_x.csv", (float)eyeZone1.x / searchZone.width);
        addNumberToFile("../stats/eye_width.csv", (float)eyeZone1.width / searchZone.width);
        addNumberToFile("../stats/eye_y.csv", (float)eyeZone1.y / searchZone.height);
        addNumberToFile("../stats/eye_height.csv", (float)eyeZone1.height / searchZone.height);
        eyeZone1.x += searchZone.x ;
		eyeZone1.y += searchZone.y ;
		if(verbose){
			rectangle(img_with_sift,eyeZone1,Scalar(0,255,0),1,8,0) ;
		}
		Point_<float> c1 = Point_<float>(eyeZone1.x+0.5*eyeZone1.size().width,eyeZone1.y+0.5*eyeZone1.size().height);
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(eyeZone1.size().width+eyeZone1.size().height),alpha));
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
    }
    else
		cout << "Error in eyes detection" << endl ;

	return keypoints_best ;
}
