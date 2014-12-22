#include "faceDetection.h"
#include "featureDetection.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;


int main(int argc, char ** argv){
    int j ;
	bool b ; //True if you want to see images and Sift detection while building the dictionary
    if(argc > 1){
        stringstream ss(argv[1]);
        ss >> j;
		stringstream ss2(argv[1]);
		ss2 >> b;
    }else{
/*
		cin >> j;
		cin >> b;
*/    }

	int featuresLEye[] = {22,2,10,50,35,11,18,115,34,21,36,20,30,89,6,55,1,74,52,64,25,53,3,107,49,58,4,104,121,26,93,114,123,61,44,42,76,83,116,60,118,92,96,9,15,8,14,79,88,24,43,78,81,124,63,68,117,65,46,82};
	int featuresREye[] = {14,42,6,18,114,45,56,11,24,67,66,39,99,36,34,122,84,126,44,16,15,116,120,76,43,33,98,10,37,8,73,22,110,88,1,21,105,26,118,46,54,72,38,85,82,30,108,4,40,68,51,17,52,69,47,32,35,109,64,75} ;
	int featuresMouth[]= {88,94,24,56,90,3,36,32,120,126,106,86,102,58,103,78,62,4,14,37,0,49,100,77,43,82,33,99,74,81,12,15,47,11,98,50,127,1,67,95,64,63,48,45,60,125,22,118,57,84,73,10,44,68,35,83,87,42,89,39} ;
	int featuresNose[]= {120,88,24,14,3,126,94,47,32,2,64,90,26,81,78,102,56,4,22,75,74,84,44,41,52,30,15,36,40,65,43,12,96,23,127,106,73,68,82,49,37,105,95,17,77,103,8,76,16,53,80,11,35,86,42,39,72,45,48,33};

	vector<int> fLEye, fREye, fMouth, fNose ;
	for(int k=0;k<60;k++){
		fLEye.push_back(featuresLEye[k]);
		fREye.push_back(featuresREye[k]);
		fMouth.push_back(featuresMouth[k]);
		fNose.push_back(featuresNose[k]);
	}
	vector<vector<int>> goodCols ;
	goodCols.push_back(fLEye);
	goodCols.push_back(fREye);
	goodCols.push_back(fMouth);
	goodCols.push_back(fNose);
	
	String database = "yale_face_db" ;
	buildPCAreducer(30,database,goodCols,false) ;

	//createSVMClassifier() ;

	predictPCA(database,goodCols) ;

	return 0 ;
}
