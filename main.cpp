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
	vector<Mat> reducers = buildPCAreducer(20,false) ;

	//createSVMClassifier() ;

	predictPCA(reducers) ;

	return 0 ;
}
