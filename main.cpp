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

using namespace std;
using namespace cv;


int main(void){
    //return detectFacesWebcam();

	/*
	int j ;
	cin >> j ;
	buildSiftDictionary(j) ;
	cout << "build OK" << endl ;
	*/

	//createSVMClassifier() ;

	predict() ;

	return 0 ;
}
