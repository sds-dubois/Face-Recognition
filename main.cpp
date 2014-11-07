#include "faceDetection.h"
#include "bofSift.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main(void 
	//int argc, const char** argv 
	){
    //return detectFacesWebcam();
	
	buildSiftDictionary() ;
	cout << "build OK" << endl ;

	getSiftDescriptor() ;
	
	return 0 ;

}
