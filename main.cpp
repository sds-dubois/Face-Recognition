//#include "faceDetection.h"
//#include "bofSift.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main(void 
	//int argc, const char** argv 
	){
    //return detectFacesWebcam();
	//void buildDictionary(void) ;

	//Step 1 - Obtain the set of bags of features.

	//to store the input file names
	char * filename = new char[100];       
	//to store the current input image
	Mat input;    

	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;
	//To store the SIFT descriptor of current image
	Mat descriptor;
	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;
	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("HARRIS");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("BRIEF");


	//Images to extract feature descriptors and build the vocabulary
	for(int f=0;f<10;f++){        
		//create the file name of an image
		sprintf(filename,"../dictionary/%i.jpg",f);
		cout << filename << endl ;

		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale   
		cout << input.cols << " " << input.rows << endl;
		imshow("I",input);
		waitKey() ;
		//detect feature points
		detector->detect(input, keypoints);
		cout << keypoints.size() << endl ;

		//compute the descriptors for each keypoint
		extractor->compute(input, keypoints,descriptor); 
		//put the all feature descriptors in a single Mat object 
		featuresUnclustered.push_back(descriptor);        
		//print the percentage
		cout << f/10 << " percent done\n" << endl ;
		//printf("%i percent done\n",f/10);
	}    

	cout << "features Unclustered " << featuresUnclustered.size() << endl ;

	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize=2;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=3;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
	//bowTrainer.add(featuresUnclustered) ;
	//cluster the feature vectors
	Mat dictionary=bowTrainer.cluster(featuresUnclustered);    
	//store the vocabulary
	FileStorage fs("../dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	//void getBofDescriptor(void) ;

	return 0 ;

}
