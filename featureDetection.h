#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv ;

extern const bool selectFeatures ;
extern const bool pca ;
extern const int nb_celebrities  ;

vector<KeyPoint> getSiftOnMouth(Mat input, Rect searchZone, CascadeClassifier mouth_classifier,Ptr<FeatureDetector> detector,float alpha,bool verbose) ;
vector<KeyPoint> getSiftOnNose(Mat input, Rect searchZone, CascadeClassifier nose_classifier,Ptr<FeatureDetector> detector,float alpha,bool verbose) ;
vector<KeyPoint> getSiftOnEyes1(Mat input,CascadeClassifier eyes_classifier,Ptr<FeatureDetector> detector,bool verbose) ;
vector<KeyPoint> getSiftOnEyes2(Mat input,Rect searchZone,CascadeClassifier eyes_classifier,Ptr<FeatureDetector> detector, float& alpha,bool verbose) ;

// use full descriptors completed with zeros when a zone is not detected
// extracts features from images and compute classifiers & reducers
void buildPCAreducer(int nb_coponents,String database,vector<vector<int> > goodCols,bool verbose) ;

// extracts SIFT descriptors for each zone and store them
void featureExtraction(String database,vector<vector<int> > goodCols,bool verbose,bool completeDetection) ;

void initClassification(map<int,string> names ,int nb_coponents,String db , vector<vector<int> > goodCols) ;
void showPCA(Mat featuresUnclustered,vector<int> classesUnclustered, String title);
pair<Mat,Mat> computePCA(Mat features,int nb_coponents) ;

// use PCA reduction & full descriptors completed with zeros when a zone is not detected
void predictPCA(String database,vector<vector<int> > goodCols) ;

// use PCA reduction & a classifier by zone
void predictPCA2(String database,vector<vector<int> > goodCols,bool completeDetection) ;

// the best to use
void classifyAndPredict(map<int,string> names ,int nb_coponents,String db , vector<vector<int> > goodCols,bool completeDetection, bool cross_valid) ;
