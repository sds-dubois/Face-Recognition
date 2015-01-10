
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


// use full descriptors completed with zeros when a zone is not detected
// extracts features from images and compute classifiers & reducers
void buildPCAreducer(int nb_coponents,String database,vector<vector<int> > goodCols,bool verbose) ;

// extracts SIFT descriptors for each zone and store them
void featureExtraction(String database,vector<vector<int> > goodCols,bool verbose, int detectionType) ;

void initClassification(map<int,string> names ,int nb_coponents,String db , vector<vector<int> > goodCols) ;

// use PCA reduction & full descriptors completed with zeros when a zone is not detected
void predictPCA(String database,vector<vector<int> > goodCols) ;

// use PCA reduction & a classifier by zone
void predictPCA2(String database,vector<vector<int> > goodCols, int detectionType) ;

// use only one descriptor for all zones, and fill with zeros when a zone is not detected
void classifyAndPredictSingleDescriptor(map<int,string> names ,int nb_coponents,String db , vector<vector<int> > goodCols,bool completeDetection, bool cross_valid) ;

// the best to use
void classifyAndPredict(int nb_coponents, String db, vector<vector<int> > goodCols, int detectionType, bool cross_valid) ;
