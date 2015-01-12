
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

// extracts SIFT descriptors for each zone and store them
// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
void featureExtraction(String database,bool verbose, int detectionType) ;

// use only one descriptor for all zones, and fill with zeros when a zone is not detected
// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
void classifyAndPredictSingleDescriptor(int nb_coponents,String db , int nb_features, int detectionType, bool cross_valid) ;

// the best to use
// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
void classifyAndPredict(int nb_coponents, String db, int nb_features, int detectionType, bool cross_valid) ;

// classify and predict using zone's SIFT descriptors and clustering
// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
void clusteringClassifyAndPredict(int dictionarySize ,String db,int detectionType, bool cross_valid) ;
