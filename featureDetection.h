
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
// use another integer as detectionType to save the features in a temporary file (and not erase previous files) 
void featureExtraction(String database,bool verbose, int detectionType) ;

// use only one descriptor for all zones, and fill with zeros when a zone is not detected
// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
// nb_components : dimension for PCA - nb_features : number of features selected
void classifyAndPredictSingleDescriptor(int nb_components,String db , int nb_features, int detectionType, bool cross_valid) ;

// the best to use
// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
// nb_components : dimension for PCA - nb_features : number of features selected
void classifyAndPredict(int nb_coponents, String db, int nb_features, int detectionType, bool cross_valid) ;

// classify and predict using zone's SIFT descriptors and clustering
// method 0 : simple - method 1 : select best face - method 2 : best face & intuite zones
void clusteringClassifyAndPredict(int dictionarySize ,String db,int detectionType, bool cross_valid) ;
