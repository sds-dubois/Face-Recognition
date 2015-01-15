#include "featureDetection.h"
#include "tools.h"

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


//compute dictionaries from sift extracted on training data
void buildBowDictionary(int dictionarySize,bool verbose,string db) ;

//compute SVM classifiers from BOW histograms, from sift extracted on training data, & store BOW descriptors of training data
int createBowClassifier(string db) ;

//compute & store BOW descriptors of test data
void computeBowTestDesciptors(string db) ;

//predict using training's & test's stored BOW descriptors, and SVM classifiers
void bowPredict(string db) ;