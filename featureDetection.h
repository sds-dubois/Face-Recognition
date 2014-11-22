#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv ;

void buildSiftDictionary(int i,bool verbose) ;
void buildEyeDictionary(int i,bool verbose) ;
void compareDescriptors(string filename) ;
CvSVMParams chooseSVMParams(void) ;
vector<CvParamGrid> chooseSVMGrids(void) ;
int createSVMClassifier(void) ;
map<int,CvSVM*> loadSVMClassifier(void) ;
void predict(void) ;
