#include "featureDetection.h"

#include <stdio.h>
#include <boost/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv ;

bool waytosort(KeyPoint p1, KeyPoint p2) ;
void addNumberToFile(const char * filename, float n) ;
void writeMatToFile(Mat& m, vector<int> classesUnclustered,String filename) ;
Mat selectCols(vector<int> goodCols,Mat m) ;

// selected features for each zone (return the nb best)
vector<vector<int> > getGoodCols(int nb) ;

CvSVMParams chooseSVMParams(void) ;
vector<CvParamGrid> chooseSVMGrids(void) ;

void showPCA(Mat featuresUnclustered,vector<int> classesUnclustered, String title);
pair<Mat,Mat> computePCA(Mat featuresUnclustered,int nb_coponents) ;
