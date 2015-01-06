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

void buildSiftDictionary(int i,String database,bool verbose) ;
void buildPCAreducer(int nb_coponents,String database,vector<vector<int>> goodCols,bool verbose) ;
void featureExtraction(String database,vector<vector<int>> goodCols,bool verbose) ;
void initClassification(map<int,string> names ,int nb_coponents,String db , vector<vector<int>> goodCols) ;

void showPCA(Mat featuresUnclustered,vector<int> classesUnclustered, String title);
pair<Mat,Mat> computePCA(Mat features,int nb_coponents) ;
CvSVMParams chooseSVMParams(void) ;
vector<CvParamGrid> chooseSVMGrids(void) ;
int createSVMClassifier(String database) ;
map<int,CvSVM*> loadSVMClassifier(void) ;
void predict(String database) ;
void predictPCA(String database,vector<vector<int>> goodCols) ;
void predictPCA2(String database,vector<vector<int>> goodCols) ;
void classifyAndPredict(map<int,string> names ,int nb_coponents,String db , vector<vector<int>> goodCols) ;
void classifyAndPredict2(map<int,string> names ,int nb_coponents,String db , vector<vector<int>> goodCols) ;
