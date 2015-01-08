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

void buildSiftDictionary(int i,String db,bool verbose) ;
int init_bowDE(BOWImgDescriptorExtractor& mouth_bowDE,BOWImgDescriptorExtractor& eyes_bowDE,BOWImgDescriptorExtractor& nose_bowDE,String dir) ;
int createSVMClassifier(String db) ;
void predictBOW(String db) ;