#include "faceDetection.h"
#include "featureDetection.h"
#include "bowMethod.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;


int main(int argc, char ** argv){
    int j ;
	bool b ; //True if you want to see images and Sift detection while building the dictionary
    if(argc > 1){
        stringstream ss(argv[1]);
        ss >> j;
		stringstream ss2(argv[1]);
		ss2 >> b;
    }else{
/*
		cin >> j;
		cin >> b;
*/    }

	/*
	showSelectedFacesFeatures(168,192,1000,"1000",rankedFeats2) ;
	showSelectedFacesFeatures(168,192,5000,"5000",rankedFeats2) ;
	showSelectedFacesFeatures(168,192,10000,"10000",rankedFeats2) ;
	showSelectedFacesFeatures(168,192,15000,"15000",rankedFeats2) ;
	showSelectedFacesFeatures(168,192,25000,"25000",rankedFeats2) ;
	showSelectedFacesFeatures(168,192,30000,"30000",rankedFeats2) ;
	*/

	String database = "yale_face_db" ;
	vector<vector<int> > goodCols = getGoodCols(60) ;

	/*
	* BOW method
	*/
	//buildBowDictionary(50,false, database) ;
	createBowClassifier(database) ;
	//computeBowTestDesciptors(database) ;
	//bowPredict(database) ;


	/*
	* void featureExtraction(String database,vector<vector<int> > goodCols,bool verbose, int detectionType)
	* choisir verbose == true pour voir les images et les zones detectees
	* detectionType:  methode 0 : simple - 1 : select best face - 2 : best face & intuite zones
	*/
	featureExtraction(database,goodCols,false,0) ;
	featureExtraction(database,goodCols,false,1) ;
	featureExtraction(database,goodCols,false,2) ;


	//classifyAndPredictSingleDescriptor(128, database , goodCols, false, false) ;
	/*
    * void classifyAndPredict(int nb_coponents, String db, vector<vector<int> > goodCols, int detectionType, bool cross_valid) ;
	* nb_components = dimension de la PCA
	* goodCols = colonnes selectionnees parmi les 128 entiers decrivant un SIFT
	* detectionType:  methode 0 : simple - 1 : select best face - 2 : best face & intuite zones
	* choisir cross_valid == true pour entrainer les cassifieurs avec cross-validation
	*/
	//classifyAndPredict(128, database, goodCols, 0,false) ;
	return 0 ;
}
