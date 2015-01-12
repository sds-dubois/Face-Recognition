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


	/*
	* Pour visualiser la 'feature selection' sur les pixels du visage
	* Method 1 : information gain - method 2 : chi square
	* Par ordre d'importance : du plus blanc au plus foncé
	*/
	//displaySelectedFeaturesOnFaces(2) ;

	String database = "yale_face_db" ;

	/*
	* BOW method
	*/
	//buildBowDictionary(50,false, database) ;
	//createBowClassifier(database) ;
	//computeBowTestDesciptors(database) ;
	//bowPredict(database) ;


	/*
	* void featureExtraction(String database,bool verbose, int detectionType)
	* choisir verbose == true pour voir les images et les zones detectees
	* detectionType:  methode 0 : simple - 1 : select best face - 2 : best face & intuite zones
	* use another integer as detectionType to save the features in a temporary file (and not erase previous files) 
	*/
	//featureExtraction(database,false,0) ;


	/*
    * void classifyAndPredict(int nb_coponents, String db,int nb_features, int detectionType, bool cross_valid) ;
	* nb_components = dimension de la PCA
	* nb_features = nombre de coordonéées selectionnees parmi les 128 entiers decrivant un SIFT
	* detectionType:  methode 0 : simple - 1 : select best face - 2 : best face & intuite zones
	* choisir cross_valid == true pour entrainer les cassifieurs avec cross-validation
	*/
    cerr << "Single descriptor" << endl;
    //classifyAndPredictSingleDescriptor(64, database , 128, false, false) ;
    cerr << "Descriptor by zone" << endl;
	classifyAndPredict(60, database, 60, 2,false) ;
    cerr << "Clustering" << endl;
    //clusteringClassifyAndPredict(90, database, 1, false);
	return 0 ;
}
