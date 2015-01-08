#include "featureDetection.h"
#include "getSiftKeypoints.h"
#include "faceDetection.h"
#include "bowMethod.h"


#include <stdio.h>
#include <boost/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv ;

#define selectFeatures  false 
#define pca false
#define nb_celebrities 3

bool waytosort(KeyPoint p1, KeyPoint p2){ return p1.response > p2.response ;}

void addNumberToFile(const char * filename, float n){
    ofstream fout(filename, ios_base::out|ios_base::app);
    if(!fout)
    {
        cout << "File " << filename << " could not be opened" << endl;
    }
    fout << n << endl;
    fout.close();
}

void writeMatToFile(Mat& m, vector<int> classesUnclustered,String filename){
    ofstream fout(filename.c_str());

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
		fout<<classesUnclustered[i] <<",";
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<",";
        }
        fout<<endl;
    }

    fout.close();
}


Mat selectCols(vector<int> goodCols,Mat m){
	if(!selectFeatures)
		return m ;
	else{
		int n = goodCols.size() ;
		Mat res = Mat(m.rows,n,CV_32FC1) ;

		for (int k=0; k<n;k++){
			//Mat temp = res.col(k) ;
			//m.col(goodCols[k]).copyTo(temp) ;
			for(int i=0; i < m.rows;i++){
				res.at<float>(i,k) = m.at<float>(i,goodCols[k]) ;
			}
		}

		return res ;
	}
}


CvSVMParams chooseSVMParams(void){
	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	//params.degree = 3 ;
	//params.gamma =  5;
	//params.coef0 = 1 ;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	return params ;
}

vector<CvParamGrid> chooseSVMGrids(void){
	/*
	Ordre :
	CvParamGrid Cgrid=CvSVM::get_default_grid(CvSVM::C)
	CvParamGrid gammaGrid=CvSVM::get_default_grid(CvSVM::GAMMA)
	CvParamGrid pGrid=CvSVM::get_default_grid(CvSVM::P)
	CvParamGrid nuGrid=CvSVM::get_default_grid(CvSVM::NU)
	CvParamGrid coeffGrid=CvSVM::get_default_grid(CvSVM::COEF)
	CvParamGrid degreeGrid=CvSVM::get_default_grid(CvSVM::DEGREE)
	*/
	vector<CvParamGrid> grids ;
	grids.push_back(CvSVM::get_default_grid(CvSVM::C)) ;
	grids.push_back(CvSVM::get_default_grid(CvSVM::GAMMA)) ;
	grids.push_back(CvSVM::get_default_grid(CvSVM::P)) ;
	grids.push_back(CvSVM::get_default_grid(CvSVM::NU)) ;
	grids.push_back(CvSVM::get_default_grid(CvSVM::COEF)) ;
	grids.push_back(CvSVM::get_default_grid(CvSVM::DEGREE)) ;

	return grids ;
}