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

void showPCA(Mat featuresUnclustered,vector<int> classesUnclustered, String title){
	cout << "Nbr classes : " << featuresUnclustered.rows << endl ;
	int num_components = 10;
    PCA principalCA(featuresUnclustered, Mat(), CV_PCA_DATA_AS_ROW, num_components);
    Mat mean = principalCA.mean.clone();
    Mat eigenvectors = principalCA.eigenvectors.clone();

    for(int j=0;j<num_components/2;j++){
        Mat x_vector = eigenvectors.row(j);
        Mat y_vector = eigenvectors.row(j+1);

        float x_max,y_max,x_min, y_min;
        bool init=true;

        int width = 400;
        int height = 1200;
        Mat planePCA = Mat::zeros(width, height, CV_8UC3);
        for(int i=0;i<featuresUnclustered.rows;i++){
            Mat feature_i = featuresUnclustered.row(i);
            int x = feature_i.dot(x_vector);
            int y = feature_i.dot(y_vector);

            if(init){
                x_max = x;
                x_min = x;
                y_min = y;
                y_max = y;
                init=false;
            }

            if(x > x_max)
                x_max = x;
            if(x<x_min)
                x_min = x;
            if(y < y_min)
                y_min = y;
            if(y > y_max)
                y_max = y;
        }
        float deltay = y_max - y_min;
        y_max += deltay/5;
        y_min -= deltay/5;
        float deltax = x_max-x_min;
        x_max += deltax/5;
        x_min -= deltax/5;
        for(int i=0;i<featuresUnclustered.rows;i++){
            Mat feature_i = featuresUnclustered.row(i);
            int x = feature_i.dot(x_vector);
            int y = feature_i.dot(y_vector);
            Scalar color(255, 255, 255);
            if(classesUnclustered.at(i) == 1)
                color = Scalar(255,0, 0);
            else if(classesUnclustered.at(i) == 2)
                color = Scalar(0, 255, 0);
			else if(classesUnclustered.at(i) == 3)
                color = Scalar(0, 0, 255);
			Point p;
			if(deltax !=0)
				p.x=(int)height*(x-x_min)/(x_max-x_min);
			else
				p.x=height/2 ;
			if(deltay !=0)
				p.y=(int)width*(y-y_min)/(y_max-y_min) ;
			else
				p.y=width/2 ;
            circle(planePCA,p, 5, color);
			cout << "Point : " << p.x << " - " << p.y << " classe " << classesUnclustered.at(i) << endl ;

        }
        imshow("PCA " + title, planePCA);
        waitKey();
	}
}

pair<Mat,Mat> computePCA(Mat featuresUnclustered,int nb_coponents){
    PCA principalCA(featuresUnclustered, Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
    Mat eigenvectors = principalCA.eigenvectors.clone();
	Mat principalVectors = Mat(nb_coponents , eigenvectors.cols,CV_32FC1);
	Mat mean = principalCA.mean.clone() ;

	cout << "Mean size : " << mean.size() << endl ;
    for(int j=0;j<nb_coponents;j++){
		principalVectors.row(j) = eigenvectors.row(j) ;
	}
	return pair<Mat,Mat>(principalVectors.t(),mean) ;
}

vector<vector<int> > getGoodCols(int nb){
	vector<vector<int> > goodCols ;

	    // These will be the good cols for each part of the face
	int featuresLEye[] = {22,2,10,50,35,11,18,115,34,21,36,20,30,89,6,55,1,74,52,64,25,53,3,107,49,58,4,104,121,26,93,114,123,61,44,42,76,83,116,60,118,92,96,9,15,8,14,79,88,24,43,78,81,124,63,68,117,65,46,82};
	int featuresREye[] = {14,42,6,18,114,45,56,11,24,67,66,39,99,36,34,122,84,126,44,16,15,116,120,76,43,33,98,10,37,8,73,22,110,88,1,21,105,26,118,46,54,72,38,85,82,30,108,4,40,68,51,17,52,69,47,32,35,109,64,75} ;
	int featuresMouth[]= {88,94,24,56,90,3,36,32,120,126,106,86,102,58,103,78,62,4,14,37,0,49,100,77,43,82,33,99,74,81,12,15,47,11,98,50,127,1,67,95,64,63,48,45,60,125,22,118,57,84,73,10,44,68,35,83,87,42,89,39} ;
	int featuresNose[]= {120,88,24,14,3,126,94,47,32,2,64,90,26,81,78,102,56,4,22,75,74,84,44,41,52,30,15,36,40,65,43,12,96,23,127,106,73,68,82,49,37,105,95,17,77,103,8,76,16,53,80,11,35,86,42,39,72,45,48,33};

	vector<int> fLEye, fREye, fMouth, fNose ;
	for(int k=0;k<nb;k++){
		fLEye.push_back(featuresLEye[k]);
		fREye.push_back(featuresREye[k]);
		fMouth.push_back(featuresMouth[k]);
		fNose.push_back(featuresNose[k]);
	}

	goodCols.push_back(fLEye);
	goodCols.push_back(fREye);
	goodCols.push_back(fMouth);
	goodCols.push_back(fNose);

	return goodCols ;
}