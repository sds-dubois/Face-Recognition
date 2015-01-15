#include "constants.h"
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
	if(goodCols.size() == 128)
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


vector<vector<int> > getGoodCols(int nb){
	vector<vector<int> > goodCols ;

	    // These will be the good cols for each part of the face
	int featuresLEye[] = {22,2,11,35,20,6,3,50,10,4,30,115,1,107,18,52,68,118,65,60,34,123,89,114,53,69,12,46,92,13,93,25,55,79,113,116,21,88,9,122,117,16,42,64,81,31,61,84,8,44,7,39,38,15,104,37,87,54,82,48,32,24,49,36,62,90,78,72,56,124,63,121,74,106,0,59,100,73,120,119,108,76,86,43,58,27,70,105,26,126,95,66,91,83,75,41,97,5,112,45,111,57,99,96,40,23,33,29,77,47,14,85,19,67,125,17,94,51,101,71,80,127,98,28,109,110,103,102};
	int featuresREye[] = {14,6,24,8,18,73,4,42,12,34,99,36,30,11,114,46,44,66,45,21,32,84,38,16,116,65,10,110,20,70,26,1,23,40,53,47,122,67,71,79,69,105,111,100,33,0,56,85,51,98,64,39,7,48,76,118,80,113,109,50,37,31,115,120,68,5,3,92,9,15,22,17,104,35,2,96,54,78,90,75,88,29,52,81,106,87,72,41,121,83,43,25,86,112,60,82,124,49,97,108,74,94,91,103,126,77,107,102,28,19,125,61,101,59,58,55,119,127,117,93,13,123,89,62,95,27,57,63};
	int featuresMouth[]= {88,94,56,24,120,126,36,86,99,90,89,100,106,57,58,37,103,125,14,3,102,62,112,95,78,32,33,49,82,98,16,68,53,127,101,63,64,84,47,80,67,10,60,43,2,52,118,42,77,69,114,75,79,0,93,113,87,50,48,44,17,27,45,73,22,115,35,55,72,76,105,13,74,15,21,41,54,18,65,26,4,40,38,12,81,97,92,117,119,31,8,111,23,20,83,11,59,107,28,7,116,104,46,1,61,91,30,39,124,121,34,25,96,9,5,51,108,109,71,66,123,85,6,29,110,19,70,122 };
	int featuresNose[]= {3,2,88,14,26,22,90,24,56,94,120,86,126,68,17,89,78,30,32,4,47,82,81,75,41,40,64,44,95,115,52,92,74,105,35,60,16,18,125,43,11,76,96,49,69,65,23,37,99,36,33,97,53,84,118,83,13,121,57,21,93,48,101,38,112,39,10,73,80,29,100,31,98,20,67,45,103,19,102,7,6,111,109,107,79,0,106,116,46,12,42,15,28,117,77,59,61,27,50,110,8,85,119,123,55,127,87,51,54,71,70,72,34,124,108,104,113,9,1,66,25,63,62,91,58,122,114,5};

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
