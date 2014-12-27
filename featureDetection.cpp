#include "featureDetection.h"
#include "faceDetection.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include <math.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <set>


using namespace std;
using namespace cv;
using namespace boost::filesystem ;

bool waytosort(KeyPoint p1, KeyPoint p2){ return p1.response > p2.response ;}
const bool pca=false;
const bool selectFeatures = true ;
const int nb_celebrities = 3 ;

void addNumberToFile(const char * filename, float n)
{
    ofstream fout(filename, ios_base::out|ios_base::app);
    if(!fout)
    {
        cout << "File " << filename << " could not be opened" << endl;
    }
    fout << n << endl;
    fout.close();
}
void writeMatToFile(Mat& m, vector<int> classesUnclustered,String filename)
{
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

inline Mat selectCols(vector<int> goodCols,Mat m){
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


vector<KeyPoint> getSiftOnMouth(Mat input, Rect searchZone, CascadeClassifier mouth_classifier,Ptr<FeatureDetector> detector,float alpha,bool verbose){
	Mat reframedImg = input(searchZone);
	Mat img_with_sift ;
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
	vector<Rect> mouths = detectMouth(mouth_classifier, reframedImg); 
	if(mouths.size() ==0){
		cout << "Erreur : aucune bouche trouvee" << endl ;
	}
	else{
		if(mouths.size() >1)
			cout << "Attention : plus d'une bouche trouvee" << endl ;
		Rect mouthZone = mouths[0] ;
        addNumberToFile("../stats/mouth_x.csv", (float)mouthZone.x/searchZone.width);
        addNumberToFile("../stats/mouth_width.csv", (float)mouthZone.width/searchZone.width);
        addNumberToFile("../stats/mouth_y.csv", (float)mouthZone.y/searchZone.height);
        addNumberToFile("../stats/mouth_height.csv", (float)mouthZone.height/searchZone.height);
		mouthZone.x += searchZone.x ;
		mouthZone.y += searchZone.y ;
		//Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
		//mask(mouthZone) = 1;
		if(verbose)
			rectangle(img_with_sift,mouthZone,Scalar(0,255,0),1,8,0) ;
		//compute the descriptors for each keypoint and put it in a single Mat object
		Point_<float> c1 = Point_<float>(mouthZone.x+0.5*mouthZone.size().width,mouthZone.y+0.5*mouthZone.size().height);
		if(verbose)
			cout << "Alpha = " << alpha << endl ;
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(mouthZone.size().width+mouthZone.size().height),alpha));
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			rectangle(img_with_sift,mouthZone,Scalar(0,255,0),1,8,0) ;
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	return keypoints_best ;
}

vector<KeyPoint> getSiftOnNose(Mat input, Rect searchZone, CascadeClassifier nose_classifier,Ptr<FeatureDetector> detector,float alpha,bool verbose){
	Mat img_with_sift ;
	Mat reframedImg = input(searchZone);
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
	vector<Rect> noses = detectMouth(nose_classifier, reframedImg); 
	if(noses.size() ==0){
		cout << "Erreur : aucun nez trouve" << endl ;
	}
	else{
		if(noses.size() >1)
			cout << "Attention : plus d'un nez trouve" << endl ;
		Rect nose = noses[0] ;
        addNumberToFile("../stats/nose_x.csv", (float)nose.x/searchZone.width);
        addNumberToFile("../stats/nose_width.csv", (float)nose.width/searchZone.width);
        addNumberToFile("../stats/nose_y.csv", (float)nose.y/searchZone.height);
        addNumberToFile("../stats/nose_height.csv", (float)nose.height/searchZone.height);
		nose.x += searchZone.x ;
		nose.y += searchZone.y ;
		//Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
		//mask(nose) = 1;
		if(verbose)
			rectangle(img_with_sift,nose,Scalar(0,255,0),1,8,0) ;
		//compute the descriptors for each keypoint and put it in a single Mat object
		Point_<float> c1 = Point_<float>(nose.x+0.5*nose.size().width,nose.y+0.5*nose.size().height);
		if(verbose)
			cout << "Alpha = " << alpha << endl ;
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(nose.size().width+nose.size().height),alpha));
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			rectangle(img_with_sift,nose,Scalar(0,255,0),1,8,0) ;
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	return keypoints_best ;
}

vector<KeyPoint> getSiftOnEyes1(Mat input,CascadeClassifier eyes_classifier,Ptr<FeatureDetector> detector,bool verbose){
	Mat img_with_sift ;
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
    vector<Rect> eyes = detectEye(eyes_classifier, input); 
	if(eyes.size() == 2){
		Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
		for (int k=0;k<2;k++){
			mask(eyes[k]) = 1;
			if(verbose)
				rectangle(img_with_sift,eyes[k],Scalar(0,255,0),1,8,0) ;
		}
		//compute the descriptors for each keypoint and put it in a single Mat object
		vector<KeyPoint> keypoints ;
		detector->detect(input, keypoints,mask);
		int count = 0 ;
		int s = keypoints.size() ;
		sort(keypoints.begin(),keypoints.end(),waytosort);
		for(int t = 0; t <s && t < 10; t++){
			keypoints_best.push_back(keypoints[t]) ;
			count ++ ;
			if(verbose){
				cout << keypoints[t].response << " - " << t  << endl ;
				cout << keypoints[t].angle << " - " << keypoints[t].size << endl ;
				drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
				imshow("Best Keypoints",img_with_sift) ;
				waitKey() ;
			}
		} 
		if(verbose)
			cout << "nbr keypoints : " << count << " - " << keypoints_best.size() << " - " << s << endl ;
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Best Keypoints",img_with_sift) ;
			drawKeypoints(input,keypoints,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			for (int k=0;k<2;k++){
				rectangle(img_with_sift,eyes[k],Scalar(0,255,0),1,8,0) ;
			}
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	else
		cout << "Error in SIFT detection " << endl ;

	return keypoints_best ;
}

vector<KeyPoint> getSiftOnEyes2(Mat input,Rect searchZone,CascadeClassifier eyes_classifier,Ptr<FeatureDetector> detector, float& alpha,bool verbose){
	Mat reframedImg = input(searchZone);
	Mat img_with_sift ;
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
    vector<Rect> eyes = detectEye(eyes_classifier, reframedImg); 
	if(eyes.size() >= 2){
		if(eyes.size() >2)
			cout << "Attention : plus de deux yeux trouvees" << endl;
		Rect eyeZone1 = eyes[0] ;
        addNumberToFile("../stats/eye_x.csv", (float)eyeZone1.x / searchZone.width);
        addNumberToFile("../stats/eye_width.csv", (float)eyeZone1.width / searchZone.width);
        addNumberToFile("../stats/eye_y.csv", (float)eyeZone1.y / searchZone.height);
        addNumberToFile("../stats/eye_height.csv", (float)eyeZone1.height / searchZone.height);
		eyeZone1.x += searchZone.x ;
		eyeZone1.y += searchZone.y ;
        Rect eyeZone2 = eyes[1] ;
        addNumberToFile("../stats/eye_x.csv", (float)eyeZone2.x / searchZone.width);
        addNumberToFile("../stats/eye_width.csv", (float)eyeZone2.width / searchZone.width);
        addNumberToFile("../stats/eye_y.csv", (float)eyeZone2.y / searchZone.height);
        addNumberToFile("../stats/eye_height.csv", (float)eyeZone2.height / searchZone.height);
		eyeZone2.x += searchZone.x ;
		eyeZone2.y += searchZone.y ;
		if(verbose){
			rectangle(img_with_sift,eyeZone1,Scalar(0,255,0),1,8,0) ;
			rectangle(img_with_sift,eyeZone2,Scalar(0,255,0),1,8,0) ;
		}
		Point_<float> c1 = Point_<float>(eyeZone1.x+0.5*eyeZone1.size().width,eyeZone1.y+0.5*eyeZone1.size().height);
		Point_<float> c2 = Point_<float>(eyeZone2.x+0.5*eyeZone2.size().width,eyeZone2.y+0.5*eyeZone2.size().height);
		alpha = (atan((c1.y-c2.y)/(c1.x-c2.x)))*180/3 ;
		if(verbose)
			cout << "Alpha = " << alpha << endl ;
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(eyeZone1.size().width+eyeZone1.size().height),alpha));
		keypoints_best.push_back(KeyPoint(c2.x,c2.y,0.5*(eyeZone2.size().width+eyeZone2.size().height),alpha));		
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	else
		cout << "Error in eyes detection" << endl ;

	return keypoints_best ;
}

void buildSiftDictionary(int i,String db,bool verbose){

	String dir_dico = "../reducers/" + db ;
	String dir_labeled_data = "../data/" + db + "/labeled" ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	initModule_nonfree() ;

	//To store the SIFT descriptor of current image
	Mat descriptorEyes;
	Mat descriptorMouth;
	Mat descriptorNose;
	//To store all the descriptors that are extracted from all the images.
	Mat eyesFeaturesUnclustered;
	Mat mouthFeaturesUnclustered;
	Mat noseFeaturesUnclustered;
	vector<int> classesUnclustered_eyes;
	vector<int> classesUnclustered_mouth;
	vector<int> classesUnclustered_nose;
	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

	//Images to extract feature descriptors and build the vocabulary
	int classPolitician=1;
	for (directory_iterator it1(dir_labeled_data); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		cout << "Folder " << p.string() << endl ;
		for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
			cout << it2->path() << endl ;
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
                // Loading file
                Mat input = imread(p2.string(), CV_LOAD_IMAGE_GRAYSCALE);
				vector<Rect> faces = detectFaces(face_classifier, input); 
				Rect searchZone ;
				vector<KeyPoint> keypoints_mouth ;
				vector<KeyPoint> keypoints_nose ;
				float alpha =0 ;
				vector<KeyPoint> keypoints_eyes ;
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
					Rect searchEyeZone = faces[0] ;
					searchEyeZone.height /= 2 ;
					keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,verbose);
					Rect searchMouthZone = faces[0] ;
					searchMouthZone.height /= 2 ;
					searchMouthZone.y += searchMouthZone.height ;
					keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,verbose);
					keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,verbose) ; 
				}
				else{
					cout << "Attention : pas de visage detecte" << endl ;
				}
				if(keypoints_eyes.size() != 0){
                    extractor->compute(input, keypoints_eyes,descriptorEyes);
					eyesFeaturesUnclustered.push_back(descriptorEyes);
					for(int i=0;i<2;i++)
						classesUnclustered_eyes.push_back(classPolitician);
				}
				if(keypoints_mouth.size() != 0){
					extractor->compute(input, keypoints_mouth,descriptorMouth);
					mouthFeaturesUnclustered.push_back(descriptorMouth);
					classesUnclustered_mouth.push_back(classPolitician);
				}
				if(keypoints_nose.size() != 0){
					extractor->compute(input, keypoints_nose,descriptorNose);
					noseFeaturesUnclustered.push_back(descriptorNose);
					classesUnclustered_nose.push_back(classPolitician);
				}
				if(keypoints_eyes.size() != 0 && keypoints_mouth.size() != 0 && keypoints_nose.size() != 0){
					cout << ">>>>>>>>> Success for : " << it2->path() << endl << endl;
				}
			}
		}
		classPolitician++;
	}

	cout << "features Unclustered " << mouthFeaturesUnclustered.size() << endl ;
	cout << "classes : -mouth " << classesUnclustered_mouth.size() << " -noses : " << classesUnclustered_mouth.size()<< endl;

	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize=i;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=3;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;

	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainerEyes(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	Mat dictionaryEyes=bowTrainerEyes.cluster(eyesFeaturesUnclustered) ;
	//store the vocabulary
	FileStorage fs1(dir_dico + "/eye_dictionary.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << dictionaryEyes;
	fs1.release();
	eyesFeaturesUnclustered.push_back(dictionaryEyes) ;
	for(int k=0;k<dictionarySize;k++){
		classesUnclustered_eyes.push_back(0) ;
	}
	
	BOWKMeansTrainer bowTrainerMouth(dictionarySize,tc,retries,flags);
	Mat dictionaryMouth=bowTrainerMouth.cluster(mouthFeaturesUnclustered) ;
	FileStorage fs2(dir_dico +"/mouth_dictionary.yml", FileStorage::WRITE);
	fs2 << "vocabulary" << dictionaryMouth;
	fs2.release();
	mouthFeaturesUnclustered.push_back(dictionaryMouth) ;
	for(int k=0;k<dictionarySize;k++){
		classesUnclustered_mouth.push_back(0) ;
	}

	BOWKMeansTrainer bowTrainerNose(dictionarySize,tc,retries,flags);
	Mat dictionaryNose=bowTrainerNose.cluster(noseFeaturesUnclustered) ;
	FileStorage fs3(dir_dico +"/nose_dictionary.yml", FileStorage::WRITE);
	fs3 << "vocabulary" << dictionaryNose;
	fs3.release();
	noseFeaturesUnclustered.push_back(dictionaryNose) ;
	for(int k=0;k<dictionarySize;k++){
		classesUnclustered_nose.push_back(0) ;
	}

	cout << " Dictionnaire OK" << endl ;
		if(pca){
		cout << endl;
		cout << "Show PCA for eyes " << endl ;
		showPCA(eyesFeaturesUnclustered,classesUnclustered_eyes,"Eyes");
		cout << "Show PCA for mouth " << endl ;
		showPCA(mouthFeaturesUnclustered,classesUnclustered_mouth,"Mouth");
		cout << "Show PCA for nose " << endl ;
		showPCA(noseFeaturesUnclustered,classesUnclustered_nose,"Nose");
	}

}

void buildPCAreducer(int nb_coponents,String db , vector<vector<int> > goodCols , bool verbose){

	String dir_classifiers = "../classifiers/" + db ;
	String dir_reducers = "../reducers/" + db ;
	String dir_labeled_data = "../data/" + db + "/labeled" ;
	String dir_classedFeatures = "../classedFeatures/" + db ;
	String dir_allFeatures = "../allFeatures/" + db ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	initModule_nonfree() ;

	//To store the SIFT descriptor of current image
	Mat descriptorLEye;
	Mat descriptorREye;
	Mat descriptorMouth;
	Mat descriptorNose;

	//To store all the descriptors that are extracted from all the images.
	Mat leyeFeaturesUnclustered;
	Mat reyeFeaturesUnclustered;
	Mat mouthFeaturesUnclustered;
	Mat noseFeaturesUnclustered;
	vector<int> classesUnclustered_eye;
	vector<int> classesUnclustered_mouth;
	vector<int> classesUnclustered_nose;

	//The SIFT feature extractor and descriptor
	const Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	const Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Mat img_with_sift;

	map<int,Mat> leye_training_set,reye_training_set,mouth_training_set,nose_training_set ;
	map<int,string> names ;
	int counter ;
	int index = 0 ;
	string celebrityName ;
	vector<int> classes ;


	//Images to extract feature descriptors and build the vocabulary
	int classPolitician=1;
	for (directory_iterator it1(dir_labeled_data); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		celebrityName = p.filename().string() ;
		cout << " -- Traite : " << celebrityName << endl ;
		Mat leye_samples = Mat(0,128,CV_32FC1);
		Mat reye_samples = Mat(0,128,CV_32FC1);
		Mat mouth_samples = Mat(0,128,CV_32FC1);
		Mat nose_samples = Mat(0,128,CV_32FC1);		
		counter = 0 ;
		for(directory_iterator it2(p); it2 != directory_iterator(); it2 ++){
			cout << it2->path() << endl ;
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
                // Loading file
                Mat input = imread(p2.string(), CV_LOAD_IMAGE_GRAYSCALE);
				vector<Rect> faces = detectFaces(face_classifier, input); 
				Rect searchZone ;
				vector<KeyPoint> keypoints_mouth ;
				vector<KeyPoint> keypoints_nose ;
				float alpha =0 ;
				vector<KeyPoint> keypoints_eyes;
				if(faces.size() >= 1){
					searchZone = faces[0] ;
					if(faces.size() > 1){
                        searchZone = selectBestFace(input, faces);
						cout << "Attention : plus d'un visage detecte" << endl ;
                    }
					if(verbose){
						rectangle(input,searchZone,Scalar(0,255,0),1,8,0) ;
						imshow("face",input) ;
						waitKey() ;
					}
					Rect searchEyeZone = faces[0] ;
					searchEyeZone.height /= 2 ;
					keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,verbose);
					Rect searchMouthZone = faces[0] ;
					searchMouthZone.height /= 2 ;
					searchMouthZone.y += searchMouthZone.height ;
					keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,verbose);
					keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,verbose) ; 
				}
				else{
					cout << "Attention : pas de visage detecte" << endl ;
				}
				if(keypoints_eyes.size() != 0){
					if(keypoints_eyes.size() != 2)
						cout << "ERROR nb d oeil retourne != 2" << endl ;
					int x1 = keypoints_eyes[0].pt.x ;
					int x2 = keypoints_eyes[1].pt.x ;
					cout << "eyes ok" << endl ;
					Mat descriptorEyes ;
                    extractor->compute(input, keypoints_eyes,descriptorEyes);
					if(x1 < x2){
						descriptorLEye = descriptorEyes.row(0) ;
						descriptorREye = descriptorEyes.row(1) ;
					}
					else{
						descriptorLEye = descriptorEyes.row(1) ;
						descriptorREye = descriptorEyes.row(0) ;
					}
					leyeFeaturesUnclustered.push_back(descriptorLEye);
					reyeFeaturesUnclustered.push_back(descriptorREye);
					classesUnclustered_eye.push_back(classPolitician);
				}
				else{
					descriptorLEye = Mat::zeros(1,128,CV_32FC1);
					descriptorREye = Mat::zeros(1,128,CV_32FC1);
				}
				if(keypoints_mouth.size() != 0){
					cout << "mouth ok" << endl ;
					extractor->compute(input, keypoints_mouth,descriptorMouth);
					mouthFeaturesUnclustered.push_back(descriptorMouth);
					classesUnclustered_mouth.push_back(classPolitician);
				}
				else
					descriptorMouth = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_nose.size() != 0){
					cout << "nose ok " << endl ;
					extractor->compute(input, keypoints_nose,descriptorNose);
					noseFeaturesUnclustered.push_back(descriptorNose);
					classesUnclustered_nose.push_back(classPolitician);
				}
				else
					descriptorNose = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_eyes.size() + keypoints_mouth.size() + keypoints_nose.size() != 0){
					counter ++ ;
					classes.push_back(index);
					leye_samples.push_back(descriptorLEye);
					reye_samples.push_back(descriptorREye);
					mouth_samples.push_back(descriptorMouth);
					nose_samples.push_back(descriptorNose);
				}
			}

		}
		if (counter > 0 ){
			leye_training_set.insert(pair<int,Mat>(index,leye_samples)) ;
			reye_training_set.insert(pair<int,Mat>(index,reye_samples)) ;
			mouth_training_set.insert(pair<int,Mat>(index,mouth_samples)) ;
			nose_training_set.insert(pair<int,Mat>(index,nose_samples)) ;
			names.insert(pair<int,string>(index,celebrityName)) ;
			index ++ ;
		}
		classPolitician++;
	}

	cout << "features extracted" << endl ;

	//Store features matrices
	writeMatToFile(leyeFeaturesUnclustered,classesUnclustered_eye,(dir_allFeatures+"/leye_features.csv")) ;
	writeMatToFile(reyeFeaturesUnclustered,classesUnclustered_eye,(dir_allFeatures+"/reye_features.csv")) ;
	writeMatToFile(mouthFeaturesUnclustered,classesUnclustered_mouth,(dir_allFeatures+"/mouth_features.csv")) ;
	writeMatToFile(noseFeaturesUnclustered,classesUnclustered_nose,(dir_allFeatures+"/nose_features.csv")) ;
	/*
	writeMatToFile(selectCols(goodCols[0],leyeFeaturesUnclustered),classesUnclustered_eye,(dir_allFeatures+"/leye_features.csv")) ;
	writeMatToFile(selectCols(goodCols[1],reyeFeaturesUnclustered),classesUnclustered_eye,(dir_allFeatures+"/reye_features.csv")) ;
	writeMatToFile(selectCols(goodCols[2],mouthFeaturesUnclustered),classesUnclustered_mouth,(dir_allFeatures+"/mouth_features.csv")) ;
	writeMatToFile(selectCols(goodCols[3],noseFeaturesUnclustered),classesUnclustered_nose,(dir_allFeatures+"/nose_features.csv")) ;
	*/
	if(pca){
		cout << endl;
		cout << "Show PCA for left eyes " << endl ;
		showPCA(leyeFeaturesUnclustered,classesUnclustered_eye,"Left Eye");
		cout << "Show PCA for right eyes " << endl ;
		showPCA(reyeFeaturesUnclustered,classesUnclustered_eye,"Right Eye");
		cout << "Show PCA for mouth " << endl ;
		showPCA(mouthFeaturesUnclustered,classesUnclustered_mouth,"Mouth");
		cout << "Show PCA for nose " << endl ;
		showPCA(noseFeaturesUnclustered,classesUnclustered_nose,"Nose");
	}

	Mat leye_reducer = computePCA(selectCols(goodCols[0],leyeFeaturesUnclustered),nb_coponents);
	Mat reye_reducer = computePCA(selectCols(goodCols[1],reyeFeaturesUnclustered),nb_coponents);
	Mat mouth_reducer = computePCA(selectCols(goodCols[2],mouthFeaturesUnclustered),nb_coponents);
	Mat nose_reducer = computePCA(selectCols(goodCols[3],noseFeaturesUnclustered),nb_coponents);
	cout << "Size reducers " << leye_reducer.size() << " " << mouth_reducer.size() << " " << nose_reducer.size() << endl ;
	
	map<int,Mat> training_set ;
	for(int k=0;k<index;k++){
		Mat samples(1,4*nb_coponents,CV_32FC1);
		//cout << "size : " << eyes_training_set[k].size() << endl ;
		//cout << "size mouth : " << mouth_training_set[k].size() << " " << nose_training_set[k].size() << endl ;
		Mat reduced_leye = selectCols(goodCols[0],leye_training_set[k]) * leye_reducer ;
		Mat reduced_reye = selectCols(goodCols[1],reye_training_set[k]) * reye_reducer ;
		Mat reduced_mouth = selectCols(goodCols[2],mouth_training_set[k]) * mouth_reducer ;
		Mat reduced_nose = selectCols(goodCols[3],nose_training_set[k]) * nose_reducer ;

		vector<Mat> matrices ;
		matrices.push_back(reduced_leye) ;
		matrices.push_back(reduced_reye) ;
		matrices.push_back(reduced_mouth) ;
		matrices.push_back(reduced_nose) ;
		hconcat( matrices,samples);
		training_set.insert(pair<int,Mat>(k,samples)) ;

		String dir = dir_classedFeatures + "/" + names[k] + ".yml" ;
		FileStorage fs(dir, FileStorage::WRITE);
		fs << "features" << samples;
		fs.release();
		//cout << samples << endl << endl ;
	}
	cout << training_set[0].size() << " " << training_set[1].size() << " " << training_set[2].size() << endl ;
	Mat allSamples ;
	//vector<Mat> matrices ;
	allSamples.push_back(training_set[0]) ;
	allSamples.push_back(training_set[1]) ;
	allSamples.push_back(training_set[2]) ;
	//vconcat( matrices,allSamples);
	vector<int> allclasses;
	allclasses.insert(allclasses.begin(),(training_set[0]).rows,1);
	allclasses.insert(allclasses.end(),(training_set[1]).rows,2);
	allclasses.insert(allclasses.end(),(training_set[2]).rows,3);
	cout << allclasses.size() << " " << allSamples.size() << endl ;
	writeMatToFile(allSamples,allclasses,(dir_allFeatures+"/allFeatures.csv"));
	

	CvSVMParams params = chooseSVMParams() ;
	vector<CvParamGrid> grids = chooseSVMGrids() ;
	int k_fold = 3 ;

	Mat labels,temp ;
	string fname ;

	for (int x=0;x<index;x++){
		Mat samples(0,3*nb_coponents,CV_32FC1) ;
		counter = 0 ;
		for(int y=0;y<index;y++){
			if(y != x){
				samples.push_back(training_set[y]) ;
				counter += training_set[y].rows ;
			}
		}
		samples.push_back(training_set[x]) ;
		labels = Mat::zeros(counter,1,CV_32FC1) ;
		temp = Mat::ones(training_set[x].rows,1,CV_32FC1) ;
		labels.push_back(temp);

		CvSVM classifier ;
		Mat samples_32f ;
		samples.convertTo(samples_32f, CV_32F);
		if(samples.rows != 0){ 
			//classifier.train(samples_32f,labels,Mat(),Mat(),params);
			classifier.train_auto(samples_32f,labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);		
		}
		else
			cout << "Le classifieur pour " <<  names[x] << " n'a pas pu etre construit" << endl ;

		fname = dir_classifiers + "/"+ names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		classifier.save(fname.c_str()) ;
		cout << "Stored" << endl ;
	}
	
	
	cout << "Classifieurs crees" << endl ;
	
	String d = dir_reducers + "/leye_reducer.yml" ;
	cout << d << endl ;
	FileStorage fs0(d, FileStorage::WRITE);
	fs0 << "reducer" << leye_reducer;
	fs0.release();
	FileStorage fs1((dir_reducers +"/reye_reducer.yml"), FileStorage::WRITE);
	fs1 << "reducer" << reye_reducer;
	fs1.release();
	FileStorage fs2((dir_reducers +"/mouth_reducer.yml"), FileStorage::WRITE);
	fs2 << "reducer" << mouth_reducer;
	fs2.release();
	FileStorage fs3((dir_reducers +"/nose_reducer.yml"), FileStorage::WRITE);
	fs3 << "reducer" << nose_reducer;
	fs3.release();

	/*
	vector<Mat> reducers ;
	reducers.push_back(leye_reducer) ;
	reducers.push_back(reye_reducer) ;
	reducers.push_back(mouth_reducer);
	reducers.push_back(nose_reducer) ;

	return reducers ;
	*/
}

void buildPCAreducer2(int nb_coponents,String db , vector<vector<int>> goodCols , bool verbose){

	String dir_leye_classifiers = "../classifiers/" + db + "/leye" ;
	String dir_reye_classifiers = "../classifiers/" + db + "/reye";
	String dir_nose_classifiers = "../classifiers/" + db + "/nose";
	String dir_mouth_classifiers = "../classifiers/" + db + "/mouth";
	String dir_reducers = "../reducers/" + db ;
	String dir_labeled_data = "../data/" + db + "/labeled" ;
	String dir_classedFeatures = "../classedFeatures/" + db ;
	String dir_allFeatures = "../allFeatures/" + db ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	initModule_nonfree() ;

	//To store the SIFT descriptor of current image
	Mat descriptorLEye;
	Mat descriptorREye;
	Mat descriptorMouth;
	Mat descriptorNose;
	//To store all the descriptors that are extracted from all the images.
	Mat leyeFeaturesUnclustered;
	Mat reyeFeaturesUnclustered;
	Mat mouthFeaturesUnclustered;
	Mat noseFeaturesUnclustered;
	vector<int> classesUnclustered_eye;
	vector<int> classesUnclustered_mouth;
	vector<int> classesUnclustered_nose;
	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Mat img_with_sift;

	map<int,Mat> leye_training_set,reye_training_set,mouth_training_set,nose_training_set ;
	map<int,string> names ;
	int counter ;
	int index = 0 ;
	string celebrityName ;


	//Images to extract feature descriptors and build the vocabulary
	for (directory_iterator it1(dir_labeled_data); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		celebrityName = p.filename().string() ;
		cout << " -- Traite : " << celebrityName << endl ;
		Mat leye_samples = Mat(0,128,CV_32FC1);
		Mat reye_samples = Mat(0,128,CV_32FC1);
		Mat mouth_samples = Mat(0,128,CV_32FC1);
		Mat nose_samples = Mat(0,128,CV_32FC1);		
		counter = 0 ;
		for(directory_iterator it2(p); it2 != directory_iterator(); it2 ++){
			cout << it2->path() << endl ;
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
                // Loading file
                Mat input = imread(p2.string(), CV_LOAD_IMAGE_GRAYSCALE);
				vector<Rect> faces = detectFaces(face_classifier, input); 
				Rect searchZone ;
				vector<KeyPoint> keypoints_mouth ;
				vector<KeyPoint> keypoints_nose ;
				float alpha =0 ;
				vector<KeyPoint> keypoints_eyes;
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
					if(verbose){
						rectangle(input,searchZone,Scalar(0,255,0),1,8,0) ;
						imshow("face",input) ;
						waitKey() ;
					}
					Rect searchEyeZone = faces[0] ;
					searchEyeZone.height /= 2 ;
					keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,verbose);
					Rect searchMouthZone = faces[0] ;
					searchMouthZone.height /= 2 ;
					searchMouthZone.y += searchMouthZone.height ;
					keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,verbose);
					keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,verbose) ; 
				}
				else{
					cout << "Attention : pas de visage detecte" << endl ;
				}
				if(keypoints_eyes.size() != 0){
					if(keypoints_eyes.size() != 2)
						cout << "ERROR nb d oeil retourne != 2" << endl ;
					int x1 = keypoints_eyes[0].pt.x ;
					int x2 = keypoints_eyes[1].pt.x ;
					cout << "eyes ok" << endl ;
					Mat descriptorEyes ;
                    extractor->compute(input, keypoints_eyes,descriptorEyes);
					if(x1 < x2){
						descriptorLEye = descriptorEyes.row(0) ;
						descriptorREye = descriptorEyes.row(1) ;
					}
					else{
						descriptorLEye = descriptorEyes.row(1) ;
						descriptorREye = descriptorEyes.row(0) ;
					}
					leyeFeaturesUnclustered.push_back(descriptorLEye);
					leye_samples.push_back(descriptorLEye);
					reyeFeaturesUnclustered.push_back(descriptorREye);
					reye_samples.push_back(descriptorREye);
					classesUnclustered_eye.push_back(index);
				}
				else{
					descriptorLEye = Mat::zeros(1,128,CV_32FC1);
					descriptorREye = Mat::zeros(1,128,CV_32FC1);
				}
				if(keypoints_mouth.size() != 0){
					cout << "mouth ok" << endl ;
					extractor->compute(input, keypoints_mouth,descriptorMouth);
					mouthFeaturesUnclustered.push_back(descriptorMouth);
					mouth_samples.push_back(descriptorMouth);
					classesUnclustered_mouth.push_back(index);
				}
				else
					descriptorMouth = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_nose.size() != 0){
					cout << "nose ok " << endl ;
					extractor->compute(input, keypoints_nose,descriptorNose);
					noseFeaturesUnclustered.push_back(descriptorNose);
					nose_samples.push_back(descriptorNose);
					classesUnclustered_nose.push_back(index);
				}
				else
					descriptorNose = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_eyes.size() + keypoints_mouth.size() + keypoints_nose.size() != 0){
					counter ++ ;
				}
			}

		}
		if (counter > 0 ){
			leye_training_set.insert(pair<int,Mat>(index,leye_samples)) ;
			reye_training_set.insert(pair<int,Mat>(index,reye_samples)) ;
			mouth_training_set.insert(pair<int,Mat>(index,mouth_samples)) ;
			nose_training_set.insert(pair<int,Mat>(index,nose_samples)) ;
			names.insert(pair<int,string>(index,celebrityName)) ;
			index ++ ;
		}
	}

	cout << "features extracted" << endl ;

	//Store features matrices
	writeMatToFile(leyeFeaturesUnclustered,classesUnclustered_eye,(dir_allFeatures+"/leye_features.csv")) ;
	writeMatToFile(reyeFeaturesUnclustered,classesUnclustered_eye,(dir_allFeatures+"/reye_features.csv")) ;
	writeMatToFile(mouthFeaturesUnclustered,classesUnclustered_mouth,(dir_allFeatures+"/mouth_features.csv")) ;
	writeMatToFile(noseFeaturesUnclustered,classesUnclustered_nose,(dir_allFeatures+"/nose_features.csv")) ;

	if(pca){
		cout << endl;
		cout << "Show PCA for left eyes " << endl ;
		showPCA(leyeFeaturesUnclustered,classesUnclustered_eye,"Left Eye");
		cout << "Show PCA for right eyes " << endl ;
		showPCA(reyeFeaturesUnclustered,classesUnclustered_eye,"Right Eye");
		cout << "Show PCA for mouth " << endl ;
		showPCA(mouthFeaturesUnclustered,classesUnclustered_mouth,"Mouth");
		cout << "Show PCA for nose " << endl ;
		showPCA(noseFeaturesUnclustered,classesUnclustered_nose,"Nose");
	}

	Mat leye_reducer = computePCA(selectCols(goodCols[0],leyeFeaturesUnclustered),nb_coponents);
	Mat reye_reducer = computePCA(selectCols(goodCols[1],reyeFeaturesUnclustered),nb_coponents);
	Mat mouth_reducer = computePCA(selectCols(goodCols[2],mouthFeaturesUnclustered),nb_coponents);
	Mat nose_reducer = computePCA(selectCols(goodCols[3],noseFeaturesUnclustered),nb_coponents);
	cout << "Size reducers " << leye_reducer.size() << " " << mouth_reducer.size() << " " << nose_reducer.size() << endl ;
	cout << leye_training_set[0].size() << " " << leye_training_set[1].size() << " " << leye_training_set[2].size() << endl ;


	map<int,Mat> leye_reduced_set,reye_reduced_set,mouth_reduced_set,nose_reduced_set ;
	/*Mat reduced_leye(1,nb_coponents,CV_32FC1);
	Mat reduced_reye(1,nb_coponents,CV_32FC1);
	Mat reduced_nose(1,nb_coponents,CV_32FC1);
	Mat reduced_mouth(1,nb_coponents,CV_32FC1);
	*/
	for(int k=0;k<index;k++){
		Mat reduced_leye = selectCols(goodCols[0],leye_training_set[k]) * leye_reducer ;
		Mat reduced_reye = selectCols(goodCols[1],reye_training_set[k]) * reye_reducer ;
		Mat reduced_mouth = selectCols(goodCols[2],mouth_training_set[k]) * mouth_reducer ;
		Mat reduced_nose = selectCols(goodCols[3],nose_training_set[k]) * nose_reducer ;

		cout << reduced_leye.size() << endl ;

		leye_reduced_set.insert(pair<int,Mat>(k,reduced_leye)) ;
		reye_reduced_set.insert(pair<int,Mat>(k,reduced_reye)) ;
		mouth_reduced_set.insert(pair<int,Mat>(k,reduced_mouth)) ;
		nose_reduced_set.insert(pair<int,Mat>(k,reduced_nose)) ;
	}
	cout << leye_reduced_set[0].size() << " " << leye_reduced_set[1].size() << " " << leye_reduced_set[2].size() << endl ;
	cout << mouth_reduced_set[0].size() << " " << mouth_reduced_set[1].size() << " " << mouth_reduced_set[2].size() << endl ;
	cout << nose_reduced_set[0].size() << " " << nose_reduced_set[1].size() << " " << nose_reduced_set[2].size() << endl ;
	
	CvSVMParams params = chooseSVMParams() ;
	vector<CvParamGrid> grids = chooseSVMGrids() ;
	int k_fold = 6 ;

	string fname ;

	for (int x=0;x<index;x++){
		Mat leye_samples(0,3*nb_coponents,CV_32FC1) ;
		Mat reye_samples(0,3*nb_coponents,CV_32FC1) ;
		Mat nose_samples(0,3*nb_coponents,CV_32FC1) ;
		Mat mouth_samples(0,3*nb_coponents,CV_32FC1) ;
		int leye_counter = 0 ;
		int reye_counter = 0 ;
		int nose_counter = 0 ;
		int mouth_counter = 0 ;
		for(int y=0;y<index;y++){
			if(y != x){
				leye_samples.push_back(leye_reduced_set[y]) ;
				leye_counter += leye_reduced_set[y].rows ;

				reye_samples.push_back(reye_reduced_set[y]) ;
				reye_counter += reye_reduced_set[y].rows ;
				
				nose_samples.push_back(nose_reduced_set[y]) ;
				nose_counter += nose_reduced_set[y].rows ;

				mouth_samples.push_back(mouth_reduced_set[y]) ;
				mouth_counter += mouth_reduced_set[y].rows ;
			}
		}
		leye_samples.push_back(leye_reduced_set[x]) ;
		reye_samples.push_back(reye_reduced_set[x]) ;
		nose_samples.push_back(nose_reduced_set[x]) ;
		mouth_samples.push_back(mouth_reduced_set[x]) ;

		Mat leye_labels = Mat::zeros(leye_counter,1,CV_32FC1) ;
		Mat reye_labels = Mat::zeros(reye_counter,1,CV_32FC1) ;
		Mat nose_labels = Mat::zeros(nose_counter,1,CV_32FC1) ;
		Mat mouth_labels = Mat::zeros(mouth_counter,1,CV_32FC1) ;

		Mat temp = Mat::ones(leye_reduced_set[x].rows,1,CV_32FC1) ;
		leye_labels.push_back(temp);
		
		temp = Mat::ones(reye_reduced_set[x].rows,1,CV_32FC1) ;
		reye_labels.push_back(temp);
		
		temp = Mat::ones(nose_reduced_set[x].rows,1,CV_32FC1) ;
		nose_labels.push_back(temp);
		
		temp = Mat::ones(mouth_reduced_set[x].rows,1,CV_32FC1) ;
		mouth_labels.push_back(temp);

		CvSVM leye_classifier,reye_classifier,nose_classifier,mouth_classifier ;
		Mat leye_samples_32f,reye_samples_32f,nose_samples_32f,mouth_samples_32f ;
		leye_samples.convertTo(leye_samples_32f, CV_32F);
		reye_samples.convertTo(reye_samples_32f, CV_32F);
		nose_samples.convertTo(nose_samples_32f, CV_32F);
		mouth_samples.convertTo(mouth_samples_32f, CV_32F);
		if(leye_samples.rows * reye_samples.rows * nose_samples.rows * mouth_samples.rows != 0){ 
			//classifier.train(samples_32f,labels,Mat(),Mat(),params);
			leye_classifier.train_auto(leye_samples_32f,leye_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);		
			reye_classifier.train_auto(reye_samples_32f,reye_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);		
			nose_classifier.train_auto(nose_samples_32f,nose_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);		
			mouth_classifier.train_auto(mouth_samples_32f,mouth_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);		
		}
		else
			cout << "Le classifieur pour " <<  names[x] << " n'a pas pu etre construit" << endl ;

		fname = dir_leye_classifiers + "/"+ names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		leye_classifier.save(fname.c_str()) ;
		cout << "Stored" << endl ;

		fname = dir_reye_classifiers + "/"+ names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		reye_classifier.save(fname.c_str()) ;
		cout << "Stored" << endl ;

		fname = dir_nose_classifiers + "/"+ names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		nose_classifier.save(fname.c_str()) ;
		cout << "Stored" << endl ;
		
		fname = dir_mouth_classifiers + "/"+ names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		mouth_classifier.save(fname.c_str()) ;
		cout << "Stored" << endl ;
	}
	
	
	cout << "Classifieurs crees" << endl ;
	
	String d = dir_reducers + "/leye_reducer.yml" ;
	cout << d << endl ;
	FileStorage fs0(d, FileStorage::WRITE);
	fs0 << "reducer" << leye_reducer;
	fs0.release();
	FileStorage fs1((dir_reducers +"/reye_reducer.yml"), FileStorage::WRITE);
	fs1 << "reducer" << reye_reducer;
	fs1.release();
	FileStorage fs2((dir_reducers +"/mouth_reducer.yml"), FileStorage::WRITE);
	fs2 << "reducer" << mouth_reducer;
	fs2.release();
	FileStorage fs3((dir_reducers +"/nose_reducer.yml"), FileStorage::WRITE);
	fs3 << "reducer" << nose_reducer;
	fs3.release();

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

Mat computePCA(Mat featuresUnclustered,int nb_coponents){
    PCA principalCA(featuresUnclustered, Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
    Mat eigenvectors = principalCA.eigenvectors.clone();
	Mat principalVectors = Mat(nb_coponents , eigenvectors.cols,CV_32FC1);

    for(int j=0;j<nb_coponents;j++){
		principalVectors.row(j) = eigenvectors.row(j) ;
	}

	return principalVectors.t() ;
}

int init_bowDE(BOWImgDescriptorExtractor& mouth_bowDE,BOWImgDescriptorExtractor& eyes_bowDE,BOWImgDescriptorExtractor& nose_bowDE,String dir){
	//prepare BOW descriptor extractor from the dictionary
	Mat eyes_dictionary;
	Mat nose_dictionary; 
	Mat mouth_dictionary; 
    FileStorage fs1(dir + "/mouth_dictionary.yml", FileStorage::READ);
    fs1["vocabulary"] >> mouth_dictionary;
    fs1.release();
	FileStorage fs2(dir + "/nose_dictionary.yml", FileStorage::READ);
    fs2["vocabulary"] >> nose_dictionary;
    fs2.release();
	FileStorage fs3(dir + "/eye_dictionary.yml", FileStorage::READ);
    fs3["vocabulary"] >> eyes_dictionary;
    fs3.release();
    cout << "dictionary loaded" << endl ;

	//Set the dictionary with the vocabulary we created in the first step
    mouth_bowDE.setVocabulary(mouth_dictionary);
    nose_bowDE.setVocabulary(nose_dictionary);
    eyes_bowDE.setVocabulary(eyes_dictionary);

	int dim = mouth_dictionary.rows ;

	return dim ;
}



//after clustering
int createSVMClassifier(String db) {

	String dir_classifiers = "../classifiers/" + db ;
	String dir_dico = "../dictionaries/" + db ;
	String dir_labeled_data = "../data/" + db + "/labeled" ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();

    //create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher) ;
	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT") ; 
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT") ; 

    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor mouth_bowDE(extractor,matcher);
    BOWImgDescriptorExtractor nose_bowDE(extractor,matcher);
    BOWImgDescriptorExtractor eyes_bowDE(extractor,matcher);

	int dim = init_bowDE(mouth_bowDE,eyes_bowDE,nose_bowDE,dir_dico);
	vector<Mat> hists ;

    //init
	string filename ;
    Mat input ;
    vector<KeyPoint> keypoints;
    Mat mouth_bowDescriptor;
	Mat eyes_bowDescriptor;
	Mat nose_bowDescriptor;
	map<int,Mat> training_set ;
	map<int,string> names ;
	int counter ;
	int index = 0 ;
	string celebrityName ;
	vector<int> classes ;
	Mat all_samples ;

	for (directory_iterator it1(dir_labeled_data); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		celebrityName = p.filename().string() ;
		cout << " -- Traite : " << celebrityName << endl ;
		Mat samples(0,3*dim,CV_32FC1) ;
		counter = 0 ;
		for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
			path p2 = it2->path() ;
			if(is_regular_file(it2->status())){
                // Load the image
				filename = p2.string();
				cout << filename << endl;
				input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
				vector<Rect> faces = detectFaces(face_classifier, input); 
				Rect searchZone ;
				vector<KeyPoint> keypoints_mouth ;
				vector<KeyPoint> keypoints_nose ;
				float alpha = 0 ;
				vector<KeyPoint> keypoints_eyes ;
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
					Rect searchEyeZone = faces[0] ;
					searchEyeZone.height /= 2 ;
					keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,false);
					Rect searchMouthZone = faces[0] ;
					searchMouthZone.height /= 2 ;
					searchMouthZone.y += searchMouthZone.height ;
					keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,false);
					keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,false) ; 
				}
				else{
					cout << "Attention : pas de visage detecte" << endl ;
				}
				//TODO : clean that
				Mat full_descriptor = Mat::zeros(1,3*dim,CV_32FC1) ; //int or float ?
				if(keypoints_eyes.size() != 0){
					cout << "eyes ok" << endl ;
                    eyes_bowDE.compute(input, keypoints_eyes,eyes_bowDescriptor);
					for(int j =0; j<dim;j++){
						full_descriptor.at<float>(0,j)=eyes_bowDescriptor.at<float>(0,j) ;
					}
				}
				if(keypoints_mouth.size() != 0){
					cout << "mouth ok" << endl ;
					mouth_bowDE.compute(input, keypoints_mouth,mouth_bowDescriptor);
					for(int j =0; j<dim;j++){
						full_descriptor.at<float>(0,dim+j)=mouth_bowDescriptor.at<float>(0,j) ;
					}
				}
				if(keypoints_nose.size() != 0){
					cout << "nose ok " << endl ;
					nose_bowDE.compute(input, keypoints_nose,nose_bowDescriptor);
					for(int j =0; j<dim;j++){
						full_descriptor.at<float>(0,2*dim+j)=nose_bowDescriptor.at<float>(0,j) ;
					}
				}
				if(keypoints_eyes.size() + keypoints_mouth.size() + keypoints_nose.size() != 0){
					counter ++ ;
					classes.push_back(index);
					cout << full_descriptor << endl << endl ;
					samples.push_back(full_descriptor);
					if(hists.empty())
						hists.push_back(full_descriptor) ;
					bool estPas =true ;
					for (vector<Mat>::iterator it = hists.begin() ; it != hists.end() && estPas; it ++){ 
						bool estIdentique = true;
						for(int k=0; k < 3*dim && estIdentique; k++){
							if(full_descriptor.at<float>(0,k) != (*it).at<float>(0,k))
								estIdentique = false ;
						}
						if(estIdentique)
							estPas = false;
					}
					if(estPas){
						hists.push_back(full_descriptor) ;
						cout << "New hist. Count = " << hists.size() << endl ; 
					}
					cout << "Nombre d'histogrammes " << hists.size() << endl ;
					cout << ">>>>>>>>> Success for : " << it2->path() << endl << endl;
				}
			}
		}
		if (counter > 0 ){
			training_set.insert(pair<int,Mat>(index,samples)) ;
			all_samples.push_back(samples);
			names.insert(pair<int,string>(index,celebrityName)) ;
			index ++ ;
		}
	}
    
	cout << "Il y a " << hists.size() << " différents" << endl ;
	cout << "Images chargees et analysees" << endl ;
	cout << "Classes " << classes.size() << endl ;

	showPCA(all_samples,classes,"descriptors") ;

	CvSVMParams params = chooseSVMParams() ;
	vector<CvParamGrid> grids = chooseSVMGrids() ;
	int k_fold = 5 ;

	Mat labels,temp ;
	string fname ;

	for (int x=0;x<index;x++){
		Mat samples(0,3*dim,CV_32FC1) ;
		counter = 0 ;

		for(int y=0;y<index;y++){
			if(y != x){
				samples.push_back(training_set[y]) ;
				counter += training_set[y].rows ;
			}
		}
		samples.push_back(training_set[x]) ;
		labels = Mat::zeros(counter,1,CV_32FC1) ;
		temp = Mat::ones(training_set[x].rows,1,CV_32FC1) ;
		labels.push_back(temp);

		CvSVM classifier ;
		Mat samples_32f ;
		samples.convertTo(samples_32f, CV_32F);
		if(samples.rows != 0){ 
			classifier.train(samples_32f,labels,Mat(),Mat(),params);
			//classifier.train_auto(samples_32f,labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);		
		}
		else
			cout << "Le classifieur pour " <<  names[x] << " n'a pas pu etre construit" << endl ;

		fname = dir_classifiers + names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		classifier.save(fname.c_str()) ;
		cout << "Stored" << endl ;
	}
	
	
	cout << "Classifieurs crees" << endl ;

	return index ;
	
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

// Do NOT use that !
map<int,CvSVM*> loadSVMClassifier(void){
	map<int,CvSVM*> classifiers ;
	char * path = new char[15];
	for (int x=0 ; x<3 ; x++){
		sprintf(path,"../classifiers/classifier%i.yml",x);
		CvSVM my_svm ;
		my_svm.load(path) ;
		classifiers.insert(pair<int,CvSVM*>(x,&my_svm)) ;
		cout << "classifieur " << x << " bien charge" << endl ;
		waitKey() ;
	}

	return classifiers ;
}

void predict(String db){
	
	String dir_classifiers = "../classifiers/" + db ;
	String dir_dico = "../dictionaries/" + db ;
	String dir_labeled_data = "../data/" + db + "/labeled" ;
	String dir_unlabeled_data = "../data/" + db + "/unlabeled" ;

	/*
	int count_folders = 0 ; //pour plus tard ...
	for(directory_iterator it("../classifiers"); it != directory_iterator(); ++it){
		count_folders ++ ;
	}
	*/
	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	CvSVM classifiers[3] ;
	String celebrities[3] ;
	int index = 0 ;
	for (directory_iterator it(dir_classifiers); it != directory_iterator() ; it++) { 
		path p = it->path() ;
		if(is_regular_file(it->status())){
			classifiers[index].load(p.string().c_str()) ;
			celebrities[index] = p.stem().string() ;
			cout << "Added " << p.string() << " = " << p.stem().string() << endl ;
			index ++ ;
		}
	}

	cout << "Classifieurs charges" << endl ;

	 //prepare BOW descriptor extractor from the dictionary
	Mat eyes_dictionary;
	Mat nose_dictionary; 
	Mat mouth_dictionary; 
    FileStorage fs1(dir_dico + "/mouth_dictionary.yml", FileStorage::READ);
    fs1["vocabulary"] >> mouth_dictionary;
    fs1.release();
	FileStorage fs2(dir_dico + "/nose_dictionary.yml", FileStorage::READ);
    fs2["vocabulary"] >> nose_dictionary;
    fs2.release();
	FileStorage fs3(dir_dico + "/eye_dictionary.yml", FileStorage::READ);
    fs3["vocabulary"] >> eyes_dictionary;
    fs3.release();
    cout << "dictionary loaded" << endl ;

    //create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher) ;
	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT") ; 
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT") ; 

    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor mouth_bowDE(extractor,matcher);
    BOWImgDescriptorExtractor nose_bowDE(extractor,matcher);
    BOWImgDescriptorExtractor eyes_bowDE(extractor,matcher);
	//Set the dictionary with the vocabulary we created in the first step
    mouth_bowDE.setVocabulary(mouth_dictionary);
    nose_bowDE.setVocabulary(nose_dictionary);
    eyes_bowDE.setVocabulary(eyes_dictionary);

	Mat input ;
    vector<KeyPoint> keypoints;  
    Mat mouth_bowDescriptor,eyes_bowDescriptor,nose_bowDescriptor;
	string filename;

	String dir_data[2] ;
	dir_data[0] = dir_unlabeled_data ;
	dir_data[1] = dir_labeled_data ;

	for(int k = 0 ; k<2 ; k++){
		for (directory_iterator it1(dir_data[k]); it1 != directory_iterator() ; it1++) { //each folder in ../data
			path p = it1->path() ;
			cout << "Folder " << p.string() << endl ;
			waitKey() ;
			for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){ //each file in the folder    
				cout << it2->path() << endl ;
				path p2 = it2->path() ;
				if(is_regular_file(it2->status())){
					filename = p2.string() ;
					input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
					vector<Rect> faces = detectFaces(face_classifier, input); 
					Rect searchZone ;
					vector<KeyPoint> keypoints_mouth ;
					vector<KeyPoint> keypoints_nose ;
					float alpha = 0 ;
					vector<KeyPoint> keypoints_eyes ;				
					if(faces.size() >= 1){
						if(faces.size() > 1)
							cout << "Attention : plus d'un visage detecte" << endl ;
						searchZone = faces[0] ;
						Rect searchEyeZone = faces[0] ;
						searchEyeZone.height /= 2 ;
						keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,false);
						Rect searchMouthZone = faces[0] ;
						searchMouthZone.height /= 2 ;
						searchMouthZone.y += searchMouthZone.height ;
						keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,false);
						keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,false) ; 
					}
					else{
						cout << "Attention : pas de visage detecte" << endl ;
					}
					if(keypoints_eyes.size() != 0 && keypoints_mouth.size() != 0 && keypoints_nose.size() != 0){
						eyes_bowDE.compute(input, keypoints_eyes,eyes_bowDescriptor);
						mouth_bowDE.compute(input, keypoints_mouth,mouth_bowDescriptor);
						nose_bowDE.compute(input, keypoints_nose,nose_bowDescriptor);

						Mat full_descriptor = Mat(1,3*eyes_dictionary.rows,CV_32FC1) ; //int or float ?
						for(int j =0; j<eyes_dictionary.rows;j++){
							full_descriptor.at<float>(0,j)=eyes_bowDescriptor.at<float>(0,j) ;
						}	
						for(int j =0; j<eyes_dictionary.rows;j++){
							full_descriptor.at<float>(0,eyes_dictionary.rows+j)=mouth_bowDescriptor.at<float>(0,j) ;
						}
						for(int j =0; j<eyes_dictionary.rows;j++){
							full_descriptor.at<float>(0,2*eyes_dictionary.rows+j)=nose_bowDescriptor.at<float>(0,j) ;
						}

						float min = 2  ;
						int prediction =0 ;
						for(int x=0;x<3;x++){
							if (classifiers[x].predict(full_descriptor,true) < min){
								prediction = x ;
								min = classifiers[x].predict(full_descriptor,true) ;
							}
							cout << classifiers[x].predict(full_descriptor,true) << " " ;
						}
						cout <<endl ;
						cout << "Classe retenue : " << prediction << " = " << celebrities[prediction] << endl ;
					}
					else{
						cout << "No keypoints found" << endl ;
					}
					cout << endl ;
				}
			}
		}
	}

}


void predictPCA(String db,vector<vector<int> > goodCols){

	String dir_classifiers = "../classifiers/" + db ;
	String dir_reducers = "../reducers/" + db ;
	String dir_labeled_data = "../data/" + db + "/labeled" ;
	String dir_unlabeled_data = "../data/" + db + "/unlabeled" ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	CvSVM classifiers[nb_celebrities] ;
	String celebrities[nb_celebrities] ;
	int index = 0 ;
	for (directory_iterator it(dir_classifiers); it != directory_iterator() ; it++) { 
		path p = it->path() ;
		if(is_regular_file(it->status())){
			classifiers[index].load(p.string().c_str()) ;
			celebrities[index] = p.stem().string() ;
			cout << "Added " << p.string() << " = " << p.stem().string() << endl ;
			index ++ ;
		}
	}
	if(index != nb_celebrities)
		cout << "Erreur : il y a un nombre différent de classifieurs et de celebrites" << endl ;

	cout << "Classifieurs charges" << endl ;

	Mat reducer_leye,reducer_reye,reducer_mouth,reducer_nose ;
	Mat r0,r1,r2,r3 ;
	FileStorage fs0((dir_reducers+ "/leye_reducer.yml"), FileStorage::READ);
	fs0["reducer"] >> r0;
	fs0.release();
	FileStorage fs1((dir_reducers+ "/reye_reducer.yml"), FileStorage::READ);
	fs1["reducer"] >> r1;
	fs1.release();
	FileStorage fs2((dir_reducers+ "/mouth_reducer.yml"), FileStorage::READ);
	fs2["reducer"] >> r2;
	fs2.release();
	FileStorage fs3((dir_reducers+ "/nose_reducer.yml"), FileStorage::READ);
	fs3["reducer"] >> r3;
	fs3.release();

	r0.convertTo(reducer_leye,CV_32FC1);
	r1.convertTo(reducer_reye,CV_32FC1);
	r2.convertTo(reducer_mouth,CV_32FC1);
	r3.convertTo(reducer_nose,CV_32FC1);
	/*
	Mat reducer_leye = reducers[0] ;
	Mat reducer_reye = reducers[1] ;
	Mat reducer_mouth = reducers[2] ;
	Mat reducer_nose = reducers[3] ;
	*/
	cout << reducer_leye.size()  << " " << reducer_reye.size()  << " "<< reducer_mouth.size() << " " << reducer_nose.size() << endl ;
	cout << "Reducers loaded" << endl ;

	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT") ; 
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT") ; 

	Mat input ;
    vector<KeyPoint> keypoints;  
	string filename;
	string celebrityName ;
	map<string,pair<int,int> > results[2] ;

	String dir_data[2] ;
	dir_data[0] = dir_unlabeled_data ;
	dir_data[1] = dir_labeled_data ;

	for(int k =0; k<2;k++){
		for (directory_iterator it1(dir_data[k]); it1 != directory_iterator() ; it1++){
			path p = it1->path() ;
			celebrityName = p.filename().string() ;
			cout << " -- Traite : " << celebrityName << endl ;	
			int nb_images = 0 ;
			int nb_error = 0 ;
			for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
				cout << it2->path() << endl ;
				path p2 = it2->path() ;
				if(is_regular_file(it2->status())){
					// Loading file
					Mat input = imread(p2.string(), CV_LOAD_IMAGE_GRAYSCALE);
					vector<Rect> faces = detectFaces(face_classifier, input); 
					Rect searchZone ;
					vector<KeyPoint> keypoints_mouth ;
					vector<KeyPoint> keypoints_nose ;
					float alpha =0 ;
					vector<KeyPoint> keypoints_eyes;
					if(faces.size() >= 1){
						if(faces.size() > 1)
							cout << "Attention : plus d'un visage detecte" << endl ;
						searchZone = faces[0] ;
						Rect searchEyeZone = faces[0] ;
						searchEyeZone.height /= 2 ;
						keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,false);
						Rect searchMouthZone = faces[0] ;
						searchMouthZone.height /= 2 ;
						searchMouthZone.y += searchMouthZone.height ;
						keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,false);
						keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,false) ; 
					}
					else{
						cout << "Attention : pas de visage detecte" << endl ;
					}
					Mat descriptorLEye, descriptorREye;
					if(keypoints_eyes.size() != 0){
						if(keypoints_eyes.size() != 2)
							cout << "ERROR nb d oeil retourne != 2" << endl ;
						int x1 = keypoints_eyes[0].pt.x ;
						int x2 = keypoints_eyes[1].pt.x ;
						cout << "eyes ok" << endl ;
						Mat descriptorEyes ;
						extractor->compute(input, keypoints_eyes,descriptorEyes);
						if(x1 < x2){
							descriptorLEye = descriptorEyes.row(0) ;
							descriptorREye = descriptorEyes.row(1) ;
						}
						else{
							descriptorLEye = descriptorEyes.row(1) ;
							descriptorREye = descriptorEyes.row(0) ;
						}
					}
					else{
						descriptorLEye = Mat::zeros(1,128,CV_32FC1);
						descriptorREye = Mat::zeros(1,128,CV_32FC1);
					}
					Mat descriptorMouth ;
					if(keypoints_mouth.size() != 0){
						cout << "mouth ok" << endl ;
						extractor->compute(input, keypoints_mouth,descriptorMouth);
					}
					else
						descriptorMouth = Mat::zeros(1,128,CV_32FC1);
					Mat descriptorNose ;
					if(keypoints_nose.size() != 0){
						cout << "nose ok " << endl ;
						extractor->compute(input, keypoints_nose,descriptorNose);
					}
					else
						descriptorNose = Mat::zeros(1,128,CV_32FC1);
					if(keypoints_eyes.size() + keypoints_mouth.size() + keypoints_nose.size() != 0){
						nb_images ++ ;
						//cout << "sizes " << reducer_leye.size() << " " << descriptorLEye.size() << endl ;
						Mat leye_samples = selectCols(goodCols[0],descriptorLEye) * reducer_leye;
						Mat reye_samples = selectCols(goodCols[1],descriptorREye) * reducer_reye;
						Mat mouth_samples = selectCols(goodCols[2],descriptorMouth) * reducer_mouth;
						Mat nose_samples = selectCols(goodCols[3],descriptorNose) * reducer_nose ;
						vector<Mat> matrices ; 
						matrices.push_back(leye_samples) ;
						matrices.push_back(reye_samples) ;
						matrices.push_back(mouth_samples) ;
						matrices.push_back(nose_samples) ;
						//cout << leye_samples.size() << endl ;
						Mat full_descriptor = Mat (1,4*leye_samples.cols,CV_32FC1) ;
						hconcat(matrices,full_descriptor) ;
						//cout << "size full descriptor : " << full_descriptor.size() << endl ;
					
						float min = 2  ;
						int prediction =0 ;
						for(int x=0;x<3;x++){
							if (classifiers[x].predict(full_descriptor,true) < min){
								prediction = x ;
								min = classifiers[x].predict(full_descriptor,true) ;
							}
							cout << classifiers[x].predict(full_descriptor,true) << " " ;
						}
						cout <<endl ;
						cout << "Classe retenue : " << prediction << " = " << celebrities[prediction] << endl ;
						if(celebrityName.compare(celebrities[prediction])){
							cout << "Erreur de classification" << endl ;
							nb_error ++ ;
						}
					}
					else{
						cout << "No keypoints found" << endl ;
					}
					cout << endl ;
				}
			}
			results[k].insert(pair<string,pair<int,int> >(celebrityName,pair<int,int>(nb_error,nb_images)));
		}
	}
	

	cout << "Resultats : " << endl ;
	
	for (int k=0;k<nb_celebrities;k++){
		cout << "- " << celebrities[k] << " : " << endl ;
		cout << "    unlabeled : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		cout << "    labeled : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl << endl ;
	}
}

void predictPCA2(String db,vector<vector<int>> goodCols){

	String dir_leye_classifiers = "../classifiers/" + db + "/leye" ;
	String dir_reye_classifiers = "../classifiers/" + db + "/reye";
	String dir_nose_classifiers = "../classifiers/" + db + "/nose";
	String dir_mouth_classifiers = "../classifiers/" + db + "/mouth";
	String dir_reducers = "../reducers/" + db ;
	String dir_labeled_data = "../data/" + db + "/labeled" ;
	String dir_unlabeled_data = "../data/" + db + "/unlabeled" ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	CvSVM leye_classifiers[nb_celebrities] ;
	CvSVM reye_classifiers[nb_celebrities] ;
	CvSVM nose_classifiers[nb_celebrities] ;
	CvSVM mouth_classifiers[nb_celebrities] ;
	String celebrities[nb_celebrities] ;
	int index = 0 ;
	for (directory_iterator it(dir_leye_classifiers); it != directory_iterator() ; it++) { 
		path p = it->path() ;
		if(is_regular_file(it->status())){
			leye_classifiers[index].load(p.string().c_str()) ;
			celebrities[index] = p.stem().string() ;
			cout << "Added " << p.string() << " = " << p.stem().string() << endl ;
			index ++ ;
		}
	}

	index = 0 ;
	for (directory_iterator it(dir_reye_classifiers); it != directory_iterator() ; it++) { 
		path p = it->path() ;
		if(is_regular_file(it->status())){
			reye_classifiers[index].load(p.string().c_str()) ;
			celebrities[index] = p.stem().string() ;
			cout << "Added " << p.string() << " = " << p.stem().string() << endl ;
			index ++ ;
		}
	}

	index = 0 ;
	for (directory_iterator it(dir_nose_classifiers); it != directory_iterator() ; it++) { 
		path p = it->path() ;
		if(is_regular_file(it->status())){
			nose_classifiers[index].load(p.string().c_str()) ;
			celebrities[index] = p.stem().string() ;
			cout << "Added " << p.string() << " = " << p.stem().string() << endl ;
			index ++ ;
		}
	}

	index = 0 ;
	for (directory_iterator it(dir_mouth_classifiers); it != directory_iterator() ; it++) { 
		path p = it->path() ;
		if(is_regular_file(it->status())){
			mouth_classifiers[index].load(p.string().c_str()) ;
			celebrities[index] = p.stem().string() ;
			cout << "Added " << p.string() << " = " << p.stem().string() << endl ;
			index ++ ;
		}
	}	
	
	if(index != nb_celebrities)
		cout << "Erreur : il y a un nombre différent de classifieurs et de celebrites" << endl ;

	cout << "Classifieurs charges" << endl ;

	Mat reducer_leye,reducer_reye,reducer_mouth,reducer_nose ;
	Mat r0,r1,r2,r3 ;
	FileStorage fs0((dir_reducers+ "/leye_reducer.yml"), FileStorage::READ);
	fs0["reducer"] >> r0;
	fs0.release();
	FileStorage fs1((dir_reducers+ "/reye_reducer.yml"), FileStorage::READ);
	fs1["reducer"] >> r1;
	fs1.release();
	FileStorage fs2((dir_reducers+ "/mouth_reducer.yml"), FileStorage::READ);
	fs2["reducer"] >> r2;
	fs2.release();
	FileStorage fs3((dir_reducers+ "/nose_reducer.yml"), FileStorage::READ);
	fs3["reducer"] >> r3;
	fs3.release();

	r0.convertTo(reducer_leye,CV_32FC1);
	r1.convertTo(reducer_reye,CV_32FC1);
	r2.convertTo(reducer_mouth,CV_32FC1);
	r3.convertTo(reducer_nose,CV_32FC1);
	/*
	Mat reducer_leye = reducers[0] ;
	Mat reducer_reye = reducers[1] ;
	Mat reducer_mouth = reducers[2] ;
	Mat reducer_nose = reducers[3] ;
	*/
	cout << reducer_leye.size()  << " " << reducer_reye.size()  << " "<< reducer_mouth.size() << " " << reducer_nose.size() << endl ;
	cout << "Reducers loaded" << endl ;

	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT") ; 
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT") ; 

	Mat input ;
    vector<KeyPoint> keypoints;  
	string filename;
	string celebrityName ;
	map<string,pair<int,int>> results[2] ;

	String dir_data[2] ;
	dir_data[0] = dir_unlabeled_data ;
	dir_data[1] = dir_labeled_data ;

	for(int k =0; k<2;k++){
		for (directory_iterator it1(dir_data[k]); it1 != directory_iterator() ; it1++){
			path p = it1->path() ;
			celebrityName = p.filename().string() ;
			cout << " -- Traite : " << celebrityName << endl ;	
			int nb_images = 0 ;
			int nb_error = 0 ;
			for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
				float prediction[nb_celebrities] ;
				for(int x=0; x < nb_celebrities; x++){
					prediction[x] = 0;
				}
				cout << it2->path() << endl ;
				path p2 = it2->path() ;
				if(is_regular_file(it2->status())){
					// Loading file
					Mat input = imread(p2.string(), CV_LOAD_IMAGE_GRAYSCALE);
					vector<Rect> faces = detectFaces(face_classifier, input); 
					Rect searchZone ;
					vector<KeyPoint> keypoints_mouth ;
					vector<KeyPoint> keypoints_nose ;
					float alpha =0 ;
					vector<KeyPoint> keypoints_eyes;
					if(faces.size() >= 1){
						if(faces.size() > 1)
							cout << "Attention : plus d'un visage detecte" << endl ;
						searchZone = faces[0] ;
						Rect searchEyeZone = faces[0] ;
						searchEyeZone.height /= 2 ;
						keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,false);
						Rect searchMouthZone = faces[0] ;
						searchMouthZone.height /= 2 ;
						searchMouthZone.y += searchMouthZone.height ;
						keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,false);
						keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,false) ; 
					}
					else{
						cout << "Attention : pas de visage detecte" << endl ;
					}
					Mat descriptorLEye, descriptorREye;
					if(keypoints_eyes.size() != 0){
						if(keypoints_eyes.size() != 2)
							cout << "ERROR nb d oeil retourne != 2" << endl ;
						int x1 = keypoints_eyes[0].pt.x ;
						int x2 = keypoints_eyes[1].pt.x ;
						cout << "eyes ok" << endl ;
						Mat descriptorEyes ;
						extractor->compute(input, keypoints_eyes,descriptorEyes);
						if(x1 < x2){
							descriptorLEye = descriptorEyes.row(0) ;
							descriptorREye = descriptorEyes.row(1) ;
						}
						else{
							descriptorLEye = descriptorEyes.row(1) ;
							descriptorREye = descriptorEyes.row(0) ;
						}
						Mat leye_samples = selectCols(goodCols[0],descriptorLEye) * reducer_leye;
						Mat reye_samples = selectCols(goodCols[1],descriptorREye) * reducer_reye;

						for(int x=0;x<nb_celebrities;x++){
							prediction[x] += leye_classifiers[x].predict(leye_samples,true) ;
							prediction[x] += reye_classifiers[x].predict(reye_samples,true) ;
						}
					}
					Mat descriptorMouth ;
					if(keypoints_mouth.size() != 0){
						cout << "mouth ok" << endl ;
						extractor->compute(input, keypoints_mouth,descriptorMouth);
						for(int x=0;x<nb_celebrities;x++){
							prediction[x] += mouth_classifiers[x].predict(selectCols(goodCols[2],descriptorMouth)* reducer_mouth,true) ;
						}
					}
					Mat descriptorNose ;
					if(keypoints_nose.size() != 0){
						cout << "nose ok " << endl ;
						extractor->compute(input, keypoints_nose,descriptorNose);
						for(int x=0;x<nb_celebrities;x++){
							prediction[x] += nose_classifiers[x].predict(selectCols(goodCols[3],descriptorNose)* reducer_nose,true) ;
						}
					}
					if(keypoints_eyes.size() + keypoints_mouth.size() + keypoints_nose.size() != 0){
						nb_images ++ ;
						float min = 2  ;
						int pred =0 ;
						for(int x=0;x<nb_celebrities;x++){
							if (prediction[x] < min){
								pred = x ;
								min = prediction[x] ;
							}
							cout << prediction[x] << " " ;
						}
						cout << endl ;
						cout << "Classe retenue : " << pred << " = " << celebrities[pred] << endl ;
						if(celebrityName.compare(celebrities[pred])){
							cout << "Erreur de classification" << endl ;
							nb_error ++ ;
						}
					}
					else{
						cout << "No keypoints found" << endl ;
					}
					cout << endl ;
				}
			}
			results[k].insert(pair<string,pair<int,int>>(celebrityName,pair<int,int>(nb_error,nb_images)));
		}
	}
	

	cout << "Resultats : " << endl ;
	
	for (int k=0;k<nb_celebrities;k++){
		cout << "- " << celebrities[k] << " : " << endl ;
		cout << "    unlabeled : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		cout << "    labeled : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl << endl ;
	}
}