#include "featureDetection.h"
#include "getSiftKeypoints.h"
#include "faceDetection.h"
#include "tools.h"

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

#define selectFeatures  false
#define pca false
#define nb_celebrities 3

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
						if(verbose)
							showAllFeatures(input,faces);
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

	Mat leye_reducer = computePCA(selectCols(goodCols[0],leyeFeaturesUnclustered),nb_coponents).first;
	Mat reye_reducer = computePCA(selectCols(goodCols[1],reyeFeaturesUnclustered),nb_coponents).first;
	Mat mouth_reducer = computePCA(selectCols(goodCols[2],mouthFeaturesUnclustered),nb_coponents).first;
	Mat nose_reducer = computePCA(selectCols(goodCols[3],noseFeaturesUnclustered),nb_coponents).first;
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

void featureExtraction(String db , vector<vector<int> > goodCols , bool verbose, int detectionType){

	String dir_labeled_data = "../data/" + db + "/labeled" ;
	String dir_unlabeled_data = "../data/" + db + "/unlabeled" ;
	String dir_allFeatures[2] ;
	dir_allFeatures[0] = "../allFeatures/" + db +"/test" ;
	dir_allFeatures[1] = "../allFeatures/" + db + "/training" ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	initModule_nonfree() ;

	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Mat img_with_sift;

	String dir_data[2] ;
	dir_data[0] = dir_unlabeled_data ;
	dir_data[1] = dir_labeled_data ;

	for(int k=0;k<2;k++){
		Mat featureDetails = Mat(0,4,CV_8U); //ordre : classe / eye / mouth / nose
		//To store the SIFT descriptor of current image
		Mat descriptorLEye;
		Mat descriptorREye;
		Mat descriptorMouth;
		Mat descriptorNose;
		//To store all the descriptors that are extracted from all the images.
		Mat leyeFeaturesUnclustered = Mat(0,128,CV_32FC1);
		Mat reyeFeaturesUnclustered= Mat(0,128,CV_32FC1);
		Mat mouthFeaturesUnclustered= Mat(0,128,CV_32FC1);
		Mat noseFeaturesUnclustered= Mat(0,128,CV_32FC1);
		vector<int> classesUnclustered_eye;
		vector<int> classesUnclustered_mouth;
		vector<int> classesUnclustered_nose;

		map<int,Mat> leye_training_set,reye_training_set,mouth_training_set,nose_training_set ;
		map<int,string> names ;
		int counter ;
		int index = 0 ;
		string celebrityName ;

		int nb_eye =0 ; int nb_mouth = 0 ; int nb_nose = 0 ;
		//Images to extract feature descriptors and build the vocabulary
		for (directory_iterator it1(dir_data[k]); it1 != directory_iterator() ; it1++){
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
					Mat zonesFound(1,4,CV_8U) ;
					zonesFound.at<uchar>(0,0) = index ; //classe de l'image traitee
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
						Rect searchEyeZone = searchZone;
						searchEyeZone.height /= 2 ;
						keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,verbose);
						Rect searchMouthZone = searchZone;
						searchMouthZone.height /= 2 ;
						searchMouthZone.y += searchMouthZone.height ;
						keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,verbose);
						keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,verbose) ;
						if(keypoints_mouth.size() > 0 && keypoints_nose.size() > 0 && alpha == 0){
							Point2f c1 = keypoints_mouth[0].pt;
							Point2f c2 = keypoints_nose[0].pt;
							alpha = (atan((c1.x-c2.x)/(c2.y-c1.y)))*180/3 ;
							keypoints_mouth[0].angle = alpha;
							keypoints_nose[0].angle = alpha;
						}
                        bool result = enhanceDetection(searchZone, keypoints_eyes, keypoints_mouth, keypoints_nose,detectionType);
						if(result){
                            Mat img_with_sift;
                            if(keypoints_eyes.size() > 0)
                                drawKeypoints(input,keypoints_eyes,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                            if(keypoints_nose.size() > 0)
                                drawKeypoints(input,keypoints_nose,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                            if(keypoints_mouth.size() > 0)
                                drawKeypoints(input,keypoints_mouth,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                            rectangle(img_with_sift,searchZone,Scalar(0,255,0),1,8,0) ;
                            imshow("Eyes",img_with_sift) ;
                            waitKey() ;
                        }

					}
					else{
						cout << "Attention : pas de visage detecte" << endl ;
					}
					if(keypoints_eyes.size() != 0){
						zonesFound.at<uchar>(0,1) = 1 ;
						nb_eye ++ ;
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
					if(keypoints_mouth.size() != 0){
						zonesFound.at<uchar>(0,2) = 1 ;
						nb_mouth ++ ;
						cout << "mouth ok" << endl ;
						extractor->compute(input, keypoints_mouth,descriptorMouth);
						mouthFeaturesUnclustered.push_back(descriptorMouth);
						mouth_samples.push_back(descriptorMouth);
						classesUnclustered_mouth.push_back(index);
					}
					if(keypoints_nose.size() != 0){
						zonesFound.at<uchar>(0,3) = 1 ;
						nb_nose ++ ;
						cout << "nose ok " << endl ;
						extractor->compute(input, keypoints_nose,descriptorNose);
						noseFeaturesUnclustered.push_back(descriptorNose);
						nose_samples.push_back(descriptorNose);
						classesUnclustered_nose.push_back(index);
					}
					if(keypoints_eyes.size() + keypoints_mouth.size() + keypoints_nose.size() != 0){
						featureDetails.push_back(zonesFound) ;
						cout << zonesFound << endl ;
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

		int i = 0 ;
		for(int pic_counter =0 ; pic_counter < featureDetails.rows ; pic_counter++){
				if(featureDetails.at<uchar>(pic_counter,1) == 1)
					i++;
		}
		cout << "NBR oeils : " << nb_eye << " " << i << " " << leyeFeaturesUnclustered.rows << " " << reyeFeaturesUnclustered.rows << endl ;
		cout << "NBR bouches : " << nb_mouth << " " << mouthFeaturesUnclustered.rows << endl ;
		cout << "NBR nez : " << nb_nose << " " << noseFeaturesUnclustered.rows << endl ;
		cout << "features extracted" << endl ;

		//Store features matrices
		writeMatToFile(leyeFeaturesUnclustered,classesUnclustered_eye,((dir_allFeatures[k])+"/leye_features.csv")) ;
		writeMatToFile(reyeFeaturesUnclustered,classesUnclustered_eye,((dir_allFeatures[k])+"/reye_features.csv")) ;
		writeMatToFile(mouthFeaturesUnclustered,classesUnclustered_mouth,((dir_allFeatures[k])+"/mouth_features.csv")) ;
		writeMatToFile(noseFeaturesUnclustered,classesUnclustered_nose,((dir_allFeatures[k])+"/nose_features.csv")) ;

		String fn;
		if(detectionType == 1)
			fn = "/all_completed.yml" ;
		else
			fn = "/all.yml" ;
		FileStorage f((dir_allFeatures[k]+fn), FileStorage::WRITE);
		f << "classes_eye" << classesUnclustered_eye;
		f << "leye" << leyeFeaturesUnclustered;
		f << "reye" << reyeFeaturesUnclustered;
		f << "classes_mouth" << classesUnclustered_mouth;
		f << "mouth" << mouthFeaturesUnclustered;
		f << "classes_nose" << classesUnclustered_nose;
		f << "nose" << noseFeaturesUnclustered;
		f << "featureDetails" << featureDetails ;
		for(int i=0;i<index;i++){
			f << ("name" + to_string(i)) << names[i] ;
		}
		f << "nb_people" << index ;
		f.release();

		FileStorage ff((dir_allFeatures[k]+fn), FileStorage::READ);
		Mat testFeatureDetails ;
		ff["featureDetails"] >> testFeatureDetails ;
		ff.release();

		for(int pic_counter =0 ; pic_counter < featureDetails.rows ; pic_counter++){
			for (int y=0;y<4;y++){
				if(featureDetails.at<uchar>(pic_counter,y) != testFeatureDetails.at<uchar>(pic_counter,y)){
					cout << "Erreur de copie a la ligne : " << pic_counter << endl ;
				}
			}
		}

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
	}
}

void initClassification(map<int,string> names ,int nb_coponents,String db , vector<vector<int> > goodCols){

	String dir_leye_classifiers = "../classifiers/" + db + "/leye" ;
	String dir_reye_classifiers = "../classifiers/" + db + "/reye";
	String dir_nose_classifiers = "../classifiers/" + db + "/nose";
	String dir_mouth_classifiers = "../classifiers/" + db + "/mouth";
	String dir_reducers = "../reducers/" + db ;
	String dir_allFeatures = "../allFeatures/" + db ;


	cout << "Reducers loaded" << endl ;
	Mat leyeFeaturesUnclustered,reyeFeaturesUnclustered,mouthFeaturesUnclustered,noseFeaturesUnclustered;
	vector<int> classesUnclustered_eye,classesUnclustered_nose,classesUnclustered_mouth ;
	FileStorage f((dir_allFeatures+"/all.yml"), FileStorage::READ);
	f["classes_eye"] >> classesUnclustered_eye;
	f["leye"] >> leyeFeaturesUnclustered;
	f["reye"] >> reyeFeaturesUnclustered;
	f["classes_mouth"] >> classesUnclustered_mouth;
	f["mouth"] >> mouthFeaturesUnclustered;
	f["classes_nose"] >> classesUnclustered_nose;
	f["nose"] >> noseFeaturesUnclustered;
	f.release();

	pair<Mat,Mat> leye_pca = computePCA(selectCols(goodCols[0],leyeFeaturesUnclustered),nb_coponents);
	Mat leye_reducer = leye_pca.first ;
	Mat leye_mean = leye_pca.second ;
	pair<Mat,Mat> reye_pca = computePCA(selectCols(goodCols[1],reyeFeaturesUnclustered),nb_coponents);
	Mat reye_reducer = reye_pca.first ;
	Mat reye_mean = reye_pca.second ;
	pair<Mat,Mat> mouth_pca = computePCA(selectCols(goodCols[2],mouthFeaturesUnclustered),nb_coponents);
	Mat mouth_reducer = mouth_pca.first ;
	Mat mouth_mean = mouth_pca.second ;
	pair<Mat,Mat> nose_pca = computePCA(selectCols(goodCols[3],noseFeaturesUnclustered),nb_coponents);
	Mat nose_reducer = nose_pca.first ;
	Mat nose_mean = nose_pca.second ;
	map<int,Mat> leye_training_set,reye_training_set,mouth_training_set,nose_training_set ;

	for(int x=0; x < classesUnclustered_eye.size() ; x++){
		leye_training_set[classesUnclustered_eye[x]].push_back(leyeFeaturesUnclustered.row(x));
		reye_training_set[classesUnclustered_eye[x]].push_back(reyeFeaturesUnclustered.row(x));
	}
	for(int x=0; x < classesUnclustered_mouth.size() ; x++){
		mouth_training_set[classesUnclustered_mouth[x]].push_back(mouthFeaturesUnclustered.row(x));
	}
	for(int x=0; x < classesUnclustered_nose.size() ; x++){
		nose_training_set[classesUnclustered_nose[x]].push_back(noseFeaturesUnclustered.row(x));
	}

	map<int,Mat> leye_reduced_set,reye_reduced_set,mouth_reduced_set,nose_reduced_set ;

	for(int k=0;k<nb_celebrities;k++){

		Mat reduced_leye = (selectCols(goodCols[0],leye_training_set[k])- (Mat::ones(leye_training_set[k].rows,1,CV_32FC1)*leye_mean)) * leye_reducer ;
		Mat reduced_reye = (selectCols(goodCols[1],reye_training_set[k])- (Mat::ones(reye_training_set[k].rows,1,CV_32FC1)*reye_mean)) * reye_reducer ;
		Mat reduced_mouth = (selectCols(goodCols[2],mouth_training_set[k])-(Mat::ones(mouth_training_set[k].rows,1,CV_32FC1)*mouth_mean)) * mouth_reducer ;
		Mat reduced_nose = (selectCols(goodCols[3],nose_training_set[k])- (Mat::ones(nose_training_set[k].rows,1,CV_32FC1)*nose_mean)) * nose_reducer ;

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
	int k_fold = 3 ;

	string fname ;

	for (int x=0;x<nb_celebrities;x++){
		Mat leye_samples(0,3*nb_coponents,CV_32FC1) ;
		Mat reye_samples(0,3*nb_coponents,CV_32FC1) ;
		Mat nose_samples(0,3*nb_coponents,CV_32FC1) ;
		Mat mouth_samples(0,3*nb_coponents,CV_32FC1) ;
		int leye_counter = 0 ;
		int reye_counter = 0 ;
		int nose_counter = 0 ;
		int mouth_counter = 0 ;
		for(int y=0;y<nb_celebrities;y++){
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
			/*
			leye_classifier.train(leye_samples_32f,leye_labels,Mat(),Mat(),params);
			reye_classifier.train(reye_samples_32f,reye_labels,Mat(),Mat(),params);
			nose_classifier.train(nose_samples_32f,nose_labels,Mat(),Mat(),params);
			mouth_classifier.train(mouth_samples_32f,mouth_labels,Mat(),Mat(),params);
			*/
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

	FileStorage fss0((dir_reducers +"/leye_mean.yml"), FileStorage::WRITE);
	fss0 << "mean" << leye_mean;
	fss0.release();
	FileStorage fss1((dir_reducers +"/reye_mean.yml"), FileStorage::WRITE);
	fss1 << "mean" << reye_mean;
	fss1.release();
	FileStorage fss2((dir_reducers +"/mouth_mean.yml"), FileStorage::WRITE);
	fss2 << "mean" << mouth_mean;
	fss2.release();
	FileStorage fss3((dir_reducers +"/nose_mean.yml"), FileStorage::WRITE);
	fss3 << "mean" << nose_mean;
	fss3.release();
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
						if(faces.size() > 1){
							searchZone = selectBestFace(input, faces);
							cout << "Attention : plus d'un visage detecte" << endl ;
						}
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

void predictPCA2(String db,vector<vector<int> > goodCols, int detectionType){

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
			cout << "Added " << p.string() << " = " << p.stem().string() <<  endl ;
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

	Mat leye_mean,reye_mean,nose_mean,mouth_mean;
	FileStorage fss0((dir_reducers +"/leye_mean.yml"), FileStorage::READ);
	fss0["mean"] >> leye_mean;
	fss0.release();
	FileStorage fss1((dir_reducers +"/reye_mean.yml"), FileStorage::READ);
	fss1["mean"] >> reye_mean;
	fss1.release();
	FileStorage fss2((dir_reducers +"/mouth_mean.yml"), FileStorage::READ);
	fss2["mean"] >> mouth_mean;
	fss2.release();
	FileStorage fss3((dir_reducers +"/nose_mean.yml"), FileStorage::READ);
	fss3["mean"] >> nose_mean;
	fss3.release();

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
						if(faces.size() > 1){
							searchZone = selectBestFace(input, faces);
							cout << "Attention : plus d'un visage detecte" << endl ;
						}
						searchZone = faces[0] ;
						Rect searchEyeZone = faces[0] ;
						searchEyeZone.height /= 2 ;
						keypoints_eyes = getSiftOnEyes2(input,searchEyeZone,eyes_classifier,detector,alpha,false);
						Rect searchMouthZone = faces[0] ;
						searchMouthZone.height /= 2 ;
						searchMouthZone.y += searchMouthZone.height ;
						keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,false);
						keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,false) ;
                        if(keypoints_mouth.size() > 0 && keypoints_nose.size() > 0 && alpha == 0){
                            Point2f c1 = keypoints_mouth[0].pt;
                            Point2f c2 = keypoints_nose[0].pt;
                            alpha = (atan((c1.x-c2.x)/(c2.y-c1.y)))*180/3 ;
                            keypoints_mouth[0].angle = alpha;
                            keypoints_nose[0].angle = alpha;
                        }
                        enhanceDetection(searchZone, keypoints_eyes, keypoints_mouth, keypoints_nose,detectionType);

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
						Mat leye_samples = (selectCols(goodCols[0],descriptorLEye)-leye_mean) * reducer_leye;
						Mat reye_samples = (selectCols(goodCols[1],descriptorREye)-reye_mean) * reducer_reye;

						for(int x=0;x<nb_celebrities;x++){
							prediction[x] += leye_classifiers[x].predict(leye_samples,true) ;
							prediction[x] += reye_classifiers[x].predict(reye_samples,true) ;
							cout << prediction[x] << " " ;
						}
						cout << endl ;
					}
					Mat descriptorMouth ;
					if(keypoints_mouth.size() != 0){
						cout << "mouth ok" << endl ;
						extractor->compute(input, keypoints_mouth,descriptorMouth);
						for(int x=0;x<nb_celebrities;x++){
							prediction[x] += mouth_classifiers[x].predict((selectCols(goodCols[2],descriptorMouth)-mouth_mean)* reducer_mouth,true) ;
							cout << prediction[x] << " " ;
						}
						cout << endl ;
					}
					Mat descriptorNose ;
					if(keypoints_nose.size() != 0){
						cout << "nose ok " << endl ;
						extractor->compute(input, keypoints_nose,descriptorNose);
						for(int x=0;x<nb_celebrities;x++){
							prediction[x] += nose_classifiers[x].predict((selectCols(goodCols[3],descriptorNose)-nose_mean)* reducer_nose,true) ;
							cout << prediction[x] << " " ;
						}
						cout << endl ;
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


void classifyAndPredict(int nb_coponents, String db, vector<vector<int> > goodCols, int detectionType, bool cross_valid){

	String dir_allFeatures_training = "../allFeatures/" + db + "/training";
	String dir_allFeatures_test = "../allFeatures/" + db + "/test" ;

	String dir_leye_classifiers = "../classifiers/" + db + "/leye" ;
	String dir_reye_classifiers = "../classifiers/" + db + "/reye";
	String dir_nose_classifiers = "../classifiers/" + db + "/nose";
	String dir_mouth_classifiers = "../classifiers/" + db + "/mouth";

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	CvSVM leye_classifiers[nb_celebrities] ;
	CvSVM reye_classifiers[nb_celebrities] ;
	CvSVM nose_classifiers[nb_celebrities] ;
	CvSVM mouth_classifiers[nb_celebrities] ;
	String celebrities[nb_celebrities] ;

	Mat leyeFeaturesUnclustered,reyeFeaturesUnclustered,mouthFeaturesUnclustered,noseFeaturesUnclustered,featureDetailsTraining;
	vector<int> classesUnclustered_eye,classesUnclustered_nose,classesUnclustered_mouth ;
	String fn ;
	if(detectionType == 1)
		fn = "/all_completed.yml" ;
	else
		fn = "/all_simple.yml" ;

	int nb_people ;

	FileStorage f((dir_allFeatures_training+fn), FileStorage::READ);
	f["classes_eye"] >> classesUnclustered_eye;
	f["leye"] >> leyeFeaturesUnclustered;
	f["reye"] >> reyeFeaturesUnclustered;
	f["classes_mouth"] >> classesUnclustered_mouth;
	f["mouth"] >> mouthFeaturesUnclustered;
	f["classes_nose"] >> classesUnclustered_nose;
	f["nose"] >> noseFeaturesUnclustered;
	f["featureDetails"] >> featureDetailsTraining ;
	f["nb_people"] >> nb_people ;

	if( nb_people != nb_celebrities)
		cout << "Error, il n'y a pas le bon nombre de personnes" << endl ;
	map<int,string> names ;
	for (int i=0; i < nb_people ; i++){
		f[("name"+to_string(i))] >> names[i] ;
	}

	f.release();

	Mat leyeFeaturesTest,reyeFeaturesTest,mouthFeaturesTest,noseFeaturesTest,featureDetailsTest;
	vector<int> classesTest_eye,classesTest_nose,classesTest_mouth ;
	FileStorage ff((dir_allFeatures_test+fn), FileStorage::READ);
	ff["classes_eye"] >> classesTest_eye;
	ff["leye"] >> leyeFeaturesTest;
	ff["reye"] >> reyeFeaturesTest;
	ff["classes_mouth"] >> classesTest_mouth;
	ff["mouth"] >> mouthFeaturesTest;
	ff["classes_nose"] >> classesTest_nose;
	ff["nose"] >> noseFeaturesTest;
	ff["featureDetails"] >> featureDetailsTest ;
	ff.release();

	PCA leye_pca,reye_pca,nose_pca,mouth_pca;
	leye_pca(selectCols(goodCols[0],leyeFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
	reye_pca(selectCols(goodCols[1],reyeFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
	nose_pca(selectCols(goodCols[3],noseFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
	mouth_pca(selectCols(goodCols[2],mouthFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);


	map<int,Mat> leye_training_set,reye_training_set,mouth_training_set,nose_training_set ;

	for(int x=0; x < classesUnclustered_eye.size() ; x++){
		leye_training_set[classesUnclustered_eye[x]].push_back(leyeFeaturesUnclustered.row(x));
		reye_training_set[classesUnclustered_eye[x]].push_back(reyeFeaturesUnclustered.row(x));
	}
	for(int x=0; x < classesUnclustered_mouth.size() ; x++){
		mouth_training_set[classesUnclustered_mouth[x]].push_back(mouthFeaturesUnclustered.row(x));
	}
	for(int x=0; x < classesUnclustered_nose.size() ; x++){
		nose_training_set[classesUnclustered_nose[x]].push_back(noseFeaturesUnclustered.row(x));
	}

	map<int,Mat> leye_reduced_set,reye_reduced_set,mouth_reduced_set,nose_reduced_set ;

	for(int k=0;k<nb_celebrities;k++){
		Mat reduced_leye = leye_pca.project(selectCols(goodCols[0],leye_training_set[k]));
		Mat reduced_reye = reye_pca.project(selectCols(goodCols[1],reye_training_set[k]));
		Mat reduced_mouth = mouth_pca.project(selectCols(goodCols[2],mouth_training_set[k])) ;
		Mat reduced_nose = nose_pca.project(selectCols(goodCols[3],nose_training_set[k])) ;

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
	int k_fold = 3 ;

	string fname ;

	for (int x=0;x<nb_celebrities;x++){
		Mat leye_samples(0,nb_celebrities*nb_coponents,CV_32FC1) ;
		Mat reye_samples(0,nb_celebrities*nb_coponents,CV_32FC1) ;
		Mat nose_samples(0,nb_celebrities*nb_coponents,CV_32FC1) ;
		Mat mouth_samples(0,nb_celebrities*nb_coponents,CV_32FC1) ;
		int leye_counter = 0 ;
		int reye_counter = 0 ;
		int nose_counter = 0 ;
		int mouth_counter = 0 ;
		for(int y=0;y<nb_celebrities;y++){
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
			if(!cross_valid){
				leye_classifier.train(leye_samples_32f,leye_labels,Mat(),Mat(),params);
				reye_classifier.train(reye_samples_32f,reye_labels,Mat(),Mat(),params);
				nose_classifier.train(nose_samples_32f,nose_labels,Mat(),Mat(),params);
				mouth_classifier.train(mouth_samples_32f,mouth_labels,Mat(),Mat(),params);
			}
			else{
				leye_classifier.train_auto(leye_samples_32f,leye_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);
				reye_classifier.train_auto(reye_samples_32f,reye_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);
				nose_classifier.train_auto(nose_samples_32f,nose_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);
				mouth_classifier.train_auto(mouth_samples_32f,mouth_labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);
			}

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
			cout << "Added " << p.string() << " = " << p.stem().string() << " " << names[index] << endl ;
			index ++ ;
		}
	}

	if(index != nb_celebrities)
		cout << "Erreur : il y a un nombre différent de classifieurs et de celebrites" << endl ;

	cout << "Classifieurs charges" << endl ;

	string celebrityName ;
	map<string,pair<int,int> > results[2] ;
	/*
	int i = 0 ;
	for(int pic_counter =0 ; pic_counter < featureDetailsTraining.rows ; pic_counter++){
			if(featureDetailsTraining.at<uchar>(pic_counter,1) == 1)
				i++;
	}
	cout << "nbr oeils " << i << endl ;
	*/
	for(int k =0; k<2;k++){
		int eye_counter =0 ; int mouth_counter = 0 ; int nose_counter = 0;
		int nb_images[nb_celebrities] ;
		int nb_error[nb_celebrities] ;
		for(int x=0; x < nb_celebrities; x++){
			nb_error[x] = 0;
			nb_images[x] = 0;
		}
		Mat featureDetails,leyeFeatures,reyeFeatures,mouthFeatures,noseFeatures ;
		if(k==0){
			featureDetails = featureDetailsTest ;
			leyeFeatures = leyeFeaturesTest ;
			reyeFeatures = reyeFeaturesTest ;
			mouthFeatures = mouthFeaturesTest ;
			noseFeatures = noseFeaturesTest ;
		}
		else{
			featureDetails = featureDetailsTraining  ;
			leyeFeatures = leyeFeaturesUnclustered ;
			reyeFeatures = reyeFeaturesUnclustered ;
			mouthFeatures = mouthFeaturesUnclustered ;
			noseFeatures = noseFeaturesUnclustered ;
		}
		for(int pic_counter =0 ; pic_counter < featureDetails.rows ; pic_counter++){
			int classe = featureDetails.at<uchar>(pic_counter,0) ;
			celebrityName = names[classe] ;
			float prediction[nb_celebrities] ;
			for(int x=0; x < nb_celebrities; x++){
				prediction[x] = 0;
			}
			if(featureDetails.at<uchar>(pic_counter,1) == 1){
				if(eye_counter >= leyeFeatures.rows || eye_counter >= reyeFeatures.rows)
					cout << "Attention eye_counter trop grand" << endl ;
				else{
					Mat descriptorLEye, descriptorREye;
					descriptorLEye = leyeFeatures.row(eye_counter).clone() ;
					descriptorREye = reyeFeatures.row(eye_counter).clone() ;
					Mat leye_samples = leye_pca.project(selectCols(goodCols[0],descriptorLEye));
					Mat reye_samples = reye_pca.project(selectCols(goodCols[1],descriptorREye));
					for(int x=0;x<nb_celebrities;x++){
						prediction[x] += leye_classifiers[x].predict(leye_samples,true) ;
						prediction[x] += reye_classifiers[x].predict(reye_samples,true) ;
					}
 				}
				eye_counter ++ ;
			}
			if(featureDetails.at<uchar>(pic_counter,2) == 1){
				if(mouth_counter >= mouthFeatures.rows)
					cout << "Attention mouth_counter trop grand" << endl ;
				else{
					Mat descriptorMouth;
					descriptorMouth = mouthFeatures.row(mouth_counter).clone() ;
					for(int x=0;x<nb_celebrities;x++){
						prediction[x] += mouth_classifiers[x].predict(mouth_pca.project(selectCols(goodCols[2],descriptorMouth)),true) ;
						//cout << prediction[x] << " " ;
					}
 				}
				mouth_counter ++ ;

			}
			if(featureDetails.at<uchar>(pic_counter,3) == 1){
				if(nose_counter >= noseFeatures.rows)
					cout << "Attention nose_counter trop grand" << endl ;
				else{
					Mat descriptorNose;
					descriptorNose = noseFeatures.row(nose_counter).clone() ;
					for(int x=0;x<nb_celebrities;x++){
						prediction[x] += nose_classifiers[x].predict(nose_pca.project(selectCols(goodCols[3],descriptorNose)),true) ;
					}
 				}
				nose_counter ++ ;
			}

			nb_images[classe] ++ ;
			float min = 100  ;
			int pred =0 ;
			for(int x=0;x<nb_celebrities;x++){
				if (prediction[x] < min){
					pred = x ;
					min = prediction[x] ;
				}
				cout << prediction[x] << " " ;
			}
			cout << endl ;
			cout << pic_counter << " " << eye_counter << " " << mouth_counter << " " << nose_counter << " Classe retenue : " << pred << " = " << names[pred] << endl ;
			if(celebrityName.compare(names[pred])){
				cout << "Erreur de classification" << endl ;
				nb_error[classe] ++ ;
			}
		}
		for(int x = 0 ; x < nb_celebrities ; x ++){
			results[k].insert(pair<string,pair<int,int> >(names[x],pair<int,int>(nb_error[x],nb_images[x])));
		}
	}


	cout << "Resultats : " << endl ;


	for (int k=0;k<nb_celebrities;k++){
		cout << "- " << celebrities[k]  << " " << names[k] << " : " << endl ;
		cout << "    unlabeled : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		cout << "    labeled : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl << endl ;
	}

	ofstream fout("../results.yml");
	for (int k=0;k<nb_celebrities;k++){
		fout << celebrities[k] << "_unlabeled" << " : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		fout << celebrities[k] << "_labeled" << " : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl ;
	}
	fout.close();

}

void classifyAndPredictSingleDescriptor(map<int,string> names ,int nb_coponents,String db , vector<vector<int> > goodCols,bool completeDetection, bool cross_valid){

	String dir_allFeatures_training = "../allFeatures/" + db + "/training";
	String dir_allFeatures_test = "../allFeatures/" + db + "/test" ;

	String dir_single_classifier = "../classifiers/" + db + "/single" ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	CvSVM classifiers[nb_celebrities] ;
	String celebrities[nb_celebrities] ;

	Mat leyeFeaturesUnclustered,reyeFeaturesUnclustered,mouthFeaturesUnclustered,noseFeaturesUnclustered,featureDetailsTraining;
	vector<int> classesUnclustered_eye,classesUnclustered_nose,classesUnclustered_mouth ;
	String fn ;
	if(completeDetection)
		fn = "/all_completed.yml" ;
	else
		fn = "/all.yml" ;
	FileStorage f((dir_allFeatures_training+fn), FileStorage::READ);
	f["classes_eye"] >> classesUnclustered_eye;
	f["leye"] >> leyeFeaturesUnclustered;
	f["reye"] >> reyeFeaturesUnclustered;
	f["classes_mouth"] >> classesUnclustered_mouth;
	f["mouth"] >> mouthFeaturesUnclustered;
	f["classes_nose"] >> classesUnclustered_nose;
	f["nose"] >> noseFeaturesUnclustered;
	f["featureDetails"] >> featureDetailsTraining ;
	f.release();

	Mat leyeFeaturesTest,reyeFeaturesTest,mouthFeaturesTest,noseFeaturesTest,featureDetailsTest;
	vector<int> classesTest_eye,classesTest_nose,classesTest_mouth ;
	FileStorage ff((dir_allFeatures_test+fn), FileStorage::READ);
	ff["classes_eye"] >> classesTest_eye;
	ff["leye"] >> leyeFeaturesTest;
	ff["reye"] >> reyeFeaturesTest;
	ff["classes_mouth"] >> classesTest_mouth;
	ff["mouth"] >> mouthFeaturesTest;
	ff["classes_nose"] >> classesTest_nose;
	ff["nose"] >> noseFeaturesTest;
	ff["featureDetails"] >> featureDetailsTest ;
	ff.release();

	PCA leye_pca,reye_pca,nose_pca,mouth_pca;
	leye_pca(selectCols(goodCols[0],leyeFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
	reye_pca(selectCols(goodCols[1],reyeFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
	nose_pca(selectCols(goodCols[3],noseFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);
	mouth_pca(selectCols(goodCols[2],mouthFeaturesUnclustered), Mat(), CV_PCA_DATA_AS_ROW, nb_coponents);

	map<int,Mat> training_set ;
	int eye_counter =0 ; int mouth_counter = 0 ; int nose_counter = 0;

	for(int pic_counter =0 ; pic_counter < featureDetailsTraining.rows ; pic_counter++){
		int classe = featureDetailsTraining.at<uchar>(pic_counter,0) ;
		Mat full_descriptor = Mat::zeros(1,4*nb_coponents,CV_32FC1) ;
		if(featureDetailsTraining.at<uchar>(pic_counter,1) == 1){
			if(eye_counter >= leyeFeaturesUnclustered.rows || eye_counter >= reyeFeaturesUnclustered.rows)
				cout << "Attention eye_counter trop grand" << endl ;
			else{
				Mat descriptorLEye, descriptorREye;
				descriptorLEye = leyeFeaturesUnclustered.row(eye_counter).clone() ;
				descriptorREye = reyeFeaturesUnclustered.row(eye_counter).clone() ;
				Mat leye_samples = leye_pca.project(selectCols(goodCols[0],descriptorLEye));
				Mat reye_samples = reye_pca.project(selectCols(goodCols[1],descriptorREye));
				for (int i=0;i< nb_coponents;i++){
					full_descriptor.at<float>(0,i) = leye_samples.at<float>(0,i) ;
					full_descriptor.at<float>(0,i+nb_coponents) = reye_samples.at<float>(0,i) ;
				}
 			}
			eye_counter ++ ;
		}
		if(featureDetailsTraining.at<uchar>(pic_counter,2) == 1){
			if(mouth_counter >= mouthFeaturesUnclustered.rows)
				cout << "Attention mouth_counter trop grand" << endl ;
			else{
				Mat descriptorMouth = mouthFeaturesUnclustered.row(mouth_counter).clone() ;
				Mat mouth_sample = mouth_pca.project(selectCols(goodCols[2],descriptorMouth)) ;
				for (int i=0;i< nb_coponents;i++){
					full_descriptor.at<float>(0,i+2*nb_coponents) = mouth_sample.at<float>(0,i) ;
				}
 			}
			mouth_counter ++ ;

		}
		if(featureDetailsTraining.at<uchar>(pic_counter,3) == 1){
			if(nose_counter >= noseFeaturesUnclustered.rows)
				cout << "Attention nose_counter trop grand" << endl ;
			else{
				Mat descriptorNose = noseFeaturesUnclustered.row(nose_counter).clone() ;
				Mat nose_sample = mouth_pca.project(selectCols(goodCols[2],descriptorNose)) ;
				for (int i=0;i< nb_coponents;i++){
					full_descriptor.at<float>(0,i+2*nb_coponents) = nose_sample.at<float>(0,i) ;
				}
 			}
			nose_counter ++ ;
		}
		training_set[classe].push_back(full_descriptor);
	}

	CvSVMParams params = chooseSVMParams() ;
	vector<CvParamGrid> grids = chooseSVMGrids() ;
	int k_fold = 3 ;

	string fname ;

	for (int x=0;x<nb_celebrities;x++){
		Mat samples(0,nb_coponents,CV_32FC1) ;
		int counter = 0 ;

		for(int y=0;y<nb_celebrities;y++){
			if(y != x){
				samples.push_back(training_set[y]) ;
				counter += training_set[y].rows ;
			}
		}
		samples.push_back(training_set[x]) ;

		Mat labels = Mat::zeros(counter,1,CV_32FC1) ;
		Mat temp = Mat::ones(training_set[x].rows,1,CV_32FC1) ;
		labels.push_back(temp);

		CvSVM classifier ;
		Mat samples_32f;
		samples.convertTo(samples_32f, CV_32F);

		if(samples.rows != 0){
			if(!cross_valid){
				classifier.train(samples_32f,labels,Mat(),Mat(),params);
			}
			else{
				classifier.train_auto(samples_32f,labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);
			}

		}
		else
			cout << "Le classifieur pour " <<  names[x] << " n'a pas pu etre construit" << endl ;

		fname = dir_single_classifier + "/"+ names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		classifier.save(fname.c_str()) ;
	}

	cout << "Classifieurs crees" << endl ;

	int index = 0 ;
	for (directory_iterator it(dir_single_classifier); it != directory_iterator() ; it++) {
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

	string celebrityName ;
	map<string,pair<int,int> > results[2] ;

	for(int k =0; k<2;k++){
		eye_counter =0 ; mouth_counter = 0 ; nose_counter = 0;
		int nb_images[nb_celebrities] ;
		int nb_error[nb_celebrities] ;
		for(int x=0; x < nb_celebrities; x++){
			nb_error[x] = 0;
			nb_images[x] = 0;
		}
		Mat featureDetails,leyeFeatures,reyeFeatures,mouthFeatures,noseFeatures ;
		if(k==0){
			featureDetails = featureDetailsTest ;
			leyeFeatures = leyeFeaturesTest ;
			reyeFeatures = reyeFeaturesTest ;
			mouthFeatures = mouthFeaturesTest ;
			noseFeatures = noseFeaturesTest ;
		}
		else{
			featureDetails = featureDetailsTraining  ;
			leyeFeatures = leyeFeaturesUnclustered ;
			reyeFeatures = reyeFeaturesUnclustered ;
			mouthFeatures = mouthFeaturesUnclustered ;
			noseFeatures = noseFeaturesUnclustered ;
		}
		for(int pic_counter =0 ; pic_counter < featureDetails.rows ; pic_counter++){
			int classe = featureDetails.at<uchar>(pic_counter,0) ;
			Mat full_descriptor = Mat::zeros(1,4*nb_coponents,CV_32FC1) ;
			celebrityName = names[classe] ;
			float prediction[nb_celebrities] ;
			for(int x=0; x < nb_celebrities; x++){
				prediction[x] = 0;
			}
			if(featureDetails.at<uchar>(pic_counter,1) == 1){
				if(eye_counter >= leyeFeatures.rows || eye_counter >= reyeFeatures.rows)
					cout << "Attention eye_counter trop grand" << endl ;
				else{
					Mat descriptorLEye, descriptorREye;
					descriptorLEye = leyeFeatures.row(eye_counter).clone() ;
					descriptorREye = reyeFeatures.row(eye_counter).clone() ;
					Mat leye_samples = leye_pca.project(selectCols(goodCols[0],descriptorLEye));
					Mat reye_samples = reye_pca.project(selectCols(goodCols[1],descriptorREye));
					for (int i=0;i< nb_coponents;i++){
						full_descriptor.at<float>(0,i) = leye_samples.at<float>(0,i) ;
						full_descriptor.at<float>(0,i+nb_coponents) = reye_samples.at<float>(0,i) ;
					}
 				}
				eye_counter ++ ;
			}
			if(featureDetails.at<uchar>(pic_counter,2) == 1){
				if(mouth_counter >= mouthFeatures.rows)
					cout << "Attention mouth_counter trop grand" << endl ;
				else{
					Mat descriptorMouth = mouthFeatures.row(mouth_counter).clone() ;
					Mat mouth_sample = mouth_pca.project(selectCols(goodCols[2],descriptorMouth)) ;
					for (int i=0;i< nb_coponents;i++){
						full_descriptor.at<float>(0,i+2*nb_coponents) = mouth_sample.at<float>(0,i) ;
					}
 				}
				mouth_counter ++ ;

			}
			if(featureDetails.at<uchar>(pic_counter,3) == 1){
				if(nose_counter >= noseFeatures.rows)
					cout << "Attention nose_counter trop grand" << endl ;
				else{
					Mat descriptorNose = noseFeatures.row(nose_counter).clone() ;
					Mat nose_sample = mouth_pca.project(selectCols(goodCols[2],descriptorNose)) ;
					for (int i=0;i< nb_coponents;i++){
						full_descriptor.at<float>(0,i+2*nb_coponents) = nose_sample.at<float>(0,i) ;
					}
 				}
				nose_counter ++ ;
			}

			for(int x=0;x<nb_celebrities;x++){
				prediction[x] = classifiers[x].predict(full_descriptor,true);
			}
			nb_images[classe] ++ ;
			float min = 100  ;
			int pred =0 ;
			for(int x=0;x<nb_celebrities;x++){
				if (prediction[x] < min){
					pred = x ;
					min = prediction[x] ;
				}
				cout << prediction[x] << " " ;
			}
			cout << endl ;
			cout << pic_counter << " " << eye_counter << " " << mouth_counter << " " << nose_counter << " Classe retenue : " << pred << " = " << names[pred] << endl ;
			if(celebrityName.compare(names[pred])){
				cout << "Erreur de classification" << endl ;
				nb_error[classe] ++ ;
			}
		}
		for(int x = 0 ; x < nb_celebrities ; x ++){
			results[k].insert(pair<string,pair<int,int> >(names[x],pair<int,int>(nb_error[x],nb_images[x])));
		}
	}
	cout << "Resultats : " << endl ;

	for (int k=0;k<nb_celebrities;k++){
		cout << "- " << celebrities[k]  << " " << names[k] << " : " << endl ;
		cout << "    unlabeled : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		cout << "    labeled : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl << endl ;
	}

	ofstream fout("../results.yml");
	for (int k=0;k<nb_celebrities;k++){
		fout << celebrities[k] << "_unlabeled" << " : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		fout << celebrities[k] << "_labeled" << " : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl ;
	}
	fout.close();

}
