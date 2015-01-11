#include "constants.h"
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
					Mat zonesFound= Mat::zeros(1,4,CV_8U) ;
					zonesFound.at<uchar>(0,0) = index ; //classe de l'image traitee
					Rect searchZone ;
					vector<KeyPoint> keypoints_mouth ;
					vector<KeyPoint> keypoints_nose ;
					float alpha =0 ;
					vector<KeyPoint> keypoints_eyes;
					if(faces.size() >= 1){
						searchZone = faces[0] ;
						if(faces.size() > 1 && detectionType >0){
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
		if(detectionType == 2)
			fn = "/all_completed.yml" ;
		else if(detectionType == 1)
			fn = "/all_bestFace.yml" ;
		else
			fn = "/all_simple.yml" ;
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

	Mat leyeFeaturesUnclustered,reyeFeaturesUnclustered,mouthFeaturesUnclustered,noseFeaturesUnclustered,featureDetailsTraining;
	vector<int> classesUnclustered_eye,classesUnclustered_nose,classesUnclustered_mouth ;
	String fn;
	if(detectionType == 2)
		fn = "/all_completed.yml" ;
	else if(detectionType == 1)
		fn = "/all_bestFace.yml" ;
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

	for (directory_iterator it(dir_leye_classifiers); it != directory_iterator() ; it++) {
		path p = it->path() ;
		if(is_regular_file(it->status())){
			string name = p.stem().string() ;
			for(int j=0; j< names.size(); j++){
				if(name.compare(names[j]) == 0)
					leye_classifiers[j].load(p.string().c_str()) ;
			}
			cout << "Added " << p.stem().string() << endl ;
		}
	}

	for (directory_iterator it(dir_reye_classifiers); it != directory_iterator() ; it++) {
		path p = it->path() ;
		if(is_regular_file(it->status())){
			string name = p.stem().string() ;
			for(int j=0; j< names.size(); j++){
				if(name.compare(names[j]) == 0)
					reye_classifiers[j].load(p.string().c_str()) ;
			}
			cout << "Added " << p.stem().string() << endl ;
		}
	}

	for (directory_iterator it(dir_nose_classifiers); it != directory_iterator() ; it++) {
		path p = it->path() ;
		if(is_regular_file(it->status())){
			string name = p.stem().string() ;
			for(int j=0; j< names.size(); j++){
				if(name.compare(names[j]) == 0)
					nose_classifiers[j].load(p.string().c_str()) ;
			}
			cout << "Added " << p.stem().string() << endl ;
		}
	}

	for (directory_iterator it(dir_mouth_classifiers); it != directory_iterator() ; it++) {
		path p = it->path() ;
		if(is_regular_file(it->status())){
			string name = p.stem().string() ;
			for(int j=0; j< names.size(); j++){
				if(name.compare(names[j]) == 0)
					mouth_classifiers[j].load(p.string().c_str()) ;

			}
			cout << "Added " << p.stem().string() << endl ;
		}
	}

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


	cerr << "Resultats : " << endl ;


	for (int k=0;k<nb_celebrities;k++){
		cerr << "- " << names[k]  << " " << names[k] << " : " << endl ;
		cerr << "    unlabeled : " << results[0].at(names[k]).first << " / " << results[0].at(names[k]).second << endl ;
		cerr << "    labeled : " << results[1].at(names[k]).first << " / " << results[1].at(names[k]).second << endl << endl ;
	}

	ofstream fout("../results.yml");
	for (int k=0;k<nb_celebrities;k++){
		fout << names[k] << "_unlabeled" << " : " << results[0].at(names[k]).first << " / " << results[0].at(names[k]).second << endl ;
		fout << names[k] << "_labeled" << " : " << results[1].at(names[k]).first << " / " << results[1].at(names[k]).second << endl ;
	}
	fout.close();

}

void classifyAndPredictSingleDescriptor(int nb_coponents,String db , vector<vector<int> > goodCols,int detectionType, bool cross_valid){

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
	String fn;
	if(detectionType == 2)
		fn = "/all_completed.yml" ;
	else if(detectionType == 1)
		fn = "/all_bestFace.yml" ;
	else
		fn = "/all_simple.yml" ;
	FileStorage f((dir_allFeatures_training+fn), FileStorage::READ);
	f["classes_eye"] >> classesUnclustered_eye;
	f["leye"] >> leyeFeaturesUnclustered;
	f["reye"] >> reyeFeaturesUnclustered;
	f["classes_mouth"] >> classesUnclustered_mouth;
	f["mouth"] >> mouthFeaturesUnclustered;
	f["classes_nose"] >> classesUnclustered_nose;
	f["nose"] >> noseFeaturesUnclustered;
	f["featureDetails"] >> featureDetailsTraining ;
	int nb_people ;
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
	cerr << "Resultats : " << endl ;

	for (int k=0;k<nb_celebrities;k++){
		cerr << "- " << celebrities[k]  << " " << names[k] << " : " << endl ;
		cerr << "    unlabeled : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		cerr << "    labeled : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl << endl ;
	}

	ofstream fout("../results.yml");
	for (int k=0;k<nb_celebrities;k++){
		fout << celebrities[k] << "_unlabeled" << " : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		fout << celebrities[k] << "_labeled" << " : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl ;
	}
	fout.close();

}


void clusteringClassifyAndPredict(int dictionarySize,String db ,int detectionType, bool cross_valid){
	
	String dir_allFeatures_training = "../allFeatures/" + db + "/training";
	String dir_allFeatures_test = "../allFeatures/" + db + "/test" ;


	Mat leyeFeaturesUnclustered,reyeFeaturesUnclustered,mouthFeaturesUnclustered,noseFeaturesUnclustered,featureDetailsTraining;
	vector<int> classesUnclustered_eye,classesUnclustered_nose,classesUnclustered_mouth ;
	String fn;
	if(detectionType == 2)
		fn = "/all_completed.yml" ;
	else if(detectionType == 1)
		fn = "/all_bestFace.yml" ;
	else
		fn = "/all_simple.yml" ;

	FileStorage f((dir_allFeatures_training+fn), FileStorage::READ);
	f["classes_eye"] >> classesUnclustered_eye;
	f["leye"] >> leyeFeaturesUnclustered;
	f["reye"] >> reyeFeaturesUnclustered;
	f["classes_mouth"] >> classesUnclustered_mouth;
	f["mouth"] >> mouthFeaturesUnclustered;
	f["classes_nose"] >> classesUnclustered_nose;
	f["nose"] >> noseFeaturesUnclustered;
	f["featureDetails"] >> featureDetailsTraining ;
	int nb_people ;
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

	//Construct BOWKMeansTrainer
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=3;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;


	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainerLEye(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	bowTrainerLEye.add(leyeFeaturesUnclustered) ;
	Mat leye_dic = bowTrainerLEye.cluster() ;

	BOWKMeansTrainer bowTrainerREye(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	bowTrainerREye.add(reyeFeaturesUnclustered) ;
	Mat reye_dic = bowTrainerREye.cluster() ;

	BOWKMeansTrainer bowTrainerMouth(dictionarySize,tc,retries,flags);
	bowTrainerMouth.add(mouthFeaturesUnclustered) ;
	Mat mouth_dic = bowTrainerMouth.cluster() ;

	BOWKMeansTrainer bowTrainerNose(dictionarySize,tc,retries,flags);
	bowTrainerNose.add(noseFeaturesUnclustered) ;
	Mat nose_dic = bowTrainerNose.cluster() ;

	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher) ;

	cout << " Dictionnaire OK" << endl ;
	if(pca){
		cout << endl;
		cout << "Show PCA for eyes " << endl ;
		showPCA(leyeFeaturesUnclustered,classesUnclustered_eye,"LEyes");
		cout << "Show PCA for eyes " << endl ;
		showPCA(reyeFeaturesUnclustered,classesUnclustered_eye,"REyes");
		cout << "Show PCA for mouth " << endl ;
		showPCA(mouthFeaturesUnclustered,classesUnclustered_mouth,"Mouth");
		cout << "Show PCA for nose " << endl ;
		showPCA(noseFeaturesUnclustered,classesUnclustered_nose,"Nose");
	}


	String dir_leye_classifiers = "../classifiers/" + db + "/leye" ;
	String dir_reye_classifiers = "../classifiers/" + db + "/reye";
	String dir_nose_classifiers = "../classifiers/" + db + "/nose";
	String dir_mouth_classifiers = "../classifiers/" + db + "/mouth";

	CvSVM leye_classifiers[nb_celebrities] ;
	CvSVM reye_classifiers[nb_celebrities] ;
	CvSVM nose_classifiers[nb_celebrities] ;
	CvSVM mouth_classifiers[nb_celebrities] ;
	String celebrities[nb_celebrities] ;

	map<int,Mat> leye_training_set,reye_training_set,mouth_training_set,nose_training_set ;
	
	vector<DMatch> leye_matches,reye_matches,nose_matches,mouth_matches ;
	vector<DMatch> leye_matches_test,reye_matches_test,nose_matches_test,mouth_matches_test ;

	matcher->match(leyeFeaturesUnclustered,leye_dic,leye_matches) ;
	matcher->match(reyeFeaturesUnclustered,reye_dic,reye_matches) ;
	matcher->match(mouthFeaturesUnclustered,mouth_dic,mouth_matches) ;
	matcher->match(noseFeaturesUnclustered,nose_dic,nose_matches) ;

	matcher->match(leyeFeaturesTest,leye_dic,leye_matches_test) ;
	matcher->match(reyeFeaturesTest,reye_dic,reye_matches_test) ;
	matcher->match(mouthFeaturesTest,mouth_dic,mouth_matches_test) ;
	matcher->match(noseFeaturesTest,nose_dic,nose_matches_test) ;


	for(int x=0; x < classesUnclustered_eye.size() ; x++){
		leye_training_set[classesUnclustered_eye[x]].push_back(leye_dic.row(leye_matches[x].trainIdx));
		reye_training_set[classesUnclustered_eye[x]].push_back(reye_dic.row(reye_matches[x].trainIdx));
	}
	for(int x=0; x < classesUnclustered_mouth.size() ; x++){
		mouth_training_set[classesUnclustered_mouth[x]].push_back(mouth_dic.row(mouth_matches[x].trainIdx));
	}
	for(int x=0; x < classesUnclustered_nose.size() ; x++){
		nose_training_set[classesUnclustered_nose[x]].push_back(nose_dic.row(nose_matches[x].trainIdx));
	}

	CvSVMParams params = chooseSVMParams() ;
	vector<CvParamGrid> grids = chooseSVMGrids() ;
	int k_fold = 3 ;

	string fname ;

	for (int x=0;x<nb_celebrities;x++){
		Mat leye_samples(0,128,CV_32FC1) ;
		Mat reye_samples(0,128,CV_32FC1) ;
		Mat nose_samples(0,128,CV_32FC1) ;
		Mat mouth_samples(0,128,CV_32FC1) ;
		int leye_counter = 0 ;
		int reye_counter = 0 ;
		int nose_counter = 0 ;
		int mouth_counter = 0 ;
		for(int y=0;y<nb_celebrities;y++){
			if(y != x){
				leye_samples.push_back(leye_training_set[y]) ;
				leye_counter += leye_training_set[y].rows ;

				reye_samples.push_back(reye_training_set[y]) ;
				reye_counter += reye_training_set[y].rows ;

				nose_samples.push_back(nose_training_set[y]) ;
				nose_counter += nose_training_set[y].rows ;

				mouth_samples.push_back(mouth_training_set[y]) ;
				mouth_counter += mouth_training_set[y].rows ;
			}
		}
		leye_samples.push_back(leye_training_set[x]) ;
		reye_samples.push_back(reye_training_set[x]) ;
		nose_samples.push_back(nose_training_set[x]) ;
		mouth_samples.push_back(mouth_training_set[x]) ;

		Mat leye_labels = Mat::zeros(leye_counter,1,CV_32FC1) ;
		Mat reye_labels = Mat::zeros(reye_counter,1,CV_32FC1) ;
		Mat nose_labels = Mat::zeros(nose_counter,1,CV_32FC1) ;
		Mat mouth_labels = Mat::zeros(mouth_counter,1,CV_32FC1) ;

		Mat temp = Mat::ones(leye_training_set[x].rows,1,CV_32FC1) ;
		leye_labels.push_back(temp);

		temp = Mat::ones(reye_training_set[x].rows,1,CV_32FC1) ;
		reye_labels.push_back(temp);

		temp = Mat::ones(nose_training_set[x].rows,1,CV_32FC1) ;
		nose_labels.push_back(temp);

		temp = Mat::ones(mouth_training_set[x].rows,1,CV_32FC1) ;
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
		vector<DMatch> leye,reye,nose,mouth ;

		if(k==0){
			featureDetails = featureDetailsTest ;
			leyeFeatures = leyeFeaturesTest ;
			reyeFeatures = reyeFeaturesTest ;
			mouthFeatures = mouthFeaturesTest ;
			noseFeatures = noseFeaturesTest ;
			leye = leye_matches_test ;
			reye = reye_matches_test ;
			nose = nose_matches_test ;
			mouth = mouth_matches_test ;
		}
		else{
			featureDetails = featureDetailsTraining  ;
			leyeFeatures = leyeFeaturesUnclustered ;
			reyeFeatures = reyeFeaturesUnclustered ;
			mouthFeatures = mouthFeaturesUnclustered ;
			noseFeatures = noseFeaturesUnclustered ;
			leye = leye_matches;
			reye = reye_matches ;
			nose = nose_matches ;
			mouth = mouth_matches ;
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
					Mat leye_samples = leye_dic.row(leye[eye_counter].trainIdx).clone() ;
					Mat reye_samples = reye_dic.row(reye[eye_counter].trainIdx).clone() ;
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
						prediction[x] += mouth_classifiers[x].predict(mouth_dic.row(mouth[mouth_counter].trainIdx),true) ;
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
						prediction[x] += nose_classifiers[x].predict(nose_dic.row(nose[nose_counter].trainIdx),true) ;
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


	cerr << "Resultats : " << endl ;


	for (int k=0;k<nb_celebrities;k++){
		cerr << "- " << celebrities[k]  << " " << names[k] << " : " << endl ;
		cerr << "    unlabeled : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		cerr << "    labeled : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl << endl ;
	}

	ofstream fout("../results.yml");
	for (int k=0;k<nb_celebrities;k++){
		fout << celebrities[k] << "_unlabeled" << " : " << results[0].at(celebrities[k]).first << " / " << results[0].at(celebrities[k]).second << endl ;
		fout << celebrities[k] << "_labeled" << " : " << results[1].at(celebrities[k]).first << " / " << results[1].at(celebrities[k]).second << endl ;
	}
	fout.close();
}