#include "featureDetection.h"
#include "getSiftKeypoints.h"
#include "faceDetection.h"
#include "bowMethod.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "boost/filesystem.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

#define selectFeatures  false 
#define pca false
#define nb_celebrities 3

void clusteringClassifyAndPredict(int i,map<int,string> names ,String db ,bool completeDetection, bool cross_valid){
	
	String dir_allFeatures_training = "../allFeatures/" + db + "/training";
	String dir_allFeatures_test = "../allFeatures/" + db + "/test" ;


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

void buildBowDictionary(int i,bool verbose,string db){
     CascadeClassifier face_classifier = getFaceCascadeClassifier();
 	initModule_nonfree() ;
 	//to store the input file names
 	string filename ;
 	//to store the current input image
 	Mat input;
 
 	//To store the keypoints that will be extracted by SIFT
 	vector<KeyPoint> keypoints;
 	//To store the SIFT descriptor of current image
 	Mat descriptor;
 	//To store all the descriptors that are extracted from all the images.
 	Mat featuresUnclustered;
 	//The SIFT feature extractor and descriptor
 	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
 	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
 	Mat img_with_sift; 
 
 	//Images to extract feature descriptors and build the vocabulary
 	for (directory_iterator it1("../data/" + db + "/labeled"); it1 != directory_iterator() ; it1++){
 		path p = it1->path() ;
 		cout << "Folder " << p.string() << endl ;
 		for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
 			cout << it2->path() << endl ;
 			path p2 = it2->path() ;
 			if(is_regular_file(it2->status())){
                 // Loading file
 				filename = p2.string() ;
 				input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
 				if(verbose){
 					imshow("img",input) ;
 					waitKey() ;
 				}
                 // Generating mask for face on the image
                 vector<Rect> faces = detectFaces(face_classifier, input); 
 				if(faces.size() != 0){
					Rect searchZone = faces.front() ;
					if(faces.size() > 1){
						searchZone = selectBestFace(input, faces);
					}
 					Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
 					mask(searchZone) = 1; 
 					//compute the descriptors for each keypoint and put it in a single Mat object
 					detector->detect(input, keypoints,mask);
 					if(verbose){
 						drawKeypoints(input,keypoints,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DEFAULT );
 						imshow("Keypoints",img_with_sift) ;
 						waitKey() ;
 					}
 					extractor->compute(input, keypoints,descriptor);
 					featuresUnclustered.push_back(descriptor);
 				}
 				else
 					cout << "Aucun visage detecte" << endl ;
 			}
 		}
 	}
 
 
 	cout << "features Unclustered " << featuresUnclustered.size() << endl ;
 
 	//Construct BOWKMeansTrainer
 	//the number of bags
 	int dictionarySize=i;
 	//define Term Criteria
 	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
 	//retries number
 	int retries=1;
 	//necessary flags
 	int flags=KMEANS_PP_CENTERS;
 	//Create the BoW (or BoF) trainer
 	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
 	//cluster the feature vectors
 	Mat dictionary=bowTrainer.cluster(featuresUnclustered) ;
 	cout << "Dico cree" << endl ;
 	//store the vocabulary
 	FileStorage fs(("../dictionary/" + db + "/dictionary.yml"), FileStorage::WRITE);
 	fs << "vocabulary" << dictionary;
 	fs.release();
 
 	cout << " Dictionnaire OK" << endl ;
 	
 }
 
 
int createBowClassifier(string db) {
 	CascadeClassifier face_classifier = getFaceCascadeClassifier();
     //prepare BOW descriptor extractor from the dictionary
     Mat dictionary; 
     FileStorage fs(("../dictionary/"+db+"/dictionary.yml"), FileStorage::READ);
     fs["vocabulary"] >> dictionary;
     fs.release();    
     cout << "dictionary loaded" << endl ;
 
     //create a nearest neighbor matcher
 	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher) ;
 	//The SIFT feature extractor and descriptor
 	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT") ; 
 	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT") ; 
 
     //create BoF (or BoW) descriptor extractor
     BOWImgDescriptorExtractor bowDE(extractor,matcher);
     //Set the dictionary with the vocabulary we created in the first step
     bowDE.setVocabulary(dictionary);
 
     //init
 	string filename ;
     Mat input ;
     vector<KeyPoint> keypoints;
     Mat bowDescriptor;
 	map<int,Mat> training_set ;
 	map<int,string> names ;
 	int counter ;
 	int index = 0 ;
 	string celebrityName ;
 
	vector<int> classes ;

	for (directory_iterator it1("../data/"+db+"/labeled"); it1 != directory_iterator() ; it1++){
 		path p = it1->path() ;
 		celebrityName = p.filename().string() ;
 		cout << " -- Traite : " << celebrityName << endl ;
 		Mat samples(0,dictionary.rows,CV_32FC1) ;
 		counter = 0 ;
 		for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
 			path p2 = it2->path() ;
 			if(is_regular_file(it2->status())){
                 // Load the image
 				filename = p2.string();
 				cout << filename << endl;
 				input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
 
 				if(input.size[0] > 0 && input.size[1] > 0){
 					// Generating mask for face on the image
 				    vector<Rect> faces = detectFaces(face_classifier, input); 
 					if(faces.size() != 0){
						Rect searchZone = faces.front() ;
						if(faces.size() > 1){
							searchZone = selectBestFace(input, faces);
						}
 						counter ++ ;
 						Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
 						mask(searchZone) = 1; 
 						//Detect SIFT keypoints (or feature points)
 						detector->detect(input,keypoints,mask);
 						//extract BoW (or BoF) descriptor from given image
 						bowDE.compute(input,keypoints,bowDescriptor);
 						samples.push_back(bowDescriptor) ;
						classes.push_back(index) ;
 					}
 					else 
 						cout << "Aucun visage detecte" << endl ;
 				}
 			}
 		}
 		if (counter > 0 ){
 			training_set.insert(pair<int,Mat>(index,samples)) ;
 			names.insert(pair<int,string>(index,celebrityName)) ;
 			index ++ ;
 		}
 	}
         
 	cout << "Images chargees et analysees" << endl ;
 
	Mat training ;
	for (int x=0;x< index;x++){
		training.push_back(training_set[x]) ;
	}
	FileStorage f(("../allFeatures/" + db + "/bow_training.yml"), FileStorage::WRITE);
 	f << "descriptors" << training;
	f << "classes" << classes ;
 	f.release();

 	CvSVMParams params;
     params.svm_type    = CvSVM::C_SVC;
 	params.kernel_type = CvSVM::POLY;
 	params.degree = 3 ;
 	params.gamma =  5;
 	params.coef0 = 1 ;
     params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
 	
 	Mat labels,temp ;
 	string fname ;
 
 	for (int x=0;x<index;x++){
 		Mat samples(0,dictionary.rows,CV_32FC1) ;
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
 		}
 		else
 			cout << "Le classifieur pour " <<  names[x] << " n'a pas pu etre construit" << endl ;
 
 		fname = "../classifiers/bow" + db +"/" + names[x] + ".yml";
 		cout << "Store : " << fname << endl ;
 		classifier.save(fname.c_str()) ;
 		cout << "Stored" << endl ;
 	}
 	
 	
 	cout << "Classifieurs crees" << endl ;
 
 	return index ;
 	
 }

void computeBowTestDesciptors(string db) {
 	CascadeClassifier face_classifier = getFaceCascadeClassifier();
     //prepare BOW descriptor extractor from the dictionary
     Mat dictionary; 
     FileStorage fs(("../dictionary/"+db+"/dictionary.yml"), FileStorage::READ);
     fs["vocabulary"] >> dictionary;
     fs.release();    
     cout << "dictionary loaded" << endl ;
 
     //create a nearest neighbor matcher
 	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher) ;
 	//The SIFT feature extractor and descriptor
 	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT") ; 
 	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT") ; 
 
     //create BoF (or BoW) descriptor extractor
     BOWImgDescriptorExtractor bowDE(extractor,matcher);
     //Set the dictionary with the vocabulary we created in the first step
     bowDE.setVocabulary(dictionary);
 
     //init
 	string filename ;
     Mat input ;
     vector<KeyPoint> keypoints;
     Mat bowDescriptor;
 	map<int,Mat> training_set ;
 	map<int,string> names ;
 	int counter ;
 	int index = 0 ;
 	string celebrityName ;
 
	vector<int> classes ;

	for (directory_iterator it1("../data/" + db + "/unlabeled"); it1 != directory_iterator() ; it1++){
 		path p = it1->path() ;
 		celebrityName = p.filename().string() ;
 		cout << " -- Traite : " << celebrityName << endl ;
 		Mat samples(0,dictionary.rows,CV_32FC1) ;
 		counter = 0 ;
 		for(directory_iterator it2(p); it2 != directory_iterator() ; it2 ++){
 			path p2 = it2->path() ;
 			if(is_regular_file(it2->status())){
                 // Load the image
 				filename = p2.string();
 				cout << filename << endl;
 				input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
 
 				if(input.size[0] > 0 && input.size[1] > 0){
 					// Generating mask for face on the image
 				    vector<Rect> faces = detectFaces(face_classifier, input); 
 					if(faces.size() != 0){
						Rect searchZone = faces.front() ;
						if(faces.size() > 1){
							searchZone = selectBestFace(input, faces);
						}
 						counter ++ ;
 						Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
 						mask(searchZone) = 1; 
 						//Detect SIFT keypoints (or feature points)
 						detector->detect(input,keypoints,mask);
 						//extract BoW (or BoF) descriptor from given image
 						bowDE.compute(input,keypoints,bowDescriptor);
 						samples.push_back(bowDescriptor) ;
						classes.push_back(index) ;
 					}
 					else 
 						cout << "Aucun visage detecte" << endl ;
 				}
 			}
 		}
 		if (counter > 0 ){
 			training_set.insert(pair<int,Mat>(index,samples)) ;
 			names.insert(pair<int,string>(index,celebrityName)) ;
 			index ++ ;
 		}
 	}
         
 	cout << "Images chargees et analysees" << endl ;
 
	Mat training ;
	for (int x=0;x< index;x++){
		training.push_back(training_set[x]) ;
	}
	FileStorage f(("../allFeatures/" + db + "/bow_test.yml"), FileStorage::WRITE);
 	f << "descriptors" << training;
	f << "classes" << classes ;
 	f.release();
 	
 }


void bowPredict(string db){

	Mat test_descriptor,training_descriptor;
	vector<int> test_classes,training_classes;

	FileStorage ftest(("../allFeatures/" + db + "/bow_test.yml"), FileStorage::WRITE);
	ftest["descriptors"] >> test_descriptor;
	ftest["classes"] >> test_classes ;
 	ftest.release();

	FileStorage ftraining(("../allFeatures/" + db + "/bow_training.yml"), FileStorage::WRITE);
	ftraining["descriptors"] >> training_descriptor;
	ftraining["classes"] >> training_classes ;
 	ftraining.release();

	CvSVM classifiers[nb_celebrities] ;
	String celebrities[nb_celebrities] ;
 	int index = 0 ;
 	for (directory_iterator it("../classifiers/bow/"+db); it != directory_iterator() ; it++) { 
 		path p = it->path() ;
 		if(is_regular_file(it->status())){
 			classifiers[index].load(p.string().c_str()) ;
 			celebrities[index] = p.stem().string() ;
 			cout << "Added " << p.string() << " = " << p.stem().string() << endl ;
 			index ++ ;
 		}
 	}
 	cout << "Classifieurs charges" << endl ;

	string celebrityName ;
	map<string,pair<int,int> > results[2] ;

	for(int k =0; k<2;k++){ 
		int nb_images[nb_celebrities] ;
		int nb_error[nb_celebrities] ;
		for(int x=0; x < nb_celebrities; x++){
			nb_error[x] = 0;
			nb_images[x] = 0;
		}
		Mat data ;
		vector<int> classes ;
		if(k==0){
			data = test_descriptor ;
			classes = test_classes ;
		}
		else{
			data = training_descriptor ;
			classes = training_classes ;
		}
		for(int pic_counter =0 ; pic_counter < data.rows ; pic_counter++){
			int classe = classes[pic_counter] ;
			celebrityName = celebrities[classe] ;
			Mat descriptor = data.row(pic_counter).clone() ;
			float prediction[nb_celebrities] ;
			for(int x=0; x < nb_celebrities; x++){
				prediction[x] = classifiers[x].predict(descriptor,true) ;
			}
			
			float min = prediction[0]  ;
			int pred =0 ;
			for(int x=0;x<nb_celebrities;x++){
				if (prediction[x] < min){
					pred = x ;
					min = prediction[x] ;
				}
				cout << prediction[x] << " " ;
			}
			cout << endl ;
			cout << " Classe retenue : " << pred << " = " << celebrities[pred] << endl ;
			
			if(celebrityName.compare(celebrities[pred])){
				cout << "Erreur de classification" << endl ;
				nb_error[classe] ++ ;
			}

			nb_images[classe] ++ ;

		}
		for(int x = 0 ; x < nb_celebrities ; x ++){
			results[k].insert(pair<string,pair<int,int> >(celebrities[x],pair<int,int>(nb_error[x],nb_images[x])));
		}
	}

	cout << "Resultats : " << endl ;

	for (int k=0;k<nb_celebrities;k++){
		cout << "- " << celebrities[k]  <<  " : " << endl ;
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