#include "featureDetection.h"
#include "getSiftKeypoints.h"
#include "faceDetection.h"
#include "bowMethod.h"
#include "common.h"

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

void bowClassifyAndPredict(int i,map<int,string> names ,String db ,bool completeDetection, bool cross_valid){
	
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
		Mat leye_samples(0,i,CV_32FC1) ;
		Mat reye_samples(0,i,CV_32FC1) ;
		Mat nose_samples(0,i,CV_32FC1) ;
		Mat mouth_samples(0,i,CV_32FC1) ;
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
