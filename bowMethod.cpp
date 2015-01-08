#include "getSiftKeypoints.h"
#include "faceDetection.h"
#include "featureDetection.h"
#include "bowMethod.h"
#include "tools.h"

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

void predictBOW(String db){

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