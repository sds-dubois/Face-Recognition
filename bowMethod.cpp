#include "constants.h"
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
 		for(directory_iterator it2(p); it2 != directory_iterator(); it2 ++){
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

 	CvSVMParams params = chooseSVMParams() ;

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
 
 		fname = "../classifiers/" + db +"/bow/" + names[x] + ".yml";
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

	FileStorage ftest(("../allFeatures/" + db + "/bow_test.yml"), FileStorage::READ);
	ftest["descriptors"] >> test_descriptor;
	ftest["classes"] >> test_classes ;
 	ftest.release();

	FileStorage ftraining(("../allFeatures/" + db + "/bow_training.yml"), FileStorage::READ);
	ftraining["descriptors"] >> training_descriptor;
	ftraining["classes"] >> training_classes ;
 	ftraining.release();

	CvSVM classifiers[nb_celebrities] ;
	String celebrities[nb_celebrities] ;
 	int index = 0 ;
 	for (directory_iterator it("../classifiers/"+db+"/bow"); it != directory_iterator() ; it++) { 
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