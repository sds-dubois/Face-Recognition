#include "featureDetection.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;

void buildSiftDictionary(int i){
	//Step 1 - Obtain the set of bags of features.
	initModule_nonfree() ;
	//to store the input file names
	char * filename = new char[100];       
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


	//Images to extract feature descriptors and build the vocabulary
	//Hollande
	for(int f=0;f<20;f++){        
		//create the file name of an image
		sprintf(filename,"../dictionary/Hollande/%i.jpg",f);
		cout << filename << endl ;

		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale   
		cout << input.cols << " " << input.rows << endl;
		//imshow("I",input);
		//waitKey() ;
		//detect feature points
		detector->detect(input, keypoints);
		cout << keypoints.size() << endl ;

		//compute the descriptors for each keypoint
		extractor->compute(input, keypoints,descriptor); 
		//put the all feature descriptors in a single Mat object 
		featuresUnclustered.push_back(descriptor);        
		//print the percentage
		//cout << f/10 << " percent done\n" << endl ;
		printf("Hollande %i percent done\n",f*10);
	}    

	//Obama
	for(int f=0;f<20;f++){        
		//create the file name of an image
		sprintf(filename,"../dictionary/Obama/%i.jpg",f);
		cout << filename << endl ;

		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale   
		cout << input.cols << " " << input.rows << endl;
		//imshow("I",input);
		//waitKey() ;
		//detect feature points
		detector->detect(input, keypoints);
		cout << keypoints.size() << endl ;

		//compute the descriptors for each keypoint
		extractor->compute(input, keypoints,descriptor); 
		//put the all feature descriptors in a single Mat object 
		featuresUnclustered.push_back(descriptor);        
		//print the percentage
		//cout << f/10 << " percent done\n" << endl ;
		printf("Obama %i percent done\n",f*10);
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
	//Mat feature ;
	//featuresUnclustered.convertTo(feature,CV_32FC1);
	Mat dictionary=bowTrainer.cluster(featuresUnclustered) ;
	cout << "Dico cree" << endl ;
	//store the vocabulary
	FileStorage fs("../dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	cout << " Dictionnaire OK" << endl ;
}

Mat getSiftDescriptor(int i) {

	//Step 2 - Obtain the BoF descriptor for given image/video frame. 

    //prepare BOW descriptor extractor from the dictionary
	//Mat udictionary ;
	//dictionary.convertTo(udictionary,CV_8UC1);
    Mat dictionary; 
    FileStorage fs("../dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();    
    cout << "dictionary loaded" << endl ;

    //create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher) ;
		//= DescriptorMatcher::create("BruteForce");
	//The SIFT feature extractor and descriptor
	Ptr<FeatureDetector> detector2 = FeatureDetector::create("SIFT") ; //("Dense")
	Ptr<DescriptorExtractor> extractor2 = DescriptorExtractor::create("SIFT") ;  //("ORB");

	cout << "init ok" << endl ;
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor2,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);
	cout << "Set voc ok" << endl ;
    //To store the image file name
    char * filename2 = new char[100];
    //To store the image tag name - only for save the descriptor in a file
    char * imageTag = new char[10];
 
    //open the file to write the resultant descriptor
    FileStorage fs1("../descriptor.yml", FileStorage::WRITE);    
    
    //the image file with the location
	sprintf(filename2,"../testimages/%i.jpg",i);        
    //read the image
    Mat img=imread(filename2,CV_LOAD_IMAGE_GRAYSCALE);    
	cout << img.cols << " x " << img.rows << endl ;
	imshow("I2",img);

    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints2;        
    //Detect SIFT keypoints (or feature points)
    detector2->detect(img,keypoints2);
    //To store the BoW (or BoF) representation of the image
    Mat bowDescriptor;        
    //extract BoW (or BoF) descriptor from given image
    bowDE.compute(img,keypoints2,bowDescriptor);
 
    //prepare the yml (some what similar to xml) file
    sprintf(imageTag,"img1");            
    //write the new BoF descriptor to the file
    fs1 << imageTag << bowDescriptor;        
             
    //release the file storage
    fs1.release();

	cout << "C'est fini" << endl ;

	return bowDescriptor ;
}

void createSVMClassifier(int n) {

    //prepare BOW descriptor extractor from the dictionary
    Mat dictionary; 
    FileStorage fs("../dictionary.yml", FileStorage::READ);
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

    //To store the image file name
    char * filename = new char[100];
    //To store the image tag name - only for save the descriptor in a file
    char * imageTag = new char[10];
    Mat input ;
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;  
	//To store the BoW (or BoF) representation of the image
    Mat bowDescriptor;   

	Mat samples(0,dictionary.rows,CV_32FC1);
    Mat labels(0,1,CV_32FC1);
	Mat img_with_sift ;

	for(int f=0;f<20;f++){        //Barack Obama
		//create the file name of an image
		sprintf(filename,"../dictionary/Obama/%i.jpg",f);
		cout << filename << endl ;
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale     
      
		//Detect SIFT keypoints (or feature points)
		detector->detect(input,keypoints);
		//drawKeypoints(input,keypoints,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DEFAULT );
		//imshow("Keypoints",img_with_sift) ;
		//waitKey() ;
		//extract BoW (or BoF) descriptor from given image
		bowDE.compute(input,keypoints,bowDescriptor);
		samples.push_back(bowDescriptor) ;
	}

	for(int f=0;f<20;f++){        //Hollande
		//create the file name of an image
		sprintf(filename,"../dictionary/Hollande/%i.jpg",f);
		cout << filename << endl ;
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale     
      
		//Detect SIFT keypoints (or feature points)
		detector->detect(input,keypoints);
		//drawKeypoints(input,keypoints,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DEFAULT );
		//imshow("Keypoints",img_with_sift) ;
		//waitKey() ;
		//extract BoW (or BoF) descriptor from given image
		bowDE.compute(input,keypoints,bowDescriptor);
		samples.push_back(bowDescriptor) ;
	}

	Mat temp ;
	temp = Mat::ones(20, 1, CV_32FC1) ;
	labels.push_back(temp) ;
	temp = Mat::zeros(20, 1, CV_32FC1) ;
	labels.push_back(temp) ;         
	cout << "Images chargees et analysees" << endl ;
	cout << samples.rows << " " << labels.rows << endl ;

	cout << "Samples : " << samples << endl << endl ;
	cout << "Labels : " << labels << endl << endl ;

	CvSVM classifier;
	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.degree = 3 ;
	params.gamma =  5;
	params.coef0 = 1 ;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	Mat samples_32f ;
	samples.convertTo(samples_32f, CV_32F);
	if(samples.rows != 0){ 
		classifier.train(samples_32f,labels,Mat(),Mat(),params);		
	}
	else
		cout << "Samples n'a qu'une ligne !" << endl ;
	
	cout << "Classifieur cree" << endl ;
	int nbr_error_Obama =0 ;
	int nbr_error_Hollande =0 ;
	//Obama
	for(int k=0;k<25;k++){
		sprintf(filename,"../dictionary/Obama/%i.jpg",k);
		cout << "Test : 1 " << filename << endl ;
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale     
		//Detect SIFT keypoints (or feature points)
		detector->detect(input,keypoints);
		//extract BoW (or BoF) descriptor from given image
		bowDE.compute(input,keypoints,bowDescriptor);

		float response = classifier.predict(bowDescriptor,true) ;
		cout << response << " = " << classifier.predict(bowDescriptor) <<endl ;
	}
	
	//Hollande
	for(int k=0;k<25;k++){
		sprintf(filename,"../dictionary/Hollande/%i.jpg",k);
		cout << "Test : 0 " << filename << endl ;
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale     
		//Detect SIFT keypoints (or feature points)
		detector->detect(input,keypoints);
		//extract BoW (or BoF) descriptor from given image
		bowDE.compute(input,keypoints,bowDescriptor);

		cout <<classifier.predict(bowDescriptor,true) << " = " << classifier.predict(bowDescriptor) <<endl ;

	}

	cout << "Erreurs Obama : " << nbr_error_Obama << ", Erreurs Hollande : " << nbr_error_Hollande << endl ;
}