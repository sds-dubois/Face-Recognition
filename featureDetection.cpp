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
const bool pca=true;

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
		mouthZone.x += searchZone.x ;
		mouthZone.y += searchZone.y ;
		Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
		mask(mouthZone) = 1;
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
		nose.x += searchZone.x ;
		nose.y += searchZone.y ;
		Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
		mask(nose) = 1;
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

vector<KeyPoint> getSiftOnEyes2(Mat input,CascadeClassifier eyes_classifier,Ptr<FeatureDetector> detector, float& alpha,bool verbose){
	Mat img_with_sift ;
	vector<KeyPoint> keypoints_best ;
    // Generating mask for face on the image
    vector<Rect> eyes = detectEye(eyes_classifier, input); 
	if(eyes.size() >= 2){
		if(eyes.size() >2)
			cout << "Attention : plus de deux yeux trouvees" << endl;
		Mat mask = Mat::zeros(input.size[0], input.size[1], CV_8U); 
		for (int k=0;k<2;k++){
			mask(eyes[k]) = 1;
			if(verbose){
				rectangle(img_with_sift,eyes[k],Scalar(0,255,0),1,8,0) ;
				cout << eyes[k].size() << endl ;
			}
		}
		Point_<float> c1 = Point_<float>(eyes[0].x+0.5*eyes[0].size().width,eyes[0].y+0.5*eyes[0].size().height);
		Point_<float> c2 = Point_<float>(eyes[1].x+0.5*eyes[1].size().width,eyes[1].y+0.5*eyes[1].size().height);
		alpha = (atan((c1.y-c2.y)/(c1.x-c2.x)))*180/3 ;
		if(verbose)
			cout << "Alpha = " << alpha << endl ;
		keypoints_best.push_back(KeyPoint(c1.x,c1.y,0.5*(eyes[0].size().width+eyes[0].size().height),alpha));
		keypoints_best.push_back(KeyPoint(c2.x,c2.y,0.5*(eyes[1].size().width+eyes[1].size().height),alpha));		
		if(verbose){
			drawKeypoints(input,keypoints_best,img_with_sift,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			for (int k=0;k<2;k++){
				rectangle(img_with_sift,eyes[k],Scalar(0,255,0),1,8,0) ;
			}
			imshow("Keypoints",img_with_sift) ;
			waitKey() ;
		}
	}
	else
		cout << "Error in eyes detection" << endl ;

	return keypoints_best ;
}

void buildSiftDictionary(int i,bool verbose){
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
	Mat img_with_sift;

	//Images to extract feature descriptors and build the vocabulary
	int classPolitician=1;
	for (directory_iterator it1("../data/labeled"); it1 != directory_iterator() ; it1++){
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
				vector<KeyPoint> keypoints_eyes = getSiftOnEyes2(input,eyes_classifier,detector,alpha,verbose);
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
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
	FileStorage fs1("../data/eye_dictionary.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << dictionaryEyes;
	fs1.release();
	eyesFeaturesUnclustered.push_back(dictionaryEyes) ;
	for(int k=0;k<dictionarySize;k++){
		classesUnclustered_eyes.push_back(0) ;
	}
	
	BOWKMeansTrainer bowTrainerMouth(dictionarySize,tc,retries,flags);
	Mat dictionaryMouth=bowTrainerMouth.cluster(mouthFeaturesUnclustered) ;
	FileStorage fs2("../data/mouth_dictionary.yml", FileStorage::WRITE);
	fs2 << "vocabulary" << dictionaryMouth;
	fs2.release();
	mouthFeaturesUnclustered.push_back(dictionaryMouth) ;
	for(int k=0;k<dictionarySize;k++){
		classesUnclustered_mouth.push_back(0) ;
	}

	BOWKMeansTrainer bowTrainerNose(dictionarySize,tc,retries,flags);
	Mat dictionaryNose=bowTrainerNose.cluster(noseFeaturesUnclustered) ;
	FileStorage fs3("../data/nose_dictionary.yml", FileStorage::WRITE);
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

vector<Mat> buildPCAreducer(int nb_coponents,bool verbose){
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
	Mat img_with_sift;

	map<int,Mat> eyes_training_set,mouth_training_set,nose_training_set ;
	map<int,string> names ;
	int counter ;
	int index = 0 ;
	string celebrityName ;
	vector<int> classes ;


	//Images to extract feature descriptors and build the vocabulary
	int classPolitician=1;
	for (directory_iterator it1("../data/labeled"); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		celebrityName = p.filename().string() ;
		cout << " -- Traite : " << celebrityName << endl ;
		Mat eyes_samples = Mat(0,128,CV_32FC1);
		Mat mouth_samples = Mat(0,128,CV_32FC1);
		Mat nose_samples = Mat(0,128,CV_32FC1);		
		counter = 0 ;
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
				vector<KeyPoint> keypoints_eyes = getSiftOnEyes2(input,eyes_classifier,detector,alpha,verbose);
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
					Rect searchMouthZone = faces[0] ;
					searchMouthZone.height /= 2 ;
					searchMouthZone.y += searchMouthZone.height ;
					keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,verbose);
					keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,verbose) ; 
				}
				else{
					cout << "Attention : pas de visage detecte" << endl ;
				}
				Mat avg_descriptorEyes = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_eyes.size() != 0){
					cout << "eyes ok" << endl ;
                    extractor->compute(input, keypoints_eyes,descriptorEyes);
					eyesFeaturesUnclustered.push_back(descriptorEyes);
					classesUnclustered_eyes.push_back(classPolitician);
					for (int k=0;k<128;k++){
						avg_descriptorEyes.at<float>(0,k) = (descriptorEyes.at<float>(0,k) + descriptorEyes.at<float>(1,k))/2 ;
					}
				}
				if(keypoints_mouth.size() != 0){
					cout << "mouth ok" << endl ;
					extractor->compute(input, keypoints_mouth,descriptorMouth);
					mouthFeaturesUnclustered.push_back(descriptorMouth);
				}
				else
					descriptorMouth = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_nose.size() != 0){
					cout << "nose ok " << endl ;
					extractor->compute(input, keypoints_nose,descriptorNose);
					noseFeaturesUnclustered.push_back(descriptorNose);
				}
				else
					descriptorNose = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_eyes.size() + keypoints_mouth.size() + keypoints_nose.size() != 0){
					counter ++ ;
					classes.push_back(index);
					eyes_samples.push_back(avg_descriptorEyes);
					mouth_samples.push_back(descriptorMouth);
					nose_samples.push_back(descriptorNose);
					classesUnclustered_eyes.push_back(classPolitician);
					classesUnclustered_mouth.push_back(classPolitician);
					classesUnclustered_nose.push_back(classPolitician);
				}
			}

		}
		if (counter > 0 ){
			eyes_training_set.insert(pair<int,Mat>(index,eyes_samples)) ;
			mouth_training_set.insert(pair<int,Mat>(index,mouth_samples)) ;
			nose_training_set.insert(pair<int,Mat>(index,nose_samples)) ;
			names.insert(pair<int,string>(index,celebrityName)) ;
			index ++ ;
		}
		classPolitician++;
	}

	cout << "features extracted" << endl ;

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

	Mat eyes_reducer = computePCA(eyesFeaturesUnclustered,nb_coponents);
	Mat mouth_reducer = computePCA(mouthFeaturesUnclustered,nb_coponents);
	Mat nose_reducer = computePCA(noseFeaturesUnclustered,nb_coponents);
	cout << "Size reducers " << eyes_reducer.size() << " " << mouth_reducer.size() << " " << nose_reducer.size() << endl ;
	
	map<int,Mat> training_set ;
	for(int k=0;k<index;k++){
		Mat samples(1,3*nb_coponents,CV_32FC1);
		//cout << "size : " << eyes_training_set[k].size() << endl ;
		//cout << "size mouth : " << mouth_training_set[k].size() << " " << nose_training_set[k].size() << endl ;
		Mat reduced_eyes = eyes_training_set[k] * eyes_reducer ;
		Mat reduced_mouth = mouth_training_set[k] * mouth_reducer ;
		Mat reduced_nose = nose_training_set[k] * nose_reducer ;
		/*cout << "eyes : " << endl ;
		cout << reduced_eyes << endl << endl ;
		cout << "mouth : " << endl ;
		cout << reduced_mouth << endl << endl ;
		cout << "nose : " << endl ;
		cout << reduced_nose << endl << endl ;
		*/
		vector<Mat> matrices ;
		matrices.push_back(reduced_eyes) ;
		matrices.push_back(reduced_mouth) ;
		matrices.push_back(reduced_nose) ;
		hconcat( matrices,samples);
		training_set.insert(pair<int,Mat>(k,samples)) ;
		cout << samples << endl << endl ;
	}

	CvSVMParams params = chooseSVMParams() ;
	vector<CvParamGrid> grids = chooseSVMGrids() ;
	int k_fold = 2 ;

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
			classifier.train(samples_32f,labels,Mat(),Mat(),params);
			//classifier.train_auto(samples_32f,labels,Mat(),Mat(),params,k_fold,grids[0],grids[1],grids[2],grids[3],grids[4],grids[5],false);		
		}
		else
			cout << "Le classifieur pour " <<  names[x] << " n'a pas pu etre construit" << endl ;

		fname = "../classifiers/" + names[x] + ".yml";
		cout << "Store : " << fname << endl ;
		classifier.save(fname.c_str()) ;
		cout << "Stored" << endl ;
	}
	
	
	cout << "Classifieurs crees" << endl ;
	
	vector<Mat> reducers ;
	reducers.push_back(eyes_reducer) ;
	reducers.push_back(mouth_reducer);
	reducers.push_back(nose_reducer) ;

	return reducers ;
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

int init_bowDE(BOWImgDescriptorExtractor& mouth_bowDE,BOWImgDescriptorExtractor& eyes_bowDE,BOWImgDescriptorExtractor& nose_bowDE){
	//prepare BOW descriptor extractor from the dictionary
	Mat eyes_dictionary;
	Mat nose_dictionary; 
	Mat mouth_dictionary; 
    FileStorage fs1("../data/mouth_dictionary.yml", FileStorage::READ);
    fs1["vocabulary"] >> mouth_dictionary;
    fs1.release();
	FileStorage fs2("../data/nose_dictionary.yml", FileStorage::READ);
    fs2["vocabulary"] >> nose_dictionary;
    fs2.release();
	FileStorage fs3("../data/eye_dictionary.yml", FileStorage::READ);
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
int createSVMClassifier(void) {
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

	int dim = init_bowDE(mouth_bowDE,eyes_bowDE,nose_bowDE);
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

	for (directory_iterator it1("../data/labeled"); it1 != directory_iterator() ; it1++){
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
				vector<KeyPoint> keypoints_eyes = getSiftOnEyes2(input,eyes_classifier,detector,alpha,false);
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
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
	int k_fold = 2 ;

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

		fname = "../classifiers/" + names[x] + ".yml";
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

void predict(void){
	
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
	for (directory_iterator it("../classifiers"); it != directory_iterator() ; it++) { 
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
    FileStorage fs1("../data/mouth_dictionary.yml", FileStorage::READ);
    fs1["vocabulary"] >> mouth_dictionary;
    fs1.release();
	FileStorage fs2("../data/nose_dictionary.yml", FileStorage::READ);
    fs2["vocabulary"] >> nose_dictionary;
    fs2.release();
	FileStorage fs3("../data/eye_dictionary.yml", FileStorage::READ);
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

	for (directory_iterator it1("../data/unlabeled"); it1 != directory_iterator() ; it1++) { //each folder in ../data
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
				vector<KeyPoint> keypoints_eyes = getSiftOnEyes2(input,eyes_classifier,detector,alpha,false);				
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
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



	for (directory_iterator it1("../data/labeled"); it1 != directory_iterator() ; it1++) { //each folder in ../data
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
				vector<KeyPoint> keypoints_eyes = getSiftOnEyes2(input,eyes_classifier,detector,alpha,false);
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
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

					//TODO : clean that
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


void predictPCA(vector<Mat> reducers){
	
	Mat reducer_eyes = reducers[0] ;
	Mat reducer_mouth = reducers[1] ;
	Mat reducer_nose = reducers[2] ;

	CascadeClassifier face_classifier = getFaceCascadeClassifier();
	CascadeClassifier eyes_classifier = getEyesCascadeClassifier();
    CascadeClassifier mouth_classifier = getMouthCascadeClassifier();
    CascadeClassifier nose_classifier = getNoseCascadeClassifier();
	CvSVM classifiers[3] ;
	String celebrities[3] ;
	int index = 0 ;
	for (directory_iterator it("../classifiers"); it != directory_iterator() ; it++) { 
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
    FileStorage fs1("../data/mouth_dictionary.yml", FileStorage::READ);
    fs1["vocabulary"] >> mouth_dictionary;
    fs1.release();
	FileStorage fs2("../data/nose_dictionary.yml", FileStorage::READ);
    fs2["vocabulary"] >> nose_dictionary;
    fs2.release();
	FileStorage fs3("../data/eye_dictionary.yml", FileStorage::READ);
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
	string celebrityName ;

	for (directory_iterator it1("../data/unlabeled"); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		celebrityName = p.filename().string() ;
		cout << " -- Traite : " << celebrityName << endl ;	
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
				vector<KeyPoint> keypoints_eyes = getSiftOnEyes2(input,eyes_classifier,detector,alpha,false);
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
					Rect searchMouthZone = faces[0] ;
					searchMouthZone.height /= 2 ;
					searchMouthZone.y += searchMouthZone.height ;
					keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,false);
					keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,false) ; 
				}
				else{
					cout << "Attention : pas de visage detecte" << endl ;
				}
				Mat avg_descriptorEyes = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_eyes.size() != 0){
					cout << "eyes ok" << endl ;
					Mat descriptorEyes ;
                    extractor->compute(input, keypoints_eyes,descriptorEyes);
					for (int k=0;k<128;k++){
						avg_descriptorEyes.at<float>(0,k) = (descriptorEyes.at<float>(0,k) + descriptorEyes.at<float>(1,k))/2 ;
					}
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
					//cout << "sizes " << reducer_eyes.size() << " " << avg_descriptorEyes.size() << endl ;
					Mat eyes_samples = avg_descriptorEyes * reducer_eyes;
					Mat mouth_samples = descriptorMouth * reducer_mouth;
					Mat nose_samples = descriptorNose * reducer_nose ;
					vector<Mat> matrices ; 
					matrices.push_back(eyes_samples) ;
					matrices.push_back(mouth_samples) ;
					matrices.push_back(nose_samples) ;
					cout << eyes_samples.size() << endl ;
					Mat full_descriptor = Mat (1,3*eyes_samples.cols,CV_32FC1) ;
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
				}
				else{
					cout << "No keypoints found" << endl ;
				}
				cout << endl ;
			}

		}
	}
	
	for (directory_iterator it1("../data/labeled"); it1 != directory_iterator() ; it1++){
		path p = it1->path() ;
		celebrityName = p.filename().string() ;
		cout << " -- Traite : " << celebrityName << endl ;	
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
				vector<KeyPoint> keypoints_eyes = getSiftOnEyes2(input,eyes_classifier,detector,alpha,false);
				if(faces.size() >= 1){
					if(faces.size() > 1)
						cout << "Attention : plus d'un visage detecte" << endl ;
					searchZone = faces[0] ;
					Rect searchMouthZone = faces[0] ;
					searchMouthZone.height /= 2 ;
					searchMouthZone.y += searchMouthZone.height ;
					keypoints_mouth = getSiftOnMouth(input,searchMouthZone,mouth_classifier,detector,alpha,false);
					keypoints_nose = getSiftOnNose(input,searchZone,nose_classifier,detector,alpha,false) ; 
				}
				else{
					cout << "Attention : pas de visage detecte" << endl ;
				}
				Mat avg_descriptorEyes = Mat::zeros(1,128,CV_32FC1);
				if(keypoints_eyes.size() != 0){
					cout << "eyes ok" << endl ;
					Mat descriptorEyes ;
                    extractor->compute(input, keypoints_eyes,descriptorEyes);
					for (int k=0;k<128;k++){
						avg_descriptorEyes.at<float>(0,k) = (descriptorEyes.at<float>(0,k) + descriptorEyes.at<float>(1,k))/2 ;
					}
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
					//cout << "sizes " << reducer_eyes.size() << " " << avg_descriptorEyes.size() << endl ;
					Mat eyes_samples = avg_descriptorEyes * reducer_eyes;
					Mat mouth_samples = descriptorMouth * reducer_mouth;
					Mat nose_samples = descriptorNose * reducer_nose ;
					vector<Mat> matrices ; 
					matrices.push_back(eyes_samples) ;
					matrices.push_back(mouth_samples) ;
					matrices.push_back(nose_samples) ;
					cout << eyes_samples.size() << endl ;
					Mat full_descriptor = Mat (1,3*eyes_samples.cols,CV_32FC1) ;
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
				}
				else{
					cout << "No keypoints found" << endl ;
				}
				cout << endl ;
			}

		}
	}

}