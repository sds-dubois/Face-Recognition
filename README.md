## Face Recognition (INF552 Project) ##

### This project aims at recognizing faces on pictures, by focusing on computer vision techniques and using simple machine learning tools. ###

C++ code using OpenCv & Boost.
Please read the report for more details.

----------------
The challenges tackled are : 
- extracting features from the image
- representing these features in a vector space
- building and managing classifiers  

Different methods are implemented to extract useful features from the pictures. In particular, one of our successful methods is to detect some key zones in the picture (like the face /eyes/mouth/ nose) and then compute SIFT descriptors on these particular zones.  
Classification is done with multi-class SVM.  

----------------------------
Best methods are implemented in the file *featureDetection.cpp*. One should first run *featureExtraction* to extract features on training and test images (this has already been done for 3 people of the Yale Face DB, see in data/yale_face_db). Then one can build classifiers and see prediction results using different methods (*classifyAndPredict*, *classifyAndPredictSingleDescriptor* or *clusteringClassifyAndPredict*). Many comments are written in header files to explain how to use these functions and what are their differences.

A Bag-of-Words method is also implemented in *BOWmethod.cpp* but the other ones are much better.

--------------------
 
Functions to run only in debug mode :

- classifyAndPredict (still fast)
- classifyAndPredictSingleDescriptor (still fast)
- createBowClassifier (very slow)
- computeBowTestDesciptors


Directory Structure :

- allFeatures ..............................The features extracted on images are stored here
- classifiers................................................................The classifiers are stored here
- data/*database*.........................The images on which to train and test the program
- dictionnary...............The dictionnaries for Bag-of-Words methods are stored here
- lib.........................................Some libraries for OpenCv (HaarCascade classifiers)
- stats...Some statistics on the position and size of eyes, nose, and mouth in a face
- viz........................Some Python script to display the features extracted on images
