Face Recognition (INF552 Project)

The goal of the project is to recognize faces on an image. To do this, we train classifiers on a dataset (training images) and examine their results on a test set.

In this project, there are multiple challenges : 

- extract features from the image : face, eyes, mouth, nose
- represent features in a vector space
- create classifiers



C++ code using OpenCv & Boost.

Functions to run only in debug mode :

- classifyAndPredict (still fast)
- classifyAndPredictSingleDescriptor (still fast)
- createBowClassifier (very slow)
- computeBowTestDesciptors


Directory Structure

- allFeatures ..............................The features extracted on images are stored here
- classifiers................................................................The classifiers are stored here
- data/*database*.........................The images on which to train and test the program
- dictionnary...............The dictionnaries for Bag-of-Words methods are stored here
- lib.........................................Some libraries for OpenCv (HaarCascade classifiers)
- stats...Some statistics on the position and size of eyes, nose, and mouth in a face
- viz........................Some Python script to display the features extracted on images