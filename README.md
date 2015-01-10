INF552 Project

The goal of the project is to recognize faces on an image. To do this, we train classifiers on a dataset (labeled images) and examine their results on a test set (“unlabeled”).

In this project, there are multiple challenges : 

- extract features from the image : face, eyes, mouth, nose
- represent features in a vector space
- create classifiers


Functions to run only in debug mode :

- classifyAndPredict (still fast)
- classifyAndPredictSingleDescriptor (still fast)
- createBowClassifier (very slow)
- computeBowTestDesciptors


Directory Structure

- data  
  The images on which to train and test the program
- stats
  Some statistics on the position and size of eyes, nose, and mouth in a face