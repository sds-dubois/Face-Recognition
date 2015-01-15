from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import genfromtxt
import csv

def show(k,path,name):
	# Load the data set
	my_data = genfromtxt(path, delimiter=',')
	data = my_data[:,1:129]
	target= my_data[:,0] # Class of each instance (1, 2 or 3)
	print "Size of the data (rows, #attributes) ", data.shape

	# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
	# Rows of A correspond to observations, columns to variables.
	M = mean(data,0) # compute the mean
	C = data - M # subtract the mean (along columns)
	W = dot(transpose(C),C) # compute covariance matrix
	eigval,eigvec = linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix
	idx = eigval.argsort()[::-1] # Sort eigenvalues
	eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues

	newData3 = dot(C,real(eigvec[:,:3])) # Project the data to the new space (3-D)

	# Plot the first three principal components 
	fig = plt.figure(k)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(newData3[:,0],newData3[:,1], newData3[:,2], c=target)
	ax.set_xlabel('1st Principal Component')
	ax.set_ylabel('2nd Principal Component')
	ax.set_zlabel('3rd Principal Component')
	ax.set_title("Projection to the top-3 Principal Components " + name)
	plt.draw()  

show(1,'../allFeatures/yale_face_db/training/leye_features.csv','left eye')
show(3,'../allFeatures/yale_face_db/training/reye_features.csv','right eye')
show(5,'../allFeatures/yale_face_db/training/mouth_features.csv','mouth eye')
show(7,'../allFeatures/yale_face_db/training/nose_features.csv','nose eye')
plt.show()