import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import *
import csv

def frobenius(Y,X): # X is the original and Y the reduced
    #returns the percentage of the frobenius norm compared to the original
    fro = linalg.norm(X-Y,ord='fro')/linalg.norm(X, ord='fro')
    return fro
	
def evalMatrix(feature_name):
	dist = zeros((128,1))
	my_data = genfromtxt(feature_name, delimiter=',')
	data = my_data[:,1:129]
	print(feature_name + str(data.shape))
	target= my_data[:,0] # Class of each instance
	M = mean(data,0) # compute the mean
	C = data - M # subtract the mean (along columns)
	W = dot(transpose(C),C) # compute covariance matrix
	eigval,eigvec = linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix
	idx = eigval.argsort()[::-1] # Sort eigenvalues
	eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
	for k in range(1,129):
		Uk = real(eigvec[:,:k:])
		newDatak = dot(C,Uk) # Project the data to the new space (dim = k)
		newDatak = dot(newDatak,transpose(Uk))
		newDatak = newDatak + M
		dist[k-1] = frobenius(newDatak,data)
	return dist

dist_leye = evalMatrix('leye_features.csv')
dist_reye = evalMatrix('reye_features.csv')
dist_mouth = evalMatrix('mouth_features.csv')
dist_nose = evalMatrix('nose_features.csv')

#print(dist_leye)
abs = zeros((128,1))
for k in range(1,129):
	abs[k-1] = k
plt.plot(abs,dist_leye[:,0],color='r',linewidth=1.0,label='leye')
plt.plot(abs,dist_reye[:,0],color='b',linewidth=1.0,label='reye')
plt.plot(abs,dist_mouth[:,0],color='g',linewidth=1.0,label='mouth')
plt.plot(abs,dist_nose[:,0],color='m',linewidth=1.0,label='nose')
plt.xlabel('nbr eigenvectors')
plt.ylabel('Frobenius norm between reduced and original features')
plt.legend()
plt.show()        

