from numpy import *
from numpy import genfromtxt
import matplotlib.pyplot as plt
from chiSQ import chiSQ
from infogain import infogain
from mpl_toolkits.mplot3d import Axes3D

def kpca(k,data):
	M = mean(data,0) # compute the mean
	C = data - M # subtract the mean (along columns)
	W = dot(transpose(C),C) # compute covariance matrix
	eigval,eigvec = linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix
	idx = eigval.argsort()[::-1] # Sort eigenvalues
	eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues

	Uk = real(eigvec[:,:k:])
	newDatak = dot(C,Uk) # Project the data to the new space (dim = k)
	newDatak = dot(newDatak,transpose(Uk))
	newDatak = newDatak + M
	
	return newDatak

def variance(Y,X): # X is the original and Y the reduced
	# usualy we compare the ratio of the sum of first k eigenvectors to 
	# all the eigenvectors but since this is going to be used 
	# for a comparison of non-pca appraches we are going to compare all the
	# eigen vectors between the compressed and the original 

	Xcentered = X - tile(mean(X,0),(X.shape[0],1)) #center data
	Cov = dot(transpose(Xcentered),Xcentered) # covariance matrix
	eigOrig= linalg.eig(Cov)[0]
	Ycentered = Y - tile(mean(Y,0),(Y.shape[0],1)) #center data
	Cov = dot(transpose(Ycentered),Ycentered) # covariance matrix
	eigComp = linalg.eig(Cov)[0]
	var = sum(eigComp) / sum(eigOrig)
	return var
	
def show3D(data):
	# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
	# Rows of A correspond to observations, columns to variables.
	M = mean(data,0) # compute the mean
	C = data - M # subtract the mean (along columns)
	W = dot(transpose(C),C) # compute covariance matrix
	eigval,eigvec = linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix
	idx = eigval.argsort()[::-1] # Sort eigenvalues
	eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues

	newData3 = dot(C,real(eigvec[:,:3])) # Project the data to the new space (3-D)
	
	return newData3
	
def show(k,path,name):
	my_data = genfromtxt(path, delimiter=',')
	data = my_data[:,1:129]
	dataBis = my_data[:,1:129]
	target= my_data[:,0]
	
	num_feat = 20
	print('Select ' + str(num_feat) + ' best features')

	# For each feature we get its feature selection value (x^2 or IG)
	## TODO: uncommnent chiSQ(X,Y) to compute chi^2 measure
	gainIG = infogain(data,target)
	gainChi = chiSQ(data,target)
	index1 = argsort(gainIG)[::-1]
	index2 = argsort(gainChi)[::-1]
	# Select the top num_feat features
	data = data[:,index2[:num_feat]]
	data = kpca(10,data)
	dataBis= kpca(10,dataBis)
	
	newData3 = show3D(data)
	newDataBis3 = show3D(dataBis)
	
	fig = plt.figure(k)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(newData3[:,0],newData3[:,1], newData3[:,2], c=target)
	ax.set_title("Projection to the top-3 eigenvectors after feature selection " + name)
	plt.draw()  
	
	fig = plt.figure(k+10)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(newDataBis3[:,0],newDataBis3[:,1], newDataBis3[:,2], c=target)
	ax.set_title("Projection to the top-3 eigenvectors " + name)
	plt.draw()  

def display_var(path):
	my_data = genfromtxt(path, delimiter=',')
	data = my_data[:,1:129]
	target= my_data[:,0]
	gainChi = chiSQ(data,target)
	index = argsort(gainChi)[::-1]
	vars1 = zeros((128,1))
	vars2 = zeros((128,1))
	for num_feat in range(1,119):
		newdata1 = data[:,index[:num_feat]]
		vars1[num_feat-1] = variance(newdata1,data)
		newdata2 = data[:,index[:(num_feat+10)]]
		newdata2 = kpca(num_feat,newdata2)
		vars2[num_feat-1] = variance(newdata2,data)
	return vars1,vars2
	
		
	
show(2,'../allFeatures/yale_face_db/leye_features.csv','left eye')
show(3,'../allFeatures/yale_face_db/reye_features.csv','right eye')
show(4,'../allFeatures/yale_face_db/mouth_features.csv','mouth')
show(5,'../allFeatures/yale_face_db/nose_features.csv','nose')

abs = zeros((128,1))
for k in range(1,129):
	abs[k-1] = k
	
var_leye1,var_leye2 = display_var('../allFeatures/yale_face_db/leye_features.csv')
var_reye1,var_reye2 = display_var('../allFeatures/yale_face_db/reye_features.csv')
var_mouth1,var_mouth2 = display_var('../allFeatures/yale_face_db/mouth_features.csv')
var_nose1,var_nose2 = display_var('../allFeatures/yale_face_db/nose_features.csv')

plt.figure(1)
plt.plot(abs,var_leye1[:,0],color='r',linewidth=1.0,label='leye')
plt.plot(abs,var_reye1[:,0],color='b',linewidth=1.0,label='reye')
plt.plot(abs,var_mouth1[:,0],color='g',linewidth=1.0,label='mouth')
plt.plot(abs,var_nose1[:,0],color='y',linewidth=1.0,label='nose')
plt.plot(abs,var_leye2[:,0],'r--',linewidth=1.0,label='leye')
plt.plot(abs,var_reye2[:,0],'b--',linewidth=1.0,label='reye')
plt.plot(abs,var_mouth2[:,0],'g--',linewidth=1.0,label='mouth')
plt.plot(abs,var_nose2[:,0],'y--',linewidth=1.0,label='nose')
plt.xlabel('nbr features')
plt.ylabel('Variance')
plt.legend()
plt.show()