# Feature selection with the Chi^2 measure

from numpy import *

# x: features (data), y: array with the classes
def chiSQ(x, y):
	cl = unique(y) # unique number of classes
	rows = x.shape[0]
	dim = x.shape[1]
	valCHI = zeros(dim) # initialize array (vector) for the chi^2 values
	
     # For each feature compute its importance
	for d in range(dim):
		feature = x[:,d]
		vals = unique(feature)
		total = 0
		for i in range(len(vals)):
			samples_val_i = where(feature==vals[i])[0]
			for j in range(len(cl)):
				ytmp = y[samples_val_i]
				Oij = len(where(ytmp==cl[j])[0])
				samples_cl_j = where(y==cl[j])[0]
				Eij = float(len(samples_val_i)*len(samples_cl_j))/rows
				total = total + pow((Oij-Eij),2)/Eij

		valCHI[d] = total
      
	chisq = valCHI
	
	return chisq
