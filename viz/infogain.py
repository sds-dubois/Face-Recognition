# Feature selection with the Information Gain measure

from numpy import *
from math import log

# x: features, y: classes
def infogain(x, y):
    info_gains = zeros(x.shape[1]) # features of x
    
    # calculate entropy of the data *hy*
    # with regards to class y
    cl = unique(y)
    hy = 0
    for i in range(len(cl)):
        c = cl[i]
        py = float(sum(y==c))/len(y) # probability of the class c in the data
        hy = hy+py*log(py,2)
    
    hy = -hy
    # compute IG for each feature (columns)
    for col in range(x.shape[1]): # features are on the columns
        values = unique(x[:,col]) # the distinct values of each feature
        # calculate conditional entropy *hyx = H(Y|X)*
        hyx = 0
        for i in range(len(values)): # for all values of the feature
            f = values[i] # value of the specific feature
            yf = y[where(x[:,col]==f)] # array with the the data points index where feature i = f
            # calculate h for classes given feature f
            yclasses = unique(yf) # number of classes
            # hyx = 0; # conditional class probability initialization
            for j in range(len(yclasses)):
                yc = yclasses[j]
                pyf = float(sum(yf==yc))/len(yf) # probability calls condition on the feature value
                hyx = hyx+pyf*log(pyf,2) # conditional entropy
                
        hyx = -hyx
        # Information gain
        info_gains[col] = hy - hyx
        
    return info_gains
    
