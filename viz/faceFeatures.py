import csv
from numpy import *
from numpy import genfromtxt
from chiSQ import chiSQ
from infogain import infogain

f = open('../allFeatures/cropped/rankedFeatures.csv', 'wb')
my_data = genfromtxt('../allFeatures/cropped/face_descriptors.csv', delimiter=',')
print(my_data.shape[1]-2)
data = my_data[:,1:(my_data.shape[1]-1)]
label= my_data[:,0]

# For each feature we get its feature selection value (x^2 or IG)
gainIG = infogain(data,label)
gainChi = chiSQ(data,label)
index1 = argsort(gainIG)[::-1]
index2 = argsort(gainChi)[::-1]

# Store
f = csv.writer(f,delimiter=',')
f.writerow(index1)
f.writerow(index2)