import numpy as np
from scipy import sparse
import csv
from sklearn import tree, svm
from sklearn.neighbors.nearest_centroid import NearestCentroid

# gender
Y = []

# [height, weight, shoe_size]
X = []

# read CSV file
genderFile = open('gender.csv')
genderReader = csv.reader(genderFile)

# skip header
iter_genderReader = iter(genderReader)
next(iter_genderReader)




# populate lists
for row in iter_genderReader:
   Y.append(row[0])
   X.append(row[1:])




num = X[1][0]
print(type(num))

