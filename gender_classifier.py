import numpy as np
from scipy import sparse
import csv
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import accuracy_score
import sys

# read CSV file
genderFile = open('gender.csv')
genderReader = csv.reader(genderFile)

# skip header
iter_genderReader = iter(genderReader)
next(iter_genderReader)

# gender
Y = []

# [height, weight, shoe_size]
X = []

# populate lists
for row in iter_genderReader:
   Y.append(row[0])
   X.append(row[1:])

# close CSV file
genderFile.close()

# covert string values to numbers
X_len = len(X)
for row in range(X_len):
    X[row][0] = float(X[row][0])
    X[row][1] = float(X[row][1])
    X[row][2] = float(X[row][2])

# initialize classifiers
clf_LinearSVC = svm.LinearSVC()
clf_NearestCentroid = NearestCentroid()
clf_SVC = svm.SVC()

# train classifiers using data set
clf_LinearSVC = clf_LinearSVC.fit(X, Y)
clf_NearestCentroid = clf_NearestCentroid.fit(X, Y)
clf_SVC = clf_SVC.fit(X, Y)

# test clasiifiers using data set
test_LinearSVC = clf_LinearSVC.predict(X)
acc_LinearSVC = accuracy_score(Y, test_LinearSVC) * 100.0


test_NearestCentroid = clf_NearestCentroid.predict(X)
acc_NearestCentroid = accuracy_score(Y, test_NearestCentroid) * 100.0


test_SVC = clf_SVC.predict(X)
acc_SVC = accuracy_score(Y, test_SVC) * 100.0

# identify best classifier 
index = np.argmax([acc_LinearSVC, acc_NearestCentroid, acc_SVC])
classifiers = {0: 'LinearSVC', 1: 'NearestCentroid', 2: 'SVC'}
best_classifier = classifiers[index]
print('Best gender classifier is {}'.format(best_classifier))

# enter user values for best classifier
height = input('What is your height? (cm)')
weight = input('What is your weight? (kg)')
shoe_size = input('What is your shoe size? (eu)')

# catch errors
try:
    height = float(height)
    weight = float(weight)
    shoe_size = float(shoe_size)
except ValueError:
    print("Invalid input entered")
    sys.exit()
   
if height <= 0.0 or weight <= 0.0 or shoe_size <= 0.0:
    print('Invalid number entered')
    sys.exit()

# predict against input
if index == 0:
    pred = clf_LinearSVC.predict([[height, weight, shoe_size]])
elif index == 1:
    pred = clf_NearestCentroid.predict([[height, weight, shoe_size]])
else:
    pred = clf_SVC.predict([[height, weight, shoe_size]])

# determine gender
gender = pred[0]
opp_gender = ''
if gender == 'Male':
    opp_gender = 'Female'
else:
    opp_gender = 'Male'

# write user values to data set
validation = input('Is %s correct? (y/n)' % gender)
if validation == "y":
    genderFile = open('gender.csv','a',newline='')
    genderWriter = csv.writer(genderFile)
    genderWriter.writerow([gender,height,weight,shoe_size])
    genderFile.close()
elif validation == "n":
    genderFile = open('gender.csv','a',newline='')
    genderWriter = csv.writer(genderFile)
    genderWriter.writerow([opp_gender,height,weight,shoe_size])
    genderFile.close()
else:
    sys.exit()




