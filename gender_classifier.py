import numpy as np
from scipy import sparse
import csv
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import accuracy_score

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

# close CSV file
genderFile.close()

# covert string values to numbers
X_len = len(X)
for row in range(X_len):
    X[row][0] = float(X[row][0])
    X[row][1] = float(X[row][1])
    X[row][2] = float(X[row][2])

# initialize classifier
clf_LinearSVC = svm.LinearSVC()
clf_NearestCentroid = NearestCentroid()
clf_SVC = svm.SVC()

# train classifier using data set
clf_LinearSVC = clf_LinearSVC.fit(X, Y)
clf_NearestCentroid = clf_NearestCentroid.fit(X, Y)
clf_SVC = clf_SVC.fit(X, Y)

# testing using data set
test_LinearSVC = clf_LinearSVC.predict(X)
acc_LinearSVC = accuracy_score(Y, test_LinearSVC) * 100.0
acc_LinearSVC = int(acc_LinearSVC)
print('Accuracy for Linear SVC: {}'.format(acc_LinearSVC) + '%')


test_NearestCentroid = clf_NearestCentroid.predict(X)
acc_NearestCentroid = accuracy_score(Y, test_NearestCentroid) * 100.0
acc_NearestCentroid = int(acc_NearestCentroid)
print('Accuracy for Nearest Centroid: {}'.format(acc_NearestCentroid) + '%')


test_SVC = clf_SVC.predict(X)
acc_SVC = accuracy_score(Y, test_SVC) * 100.0
acc_SVC = int(acc_SVC)
print('Accuracy for SVC: {}'.format(acc_SVC) + '%')

# identify best classifier 
index = np.argmax([acc_LinearSVC, acc_NearestCentroid, acc_SVC])
classifiers = {0: 'LinearSVC', 1: 'NearestCentroid', 2: 'SVC'}
print('Best gender classifier is {}'.format(classifiers[index]))


# enter user values for classifier
height = input('What is your height? (cm)')
weight = input('What is your weight? (kg)')
shoe_size = input('What is your shoe size? (eu)')

# catch string inputs
try:
    height = float(height)
    weight = float(weight)
    shoe_size = float(shoe_size)
except ValueError:
    print("Invalid input entered")
    exit()
# catch negative and zero inputs
if height <= 0.0 or weight <= 0.0 or shoe_size <= 0.0:
    print('Invalid number entered')
    exit()


# input prediction
pred = clf_LinearSVC.predict([[height, weight, shoe_size]])



# display predictions
gender_LinearSVC = pred_LinearSVC[0]
gender_NearestCentroid = pred_NearestCentroid[0]
gender_SVC = pred_SVC[0]

print("Linear Support Vector - " + gender_LinearSVC)
print("Nearest Centroid - " + gender_NearestCentroid)
print("C-Support Vector - " + gender_SVC)

# determine final prediction
result = ''
if gender_LinearSVC == gender_NearestCentroid or gender_LinearSVC == gender_SVC:
    result = gender_LinearSVC
else:
    result = gender_SVC

# determine gender that is not prediction
anti_result = ''
if result == 'Male':
    anti_result = 'Female'
else:
    anti_result = 'Male'

# write user values to data set
validation = input('Is ' + result + ' correct? (y/n)')
if validation == "y":
    genderFile = open('gender.csv','a',newline='')
    genderWriter = csv.writer(genderFile)
    genderWriter.writerow([result,height,weight,shoe_size])
    genderFile.close()
elif validation == "n":
    genderFile = open('gender.csv','a',newline='')
    genderWriter = csv.writer(genderFile)
    genderWriter.writerow([anti_result,height,weight,shoe_size])
    genderFile.close()
else:
    exit()




