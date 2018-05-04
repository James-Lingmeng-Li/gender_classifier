import numpy as np
from scipy import sparse
import csv
from sklearn import tree, svm
from sklearn.linear_model import SGDClassifier

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

# initialize classifier
clf = tree.DecisionTreeClassifier()
clf2 = SGDClassifier()
clf3 = svm.SVC()

# train classifier using data set
clf = clf.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

# enter user values for  classifier
height = float(input('What is your height? (cm)'))
weight = float(input('What is your weight? (kg)'))
shoe_size = float(input('What is your shoe size? (eu)'))

# classifier prediction
prediction = clf.predict([[height, weight, shoe_size]])
prediction2 = clf2.predict([[height, weight, shoe_size]])
prediction3 = clf3.predict([[height, weight, shoe_size]])

# display prediction
gender = prediction[0]
gender2 = prediction2[0]
gender3 = prediction3[0]
print("Decision Tree - " + gender)
print("SGD Classifier - " + gender2)
print("Linear SVC - " + gender3)

# determine result
result = ''
if gender == gender2 or gender == gender3:
    result = gender
else:
    result = gender3


# write user values to data set if prediction is correct
validation = input('Is ' + result + ' correct? (y/n)')
if validation == "y":
    genderFile = open('gender.csv','a',newline='')
    genderWriter = csv.writer(genderFile)
    genderWriter.writerow([result,height,weight,shoe_size])
    genderFile.close()
else:
    exit()




