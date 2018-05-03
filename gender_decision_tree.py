import numpy as np
from scipy import sparse
import csv
from sklearn import tree

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

# train classifier using data set
clf = clf.fit(X, Y)

# enter user values for  classifier
height = input('What is your height? (cm)')
weight = input('What is your weight? (kg)')
shoe_size = input('What is your shoe size? (eu)')

#classifier prediction
prediction = clf.predict([[height, weight, shoe_size]])
gender = prediction[0]

# write user values to data set if prediction is correct
validation = input('Is ' + gender + ' correct? (y/n)')
if validation == "y":
    genderFile = open('gender.csv','a',newline='')
    genderWriter = csv.writer(genderFile)
    genderWriter.writerow([gender,height,weight,shoe_size])
    genderFile.close()
else:
    exit()




