from sklearn import tree
import numpy as np
from scipy import sparse
import csv

# gender
independent_list = []

# [height, weight, shoe_size]
dependent_list = []

# read CSV file
genderFile = open('gender.csv')
genderReader = csv.reader(genderFile)

# skip header
iter_genderReader = iter(genderReader)
next(iter_genderReader)

# populate lists
for row in iter_genderReader:
   independent_list.append(row[0])
   dependent_list.append(row[1:])

# instantiate the classifier
clf = tree.DecisionTreeClassifier()

# train the classifier using the data
clf = clf.fit(dependent_list, independent_list)

# enter prediction values
height = input('What is your height? (cm)')
weight = input('What is your weight? (kg)')
shoe_size = input('What is your shoe size? (eu)')

#classifier prediction
prediction = clf.predict([[height, weight, shoe_size]])

print(prediction)