from sklearn import tree
import numpy as np
from scipy import sparse
import csv
import sys

# gender
independent_list = []

# [height, weight, shoe_size]
dependent_list = []

genderFile = open('gender.csv')
genderReader = csv.reader(genderFile)

iter_genderReader = iter(genderReader)
next(iter_genderReader)

for row in iter_genderReader:
   independent_list.append(row[0])
   dependent_list.append(row[1:])


clf = tree.DecisionTreeClassifier()


# train algorithm on the data
clf = clf.fit(dependent_list, independent_list)

height = int(input('What is your height? (cm)'))
weight = int(input('What is your weight? (kg)'))
shoe_size = int(input('What is your shoe size?'))


prediction = clf.predict([[height, weight, shoe_size]])

print(prediction)


# CHALLENGE - create 3 more classifiers...
# 1
# 2 
# 3