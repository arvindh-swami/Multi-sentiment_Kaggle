from __future__ import print_function

import sys
import pandas as pd
import numpy as np
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC



trainingData = sys.argv[1]
testData = sys.argv[2]

trainingData = pd.read_csv(trainingData, sep=',', quotechar='"', header=0)
trainNumRows = trainingData.shape[0]
trainNumColumns = trainingData.shape[1]
#print(trainNumRows)
#print(trainNumColumns)
ogX = trainingData
size = int(trainNumRows*(50.0/100))

#data = pd.read_csv(testData, sep=',', error_bad_lines=False, header=None)
testData = pd.read_csv(testData, sep=',', quotechar='"', header=0)
testNumRows = testData.shape[0]
testNumColumns = testData.shape[1]
#print(testNumRows)
#print(testNumColumns)c



# Create bag of words
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = 'english', max_features = 5000)
train_data_features = vectorizer.fit_transform(trainingData["text"])
test_data_features = vectorizer.transform(testData["text"])

'''
# Train Classifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, trainingData["sentiment"])
result = forest.predict(test_data_features)
'''

model = LinearSVC()
model.fit(train_data_features, trainingData["sentiment"])
result = model.predict(test_data_features)

'''
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, trainingData["sentiment"])
result = forest.predict(test_data_features)
'''

print(result)

#result = [1,2]
i = 0
solution = open('solution7.csv', 'w')
with solution:
   writer = csv.writer(solution)
   writer.writerow(["id", "sentiment"])
   for value in result:
       writer.writerow([i, value])
       i += 1