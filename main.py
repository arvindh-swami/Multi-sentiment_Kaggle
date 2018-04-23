import sys
import pandas as pd
import numpy as np

trainingData = sys.argv[1]
testData = sys.argv[2]

data = pd.read_csv(trainingData, sep=',', quotechar='"', header=0)
trainingData=data.as_matrix()
trainNumRows = trainingData.shape[0]
trainNumColumns = trainingData.shape[1]
print(trainNumRows)
print(trainNumColumns)

data = pd.read_csv(testData, sep=',', error_bad_lines=False, header=None)
testData=data.as_matrix()
testNumRows = testData.shape[0]
testNumColumns = testData.shape[1]
print(testNumRows)
print(testNumColumns)

print(trainingData)
print(testData)
