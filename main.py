import sys
import pandas as pd
import numpy as np
import csv

trainingData = sys.argv[1]
testData = sys.argv[2]

data = pd.read_csv(trainingData, sep=',', quotechar='"', header=0)
trainingData=data.as_matrix()
trainNumRows = trainingData.shape[0]
trainNumColumns = trainingData.shape[1]
#print(trainNumRows)
#print(trainNumColumns)

#data = pd.read_csv(testData, sep=',', error_bad_lines=False, header=None)
data = pd.read_csv(testData, sep=',', quotechar='"', header=0)
testData=data.as_matrix()
testNumRows = testData.shape[0]
testNumColumns = testData.shape[1]
#print(testNumRows)
#print(testNumColumns)

#print(trainingData)
#print(testData)
zero = {} # Very Negative
one = {} # Negative
two = {} # Neutral
three = {} # Positive
four = {} # Very Positive

for id, text, sentiment in trainingData:
    text = text.replace(';', '')
    text = text.replace(',', '')
    text = text.replace('\'', '')

    for word in text.split():
        if sentiment == 0:
            zero[word] = zero.get(word, 0) + 1
        elif sentiment == 1:
            one[word] = one.get(word, 0) + 1
        elif sentiment == 2:
            two[word] = two.get(word, 0) + 1
        elif sentiment == 3:
            three[word] = three.get(word, 0) + 1
        elif sentiment == 4:
            four[word] = four.get(word, 0) + 1

redundantFeatures = ["the", "a", "an", "of", "to", "in", "its", "on", "is", "by", "from", "it", "and"]
result = {}
#for i in range(testNumRows):
#    result.append(0)

for id, text in testData:
    if id == "id":
        continue

    probZero, probOne, probTwo, probThree, probFour = 0.0, 0.0, 0.0, 0.0, 0.0

    text = text.replace(';', '')
    text = text.replace(',', '')
    text = text.replace('\'', '')

    for word in text.split():
        word = word.lower()
        if word in redundantFeatures:
            print("Redundant Feature")
            continue
        countZero = zero.get(word, 0) + 1.0
        countOne = one.get(word, 0) + 1.0
        countTwo = two.get(word, 0) + 1.0
        countThree = three.get(word, 0) + 1.0
        countFour = four.get(word, 0) + 1.0

        probZero += countZero / (countZero + countOne + countTwo + countThree + countFour)
        probOne += countOne / (countZero + countOne + countTwo + countThree + countFour)
        probTwo += countTwo / (countZero + countOne + countTwo + countThree + countFour)
        probThree += countThree / (countZero + countOne + countTwo + countThree + countFour)
        probFour += countFour / (countZero + countOne + countTwo + countThree + countFour)

    prediction = np.array([probZero, probOne, probTwo, probThree, probFour]).argmax(axis = 0)
    #print(prediction)
    # Use id to store?
    result[int(id)] = prediction

#print(result)

#result = [['id', 'sentiment'], [0, 3], [1, 4]]
solution = open('solution2.csv', 'w')
with solution:
   writer = csv.writer(solution)
   writer.writerow(["id", "sentiment"])
   for key, value in result.items():
       writer.writerow([key, value])
