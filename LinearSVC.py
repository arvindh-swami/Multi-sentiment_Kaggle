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
from sklearn import linear_model
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


trainingData = sys.argv[1]
testData = sys.argv[2]

trainingData = pd.read_csv(trainingData, sep=',', quotechar='"', header=0)
trainNumRows = trainingData.shape[0]
trainNumColumns = trainingData.shape[1]
#print(trainNumRows)
#print(trainNumColumns)
ogX = trainingData
size = int(trainNumRows*(90.0/100))

#data = pd.read_csv(testData, sep=',', error_bad_lines=False, header=None)
testData = pd.read_csv(testData, sep=',', quotechar='"', header=0)
testNumRows = testData.shape[0]
testNumColumns = testData.shape[1]
#print(testNumRows)
#print(testNumColumns)c

#trainingData = trainingData[:size]
#print(trainingData.shape[0])

class LemmaTokenizer(object): #tokenizer for CountVectorizer for stemming using Wordnet Corpora
      def __init__(self):
            self.wnl = WordNetLemmatizer()
      def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# Create bag of words
#vectorizer = CountVectorizer(analyzer = "word", ngram_range = (1,3), tokenizer = None, preprocessor = None, stop_words = 'english', max_features = 5000)
#vectorizer = CountVectorizer(analyzer = "word", ngram_range = (1,8), tokenizer = None, preprocessor = None, stop_words = 'english', max_features = 5000)
#vectorizer = HashingVectorizer(analyzer = "word", ngram_range = (1,3), tokenizer = None, preprocessor = None, stop_words = 'english')
vectorizer = HashingVectorizer(analyzer = "word", ngram_range = (1,3), tokenizer = LemmaTokenizer(), preprocessor = None, stop_words = 'english')

train_data_features = vectorizer.fit_transform(trainingData["text"])
train_data_features = vectorizer.fit_transform(trainingData["text"])
test_data_features = vectorizer.transform(testData["text"])

model = LinearSVC()
model.fit(train_data_features, trainingData["sentiment"])
result = model.predict(test_data_features)

'''model = MLPClassifier()
model.fit(train_data_features, trainingData["sentiment"])
result = model.predict(test_data_features)'''

'''
model = linear_model.SGDClassifier()
model.fit(train_data_features, trainingData["sentiment"])
result = model.predict(test_data_features)


model = PassiveAggressiveClassifier(random_state=0)
model.fit(train_data_features, trainingData["sentiment"])
result = model.predict(test_data_features)
'''

print(result)

#result = [1,2]
i = 0
solution = open('solution16.csv', 'w')
with solution:
   writer = csv.writer(solution)
   writer.writerow(["id", "sentiment"])
   for value in result:
       writer.writerow([i, value])
       i += 1

result = model.predict(train_data_features)
correct = 0.0
for i in range(len(result)):
    if result[i] == trainingData["sentiment"][i]:
        correct += 1

print(correct/len(result))
