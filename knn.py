


#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import string
from nltk.stem import PorterStemmer
import re
import spacy
from io import StringIO
from nltk.tokenize import word_tokenize


ps = PorterStemmer()
def removeHtml(review):
    review = re.compile(r'<[^>]+>').sub('', review)
    return review.lower().strip()

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    #tokens = re.split('W+',text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if not word in stop_words]
    return tokens

def preProcess(review: str):
    """This function takes in review from the reviews list and converts into usable data"""

    #Remove the HTML tags
    review = re.compile(r'<[^>]+>').sub('', review)

    doc = nlp(review)
    ps = PorterStemmer()

    tokens = [] #list of tokens
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and token not in string.punctuation:
            cleaned_tokens.append(ps.stem(token)) #Stem the token
    #print(cleaned_tokens)
    return cleaned_tokens


def findSimilarities(train_vect, test_vect):
    cosineSimilarities = np.dot(test_vect, np.transpose(train_vect)) #dot product of the 2 vectors
    similarities = cosineSimilarities.toarray()
    return similarities


def findNeighbors(similarities, k):
    return np.argsort(-similarities)[:k]

def predict(nearestNeighbors, labels):
    """Takes in the list of K nearest Neighbors and the full training labels list, and 
        calculates the count of positive and negative reviews. 
        If positive reviews are more, then the test review is positive and vice-versa"""
    #print(labels)

    positiveReviewsCount = 0
    negativeReviewsCount = 0
    for neighbor in nearestNeighbors:
        #print(neighbor)
        if int(labels[neighbor]) == 1:
            positiveReviewsCount += 1
        else:
            negativeReviewsCount += 1
    if positiveReviewsCount > negativeReviewsCount:
        return 1
    else:
        return -1

nlp = spacy.load('en_core_web_sm')

#Set the stopwords
stop_words = set(nltk.corpus.stopwords.words("english"))

def remove_stopwords(text):
    output= [i for i in text if i not in stop_words]
    return output

#Read in training file
train_data = pd.read_csv('./train_file.txt', sep='\t', header = None, skiprows=[0])
t = open("./test_file.txt", "r")
test_data = t.readlines()
test_data = pd.DataFrame(test_data)
#print(test_data)

#Set the column headers 
column_names = ['sentiment', 'review']
train_data.columns = column_names
test_data.columns = ['review']
#print(data['sentiment'].value_counts())

#Preprocess training batch and test batch
"""for index, row in train_data.iterrows():
    train_data.at[index, 'review'] = preProcess(row['review'])

for index, row in test_data.iterrows():
    test_data.at[index, 'review'] = preProcess(row['review'])"""
train_data['review'] = train_data['review'].apply(lambda x:removeHtml(x))
test_data['review'] = test_data['review'].apply(lambda x:removeHtml(x))


#Remove puntuation
train_data['review'] = train_data['review'].apply(lambda x:remove_punctuation(x))
test_data['review'] = test_data['review'].apply(lambda x:remove_punctuation(x))

#Tokenization
train_data['review'] = train_data['review'].apply(lambda x:tokenization(x))
test_data['review'] = test_data['review'].apply(lambda x:tokenization(x))

#Stemming
def stemming(text):
    stem_text = [ps.stem(word) for word in text]
    return ' '.join(stem_text)

train_data['review'] = train_data['review'].apply(lambda x:stemming(x))
test_data['review'] = test_data['review'].apply(lambda x:stemming(x))

#set 
y = train_data['sentiment']
x = train_data['review']
test_reviews = test_data['review']

#print(test_data)


print("POINT 1")

#split data into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = 0.25, random_state = 0)
#print("Train: ", x_train.shape, y_train.shape, "Test: ", (x_test.shape, y_test.shape))

vectorizer = TfidfVectorizer()
#tf_x_train = vectorizer.fit_transform(x_train)
tf_x_train = vectorizer.fit_transform(x)
#tf_x_test = vectorizer.transform(x_test)
tf_x_test = vectorizer.transform(test_reviews)

print(tf_x_train, tf_x_test)
#print(vectorizer.idf_)


similarities = findSimilarities(tf_x_train, tf_x_test)

print("POINT 2")
print(similarities)

k = 500
test_sentiments = list()

for similarity in similarities:
    knn = findNeighbors(similarity, k)
    prediction = predict(knn, y)
    
    #To write to the list as +1 instead of just a 1 for positive reviews
    if prediction == 1:
        test_sentiments.append('1')
    else:
        test_sentiments.append('-1')

print("POINT 3")
#Write the result to a txt file
output = open('format.txt', 'w')

output.writelines( "%s\n" % item for item in test_sentiments )

output.close()
