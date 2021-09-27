


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
import nltk
from io import StringIO
from nltk.tokenize import word_tokenize


ps = PorterStemmer()

#Stemming Function
def stemming(review):
    stem_review = [ps.stem(word) for word in review]
    return ' '.join(stem_review)

#Fuction to remove HTML
def removeHtml(review):
    review = re.compile(r'<[^>]+>').sub('', review)
    return review.lower().strip()

#Function to remove punctuation
def remove_punctuation(review):
    punctuationfree="".join([i for i in review if i not in string.punctuation])
    return punctuationfree

#Tokenizer
def tokenization(review):
    #tokens = re.split('W+',text)
    tokens = word_tokenize(review)
    tokens = [word for word in tokens if not word in stop_words]
    return tokens

#Finds similarities between training vectors and test vector
def findSimilarities(train_vect, test_vect):
    cosineSimilarities = np.dot(test_vect, np.transpose(train_vect)) #dot product of the 2 vectors
    similarities = cosineSimilarities.toarray()
    return similarities


def findNeighbors(similarities, k):
    return np.argsort(-similarities)[:k]

def predict(nearestNeighbors, sentiments):
    """Calculates the count of positive and negative reviews from similarity vector. 
        If positive reviews are more, then the test review is positive and vice-versa"""
    #print(labels)

    positiveReviews = 0
    negativeReviews = 0
    for neighbor in nearestNeighbors:
        #print(neighbor)
        if int(sentiments[neighbor]) == 1:
            positiveReviews += 1
        else:
            negativeReviews += 1
    if positiveReviews > negativeReviews:
        return 1
    else:
        return -1

nlp = spacy.load('en_core_web_sm')

#Set the stopwords
stop_words = set(nltk.corpus.stopwords.words("english"))

def remove_stopwords(review):
    output= [i for i in review if i not in stop_words]
    return output

#Read in training file
train_data = pd.read_csv('./train_file.txt', sep='\t', header = None, skiprows=[0])
t = open("./test_file.txt", "r")
test_data = t.readlines()
test_data = pd.DataFrame(test_data)
print("Reading in data...\n")

#Set the column headers 
column_names = ['sentiment', 'review']
train_data.columns = column_names
test_data.columns = ['review']
print("Preprocessing training and testing files...\n")

#Remove HTML
train_data['review'] = train_data['review'].apply(lambda x:removeHtml(x))
test_data['review'] = test_data['review'].apply(lambda x:removeHtml(x))

#Remove puntuation
train_data['review'] = train_data['review'].apply(lambda x:remove_punctuation(x))
test_data['review'] = test_data['review'].apply(lambda x:remove_punctuation(x))

#Tokenization
train_data['review'] = train_data['review'].apply(lambda x:tokenization(x))
test_data['review'] = test_data['review'].apply(lambda x:tokenization(x))

#Stemming
train_data['review'] = train_data['review'].apply(lambda x:stemming(x))
test_data['review'] = test_data['review'].apply(lambda x:stemming(x))

#set 
y = train_data['sentiment']
x = train_data['review']
test_reviews = test_data['review']

#split data into training and testing sets
#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = 0.25, random_state = 0)
#print("Train: ", x_train.shape, y_train.shape, "Test: ", (x_test.shape, y_test.shape))

print("Creating TF-IDF vectors...\n")
#Vectorization
vectorizer = TfidfVectorizer()
tf_x_train = vectorizer.fit_transform(x)
tf_x_test = vectorizer.transform(test_reviews)


#print(tf_x_train, tf_x_test)
#print(vectorizer.idf_)
print("Creating similarity model...\n")

#Find Similarities
similarities = findSimilarities(tf_x_train, tf_x_test)


k = 250
test_sentiments = list()

print("Predicting Sentiments...\n")
for similarity in similarities:
    knn = findNeighbors(similarity, k)
    prediction = predict(knn, y)
    
    #To write to the list as +1 instead of just a 1 for positive reviews
    if prediction == 1:
        test_sentiments.append('1')
    else:
        test_sentiments.append('-1')


#Write the result to a txt file
output = open('format.txt', 'w')

output.writelines( "%s\n" % item for item in test_sentiments )

output.close()
print("Predictions complete")