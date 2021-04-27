

import pickle
import fasttext
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import torch
from transformers import BertTokenizer
from scipy.sparse import hstack
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_files
import sklearn.metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random

nltk.download('punkt')
nltk.download('stopwords')

### ---------------------------------------------------
####--------Helper Functions---------------------------
### ---------------------------------------------------

# Remove stop words function
stop_words = set(stopwords.words('english'))
def get_filtered_data(stop_words, data):
    '''
    function: Remove stop words
    return: [text1,text2,...] filtered
    '''
    filteredData = []
    for content in data.data:
        word_tokens = word_tokenize(content)
        contentList = ""
        for word in word_tokens:
            if word not in stop_words:
                contentList += word
        filteredData.append(contentList)
    return filteredData


def get_words_df(train_texts):
    '''
    function: get vectors after tf-idf
    return: dataframe
    '''
    vocab_size = 10000
    vectorizer = TfidfVectorizer(max_features=vocab_size)
    vectors = vectorizer.fit_transform(train_texts)
    word_id_pair = vectorizer.vocabulary_
    words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())
    return words_df, word_id_pair


def get_confusion_matrix(pred_labels, test_labels, file):
    mat = confusion_matrix(pred_labels, test_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True, cmap='Greens')
    #             xticklabels=train.target_names, yticklabels=train.target_names)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig(file)
    plt.close()


def print_report(y_true, y_pred):
    target_names = ['labels-0', 'labels_1', 'labels_2', 'labels_3', 'labels-4']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))


def model_result_save(y_test, X_test, model, modelname):
    pred_labels = model.predict(X_test)
    get_confusion_matrix(pred_labels, y_test, modelname + '_confusionmatrix.png')
    print('*********************************************************')
    print('report of {} model: '.format(modelname))
    print_report(y_test, pred_labels)


def scores(name, model, y_test, X_test):
    pred_labels = model.predict(X_test)
    return [name, sklearn.metrics.precision_score(y_test, pred_labels, average="macro") * 100.0,
            sklearn.metrics.recall_score(y_test, pred_labels, average="macro") * 100.0,
            sklearn.metrics.f1_score(pred_labels, y_test, average='macro') * 100.0]



# n-grams

def get_ngrams_words_df(train_texts, ngram1=3, ngram2=3):
    vocab_size = 5000
    vectorizer = TfidfVectorizer(max_features=vocab_size, ngram_range=(ngram1, ngram2))
    vectors = vectorizer.fit_transform(train_texts)
    word_id_pair = vectorizer.vocabulary_
    words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())
    return words_df

def write_train2(file_path,train_texts,train_labels):
    with open(file_path, 'w',encoding='utf-8') as f:
        for i,text in enumerate(train_texts):
                f.write('%s __label__%d\n' % (text.replace('\n',' '), train_labels[i]))

def get_clean_texts(test_texts):
    clean_test_texts = []
    for i in test_texts:
        text = i.replace('\n',' ')
        clean_test_texts.append(text)
    return clean_test_texts

def train_models(ngram,file,test_labels, test_data, epoch=20):
    model = fasttext.train_supervised('cooking.train',lr=1,loss = 'ns',ws=70,wordNgrams=ngram,epoch=epoch)
    predits = [int(model.predict(test_data[i])[0][0][-1]) for i in range(len(test_data))]
    print_report(test_labels,predits)
    mat = confusion_matrix(predits,test_labels)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,cmap='Greens');
    #             xticklabels=train.target_names, yticklabels=train.target_names)
    plt.xlabel('predicted label');
    plt.ylabel('true label');
    plt.savefig(file)
    plt.close()


def fasttext_features(ngram, test_data, cohind, epoch=20):
    cohind_res = list(cohind)
    model = fasttext.train_supervised('cooking.train',lr=1,loss = 'ns',ws=70,wordNgrams=ngram,epoch=epoch)
    for i in range(len(test_data)):
        cohind_res[i].append(int(model.predict(test_data[i])[0][0][-1]) )
    return cohind_res
