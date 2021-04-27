import random
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
import pandas as pd


def read_cohesive_indices(filepath):
    # Read in cohesive indices
    df1 = pd.read_csv(f'{filepath}')

    # Create label column using filenames
    df1['labels'] = [int(w.split('.')[-2]) for w in df1['Filename']]

    # Drop all labels == 5
    df1 = df1[df1['labels'] < 5]

    # For future reference
    df1filenames = df1['Filename']
    del df1['Filename']

    # Aggregate all values in separate columns into a single list and create a column with all those lists
    cohesive_train_features = []

    for row in df1.index:
      k = []
      for item in df1.columns[:-1]:
        val = float(df1[item][row])
        k.append(val)
      cohesive_train_features.append(k)

    df1['cohesive_indices'] = cohesive_train_features

    cohindice_dict = dict()

    for fn, ci in zip(df1filenames, df1['cohesive_indices']):
      cohindice_dict[fn] = ci

    return df1['cohesive_indices'], df1['labels'], df1filenames, cohindice_dict


def read_cohesive_indices_weebit(filepath):
    # Read in cohesive indices
    df1 = pd.read_csv(f'{filepath}')

    # Create label column using filenames
    df1['labels'] = [int(w.split('_')[-1]) for w in df1['Filename']]

    # For future reference
    df1filenames = df1['Filename']
    del df1['Filename']

    # Aggregate all values in separate columns into a single list and create a column with all those lists
    cohesive_train_features = []

    for row in df1.index:
      k = []
      for item in df1.columns[:-1]:
        val = float(df1[item][row])
        k.append(val)
      cohesive_train_features.append(k)

    df1['cohesive_indices'] = cohesive_train_features

    cohindice_dict = dict()

    for fn, ci in zip(df1filenames, df1['cohesive_indices']):
      cohindice_dict[fn] = ci

    return df1['cohesive_indices'], df1['labels'], df1filenames, cohindice_dict



def concatenate_bert(all_true_labels, all_pred_labels, cohind):
    km = []
    multiplier = 1

    for ci, pl in zip(cohind, all_pred_labels):
      ci_res = ci[:]
      ci_res.append(pl*multiplier)
      km.append(ci_res)

    return km


def with_TFIDF(data, train_texts, test_texts, cohindice_dict):
    vocab_size=10000
    vectorizer = TfidfVectorizer(max_features=vocab_size)
    vectors = vectorizer.fit_transform(train_texts)
    word_id_pair = vectorizer.vocabulary_
    words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())

    master_dictionary = dict()
    master_ids = dict()
    master_dictionary_reversed = dict()

    idx=0
    for text, filename in zip(data.data, data.filenames):
      filename = os.path.basename(filename)
      master_dictionary[filename] = text
      master_ids[filename] = idx
      master_dictionary_reversed[text] = filename
      idx += 1

    TFIDF_and_cohindex_train = []
    multiplier = 1

    for text, stack in zip(train_texts, vectors):
      TFIDF_and_cohindex_train.append(hstack((stack, np.array(cohindice_dict[master_dictionary_reversed[text]]) * multiplier)).toarray()[0])

    vectorizer = TfidfVectorizer(max_features=vocab_size)
    vectors = vectorizer.fit_transform(test_texts)
    word_id_pair = vectorizer.vocabulary_
    words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())
    # words_df.head()

    TFIDF_and_cohindex_test = []

    for text, stack in zip(test_texts, vectors):
      TFIDF_and_cohindex_test.append(hstack((stack, np.array(cohindice_dict[master_dictionary_reversed[text]]) * multiplier)).toarray()[0])

    return TFIDF_and_cohindex_train, TFIDF_and_cohindex_test
