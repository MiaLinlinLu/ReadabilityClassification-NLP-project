#!/usr/bin/env python
# coding: utf-8
# Code for machine learning_NLP

from helper_functions import load_files, os, random, np, train_test_split, get_words_df, get_filtered_data, \
    stop_words, tabulate, pd, get_ngrams_words_df, write_train2, get_clean_texts, train_models, pickle, torch, fasttext_features
from tfidf_plain import svmResults, nbResults, rfResults, lrResults
from bert_related import finetune_bert, load_bert, bert_eval, bert_set_device, test_bert
from cohesive_indices import read_cohesive_indices, concatenate_bert, with_TFIDF, read_cohesive_indices_weebit
import sys


# # Log all results in a file
# logger = open('results.log', 'a', encoding='utf-8', errors='ignore')
# sys.stdout = logger
# sys.stderr = logger


# Load data using sklearn's load_files function
p1 = r"Newsela_categorized"
data = load_files(p1, encoding='utf-8')


# Set seeds
randomSeed = 12345
random.seed(randomSeed)
np.random.seed(randomSeed)


# Split data
print("Splitting texts...")
train_texts, test_texts, train_labels, test_labels = train_test_split(data.data, data.target, test_size=0.4)

words_df,word_id_pair = get_words_df(train_texts)
train_texts1, test_texts1, train_labels1, test_labels1=train_texts, test_texts, train_labels, test_labels

X = words_df
y = train_labels

# To create filtered text
print("Creating filtered texts... This might take up to one minute.")
filteredData = get_filtered_data(stop_words,data)
filtered_train_texts, filtered_test_texts, filtered_train_labels, filtered_test_labels = train_test_split(filteredData, data.target, test_size=0.4)
f_words_df,f_word_id_pair = get_words_df(filtered_train_texts)
f_X = f_words_df
f_y = filtered_train_labels


print('*'*80)
print('*'*80)


### Prepare test data
print('*'*80)
print('Preparing test data')
words_df,_ = get_words_df(test_texts)
f_words_df,_ = get_words_df(filtered_test_texts)

X_test = words_df
y_test = test_labels
f_X_test = f_words_df
f_y_test = filtered_test_labels

print('Test data was prepared')


# Train models using plain texts as input
print('Training models using Plain texts.')

svm_scores = svmResults(randomSeed, X, y, y_test, X_test, 'SVM')
nb_scores = nbResults(randomSeed, X, y, y_test, X_test, 'NB')
forest_scores = rfResults(randomSeed, X, y, y_test, X_test,'Forest')
lr_scores = lrResults(randomSeed, X, y, y_test, X_test, 'LogisticRegression')


print('*'*80)
print('*'*80)


# Train models using filtered texts as input
print('Training models using Filtered texts.')
f_svm_scores = svmResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fSVM')
f_nb_scores = nbResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fNB')
f_lr_scores = lrResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fLR')
f_forest_scores = rfResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fRF')


# Print outputs of the models that were trained above

datascores = [svm_scores, nb_scores, forest_scores, lr_scores]
columns = ["Name", "Precision", "Recall", "F1 Measure"]
scores1 = pd.DataFrame(data=datascores,columns=columns)
print("Scores")
print(tabulate(scores1, headers='keys', tablefmt='fancy_grid'))

f_data = [f_svm_scores, f_nb_scores, f_forest_scores, f_lr_scores]
f_scores = pd.DataFrame(data=f_data,columns=columns)
print("F_Scores")
print(tabulate(f_scores, headers='keys', tablefmt='fancy_grid'))


### **For ngrams**
words_df = get_ngrams_words_df(train_texts)
f_words_df = get_ngrams_words_df(filtered_train_texts)

X = words_df
y = train_labels
f_X = f_words_df
f_y = filtered_train_labels





### Prepare test data
print('*'*80)
print('Prepare test data')
words_df = get_ngrams_words_df(test_texts)
f_words_df = get_ngrams_words_df(filtered_test_texts)

X_test = words_df
y_test = test_labels
f_X_test = f_words_df
f_y_test = filtered_test_labels
print('Test data was prepared')

###
print('*'*80)
print('*'*80)
print('with ngram- Train models without removing stop words.')

ngram_svm_results = svmResults(randomSeed,X,y,y_test,X_test,'ngramSVM')
ngram_nb_results = nbResults(randomSeed,X,y,y_test,X_test,'ngramNB')
ngram_rf_results = rfResults(randomSeed,X,y,y_test,X_test,'ngramForest')
ngram_lr_results = lrResults(randomSeed,X,y,y_test,X_test,'ngramLogisticRegression')


print('*'*80)
print('*'*80)
print('with ngram-Train models after removing stop words.')

f_ngram_svm_results = svmResults(randomSeed,f_X, f_y,f_y_test,f_X_test,'ngram_fSVM')
f_ngram_nb_results = nbResults(randomSeed,f_X, f_y,f_y_test,f_X_test,'ngram_fNB')
f_ngram_rf_results = rfResults(randomSeed,f_X, f_y,f_y_test,f_X_test,'ngram_fForest')
f_ngram_lr_results = lrResults(randomSeed,f_X, f_y,f_y_test,f_X_test,'ngram_fLogisticRegression')


datascores2 = [ngram_svm_results, ngram_nb_results, ngram_rf_results, ngram_lr_results]
columns = ["Name", "Precision", "Recall", "F1 Measure"]
scores2 = pd.DataFrame(data=datascores2,columns=columns)
print("Scores")
print(tabulate(scores2, headers='keys', tablefmt='fancy_grid'))

f_data = [f_ngram_svm_results, f_ngram_nb_results, f_ngram_rf_results, f_ngram_lr_results]
f_scores = pd.DataFrame(data=f_data,columns=columns)
print("F_Scores")
print(tabulate(f_scores, headers='keys', tablefmt='fancy_grid'))


# Fasttext

print('*'*80)
print('Fasttext Model')

print('*'*100)

testsize=0.4
print('with TESTSIZE={}-------'.format(testsize))
file_path = 'cooking.train'

# Get data
p1 = r"Newsela_categorized"
data = load_files(p1, encoding='utf-8')
# Split data
train_texts1, test_texts1, train_labels1, test_labels1 = train_test_split(data.data, data.target, test_size=testsize)

# Prepare training data
write_train2(file_path, train_texts1, train_labels1)
# Prepare testing data
test_data = get_clean_texts(test_texts1)

# Training n-grams from 1gram to 8gram models.
ngramnum=1 # n for denoting n-grams
for i in range(8):
    print('train fasttext models with {} gram'.format(ngramnum))
    fig_name = 'fasttext_'+str(ngramnum)+'gram_confusionmatrix.png'
    train_models(ngramnum,fig_name,test_labels = test_labels1,test_data=test_data)
    ngramnum+=1


print('*'*80)
print('---------------BERT------------------------------')

device = bert_set_device()

# # Fine-tune and save model (this part is already done):
# model_bert = finetune_bert(train_texts, train_labels, "finetuned_BERT_for_newsela.pt", device)
# print('---------------Finetuning complete------------------------------')

# When loading from model:
model_bert = load_bert("finetuned_BERT_for_newsela.pt", device)

# Make sure that model is loaded to whichever device that is being used
model_bert = model_bert.to(device)
true_labels, pred_labels = test_bert(test_texts, test_labels, model_bert, device)

# Evaluate BERT using test set and print results
bert_evaluated_df = bert_eval(true_labels, pred_labels)
print(bert_evaluated_df)


print('*'*80)
print('---------------SVM With Cohesive Indices---------------------------')

cohind, cohlabels, df1filenames, cohindice_dict = read_cohesive_indices("newsela_features_cohesion_selected.csv")


# Split data
coh_train_texts, coh_test_texts, coh_train_labels, coh_test_labels = train_test_split(cohind, cohlabels, test_size= 0.4)


# For future integration with other models
train_text_ids = [df1filenames[w] for w in list(coh_train_texts.index)]
test_text_ids = [df1filenames[w] for w in list(coh_test_texts.index)]


coh_train_texts = list(coh_train_texts)
coh_test_texts = list(coh_test_texts)
coh_train_labels = list(coh_train_labels)
coh_test_labels = list(coh_test_labels)


coh_svm_scores = svmResults(randomSeed, coh_train_texts, coh_train_labels, coh_test_labels, coh_test_texts, 'SVM with Cohesive Indices')


print('*'*80)
print('---------------TF-IDF With Cohesive Indices---------------------------')

coh_tfidf_features_train, coh_tfidf_features_test = with_TFIDF(data, train_texts, test_texts, cohindice_dict)
coh_tfidf_scores = svmResults(randomSeed, coh_tfidf_features_train, coh_train_labels, coh_test_labels, coh_tfidf_features_test, 'SVM with TF-IDF Indices')


print('*'*80)
print('---------------Fasttext With Cohesive Indices---------------------------')

refined_dat = get_clean_texts(data.data)
ft_features = fasttext_features(4, refined_dat, cohind)

ft_coh_train_texts, ft_coh_test_texts, ft_coh_train_labels, ft_coh_test_labels = train_test_split(ft_features, cohlabels, test_size= 0.4)

coh_fasttext_scores = svmResults(randomSeed, ft_coh_train_texts, ft_coh_train_labels, ft_coh_test_labels, ft_coh_test_texts, 'FastText with Cohesive Indices')

print('*'*80)
print('---------------BERT With Cohesive Indices---------------------------')

true_labels, pred_labels = test_bert(data.data, data.target, model_bert, device)
bert_and_cohesive_features = concatenate_bert(true_labels, pred_labels, cohind)

bert_coh_train_texts, bert_coh_test_texts, bert_coh_train_labels, bert_coh_test_labels = train_test_split(bert_and_cohesive_features, cohlabels, test_size=0.4)

coh_BERT_scores = svmResults(randomSeed, bert_coh_train_texts, bert_coh_train_labels, bert_coh_test_labels, bert_coh_test_texts, 'Bert with Cohesive Indices')



print('*'*80)
print('---------------Testing the best models on Weebit---------------------------')


p2 = r"WeeBit-TextOnly_categorized_fortesting"
wdata = load_files(p2, encoding='utf-8', decode_error='ignore')


print('*'*80)
print('---------------Starting with BERT---------------------------')
true_labels, pred_labels = test_bert(wdata.data, wdata.target, model_bert)

# Evaluate BERT using test set and print results
bert_evaluated_df = bert_eval(true_labels, pred_labels)
print(bert_evaluated_df)


print('*'*80)
print('---------------With Fasttext---------------------------')

refined_dat = get_clean_texts(wdata.data)

fig_name = '4gramforWeebit.png'
train_models(4 ,fig_name, test_labels=wdata.target, test_data=refined_dat, epoch=20)



print('*'*80)
print('---------------SVM With Cohesive Indices---------------------------')
wcohind, wcohlabels, wdf1filenames, wcohindice_dict = read_cohesive_indices_weebit("weebit_features_cohesion_selected.csv")

# Split data
wcoh_train_texts, wcoh_test_texts, wcoh_train_labels, wcoh_test_labels = train_test_split(cohind, cohlabels, test_size= 0.4)


wcoh_train_texts = list(wcoh_train_texts)
wcoh_test_texts = list(wcoh_test_texts)
wcoh_train_labels = list(wcoh_train_labels)
wcoh_test_labels = list(wcoh_test_labels)


coh_svm_scores = svmResults(randomSeed, coh_train_texts, coh_train_labels, wcoh_test_labels, wcoh_test_texts, 'SVM with Cohesive Indices')


print('*'*80)
print('---------------Fasttext With Cohesive Indices---------------------------')
wrefined_dat = get_clean_texts(wdata.data)
wft_features = fasttext_features(4, wrefined_dat, wcohind)

wft_coh_train_texts, wft_coh_test_texts, wft_coh_train_labels, wft_coh_test_labels = train_test_split(wft_features, wcohlabels, test_size= 0.4)

coh_fasttext_scores = svmResults(randomSeed, ft_coh_train_texts, ft_coh_train_labels, wft_coh_test_labels, wft_coh_test_texts, 'FastText with Cohesive Indices')
