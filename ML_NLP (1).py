#!/usr/bin/env python
# coding: utf-8

# Code for machine learning_NLP

# In[1]:


# !pip install -r requirements.txt


# In[2]:


# save print contents to 'a.log'

from cs88.helper_functions import load_files, os, random, np, train_test_split, get_words_df, get_filtered_data, stop_words, tabulate, pd
from cs88.tfidf_plain import svmResults, nbResults, rfResults, lrResults


# Get data
# path = os.getcwd()+"Newsela_categorized.zip"
# os.system('!unzip '+path+' -d newsela')
# os.system('!rm -rf '+os.getcwd()+'/newsela/Newsela_categorized/5')
p1 = r"Newsela_categorized"
data = load_files(p1, encoding='utf-8')
# !rm -rf Newsela_categorized/5

# w1 = r"WeeBit-TextOnly_categorized_fortesting"
# weebit_data = load_files(w1, encoding='utf-8', decode_error='ignore')


# In[12]:


# Split data
random.seed(12345)
np.random.seed(12345)
train_texts, test_texts, train_labels, test_labels = train_test_split(data.data, data.target, test_size=0.4)
words_df,word_id_pair = get_words_df(train_texts)
# train_texts1, test_texts1, train_labels1, test_labels1=train_texts, test_texts, train_labels, test_labels
X = words_df
y = train_labels


# In[13]:


filteredData = get_filtered_data(stop_words,data)
filtered_train_texts, filtered_test_texts, filtered_train_labels, filtered_test_labels = train_test_split(filteredData, data.target, test_size=0.4)
f_words_df,f_word_id_pair = get_words_df(filtered_train_texts)
f_X = f_words_df
f_y = filtered_train_labels

### 
print('*'*80)
print('*'*80)
print('Train models without removing stop words.')
randomSeed = 12345

### Prepare test data
print('*'*80)
print('Prepare test data')
words_df,_ = get_words_df(test_texts)
f_words_df,_ = get_words_df(filtered_test_texts)
X_test = words_df
y_test = test_labels
f_X_test = f_words_df
f_y_test = filtered_test_labels
print('Test data was prepared')


svm_scores = svmResults(randomSeed, X, y, y_test, X_test, 'SVM')
nb_scores = nbResults(randomSeed, X, y, y_test, X_test, 'NB')
forest_scores = rfResults(randomSeed, X, y, y_test, X_test,'Forest')
lr_scores = lrResults(randomSeed, X, y, y_test, X_test, 'LogisticRegression')

# In[22]:

print('*'*80)
print('*'*80)
print('Train models after removing stop words.')

f_svm_scores = svmResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fSVM')
f_nb_scores = nbResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fNB')
f_lr_scores = lrResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fLR')
f_forest_scores = rfResults(randomSeed, f_X, f_y, f_y_test, f_X_test, '_fRF')



data = [svm_scores, nb_scores, forest_scores, lr_scores]
columns = ["Name", "Precision", "Recall", "F1 Measure"]
scores1 = pd.DataFrame(data=data,columns=columns)
print("Scores")
print(tabulate(scores1, headers='keys', tablefmt='fancy_grid'))

f_data = [f_svm_scores, f_nb_scores, f_forest_scores, f_lr_scores]
f_scores = pd.DataFrame(data=f_data,columns=columns)
print("F_Scores")
print(tabulate(f_scores, headers='keys', tablefmt='fancy_grid'))


# ### **For ngrams**
#
# https://colab.research.google.com/github/jmbanda/CSC8980_NLP_Spring2021/blob/main/Class06_Language_Models_II.ipynb

# In[27]:


words_df = get_ngrams_words_df(train_texts)
f_words_df = get_ngrams_words_df(filtered_train_texts)

X = words_df
y = train_labels
f_X = f_words_df
f_y = filtered_train_labels


# In[28]:


###
print('*'*80)
print('*'*80)
print('with ngram- Train models without removing stop words.')
randomSeed = 12345
# Create and train a linear support vector classifier (LinearSVC)
print('*'*80)
print('train A SVM ')
SVM = LinearSVC(random_state=randomSeed)
SVM.fit(X, y)

# Create and train a multinomial naive bayes classifier (MultinomialNB)
print('*'*80)
print("Train B NB")
NB = MultinomialNB()
NB.fit(X, y)

# Create and train a random forest classifier
print('*'*80)
print("Train C RF")
forest = RandomForestClassifier(random_state=randomSeed)
forest.fit(X, y)

# Create and train a Logistic regression classifier
print('*'*80)
print("Train D LR")
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs',random_state=randomSeed,max_iter=200)
lr.fit(X, y)


# In[29]:


print('*'*80)
print('*'*80)
print('with ngram-Train models after removing stop words.')

print('*'*80)
print('train A SVM ')
f_SVM = LinearSVC(random_state=randomSeed)
f_SVM.fit(f_X, f_y)

print('*'*80)
print("Train B NB")
f_NB = MultinomialNB()
f_NB.fit(f_X, f_y)

print('*'*80)
print("Train C RF")
f_forest = RandomForestClassifier(random_state=randomSeed)
f_forest.fit(f_X, f_y)

print('*'*80)
print("Train D LR")
f_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs',random_state=randomSeed,max_iter=200)
f_lr.fit(f_X, f_y)


# In[30]:


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


# In[31]:


model_result_save(y_test,X_test,SVM,'ngramSVM')


# In[32]:


model_result_save(y_test,X_test,NB,'ngramNB')


# In[33]:


model_result_save(y_test,X_test,forest,'ngramForest')


# In[34]:


model_result_save(y_test,X_test,lr,'ngramLogisticRegression')


# In[35]:


model_result_save(y_test,X_test,f_SVM,'ngram_fSVM')


# In[36]:


model_result_save(y_test,X_test,f_NB,'ngram_fNB')


# In[37]:


model_result_save(y_test,X_test,f_forest,'ngram_fForest')


# In[38]:


model_result_save(y_test,X_test,f_lr,'ngram_fLogisticRegression')


# In[39]:


forest_scores = scores("Forest", forest,y_test,X_test)
nb_scores = scores("NB", NB,y_test,X_test)
svm_scores = scores("SVM", SVM,y_test,X_test)
lr_scores = scores("LR", lr,y_test,X_test)

f_forest_scores = scores("f_Forest", f_forest,y_test,X_test)
f_nb_scores = scores("f_NB", f_NB, y_test,X_test)
f_svm_scores = scores("f_SVM", f_SVM,y_test,X_test)
f_lr_scores = scores("f_lr", f_lr,y_test,X_test)


# In[40]:


data = [svm_scores, nb_scores, forest_scores, lr_scores]
columns = ["Name", "Precision", "Recall", "F1 Measure"]
scores2 = pd.DataFrame(data=data,columns=columns)
print("Scores")
print(tabulate(scores2, headers='keys', tablefmt='fancy_grid'))

f_data = [f_svm_scores, f_nb_scores, f_forest_scores, f_lr_scores]
f_scores = pd.DataFrame(data=f_data,columns=columns)
print("F_Scores")
print(tabulate(f_scores, headers='keys', tablefmt='fancy_grid'))


# # Fasttext

# In[41]:


###-------------
# print('*'*80)
# print('Fasttext Model')
#
#
# # In[42]:
#
#
#
# file_path = 'cooking.train'
# write_train2(file_path, train_texts1, train_labels1)
#
#
# # In[43]:
#
#
#
# test_data = get_clean_texts(test_texts1)
#
#
# # In[44]:
#
#
# def train_models(ngram,file,test_labels1=test_labels1,epoch=20):
#         model = fasttext.train_supervised('cooking.train',lr=1,loss = 'ns',ws=70,wordNgrams=ngram,epoch=epoch)
#         predits = [int(model.predict(test_data[i])[0][0][-1]) for i in range(len(test_data))]
#         print_report(test_labels,predits)
#         mat = confusion_matrix(predits,test_labels1)
#         sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,cmap='Greens');
#         #             xticklabels=train.target_names, yticklabels=train.target_names)
#         plt.xlabel('predicted label');
#         plt.ylabel('true label');
#         plt.savefig(file)
#
#
# # In[45]:
#
#
# train_models(1,'fasttext_1gram_confusionmatrix.png')
#
#
# # In[46]:
#
#
# train_models(2,'fasttext_2gram_confusionmatrix.png')
#
#
# # In[47]:
#
#
# train_models(3,'fasttext_3gram_confusionmatrix.png')
#
#
# # In[48]:
#
#
# train_models(4,'fasttext_4gram_confusionmatrix.png')


# In[49]:


print('*'*80)
print('---------------BERT------------------------------')


# In[50]:


import tensorflow as tf
import torch
from transformers import BertTokenizer


# In[51]:


# !pip freeze > requirements.txt

