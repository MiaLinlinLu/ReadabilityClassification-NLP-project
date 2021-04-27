# Project Title: Readability Formulas and Text Coherence

Group members: Joon Suh Choi, Linlin Lu, Nahom Ogbazghi, Vinayak Renu Nair


## General Information
This project uses the Newsela corpus and different word vectorization and machine learning (ML) algorithms to produce models predicting text readability.<br/><br/>
Word vectorization is needed to make the data consumable for the models. For the ML algorithms used include 1) SVM, 2) Multinomial Logistic Regression, 3) Naive Bayes, and 4) Transformers that is done through TF-IDF, while fastText and Bert have there own vectorizers.<br/><br/>
Data is fed into the vectorizers in the formats of: plain text, ngram, and removed stopwords.
## Before we start
The dataset files and a fine-tuned bert model can be downloaded from: https://drive.google.com/drive/folders/1C54tPVXpzvXy8aNOvbuMzcJL0Ra87j8x?usp=sharing
Please download and put this files in the same directory of your project before running the project.
## Prerequisite
This project has been tested and installed on Windows. For optimal performance using a device with GPUs will significantly speed up BERT tuning. If the device used does not have GPUs the CPU will be used instead for tuning and that could potentially take hours.<br/>
An NVIDIA GPU and CUDA toolkit is required for the code to run on GPU.<br/>
<br/>Python versions: 3.7-3.9<br/>
The project was tested in Python 3.7.1. to match the Python version used in Google Colab.

## Dependencies used 
pickle, fasttext, pandas, tensorflow, torch, transformers, scipy, os, pandas, numpy, nltk, warnings, tabulate, sklearn, matplotlib.pyplot, seaborn, random

## Getting Started
Create a new virtual environment and activate it by executing activate.bat
```
python -m venv <NAME OF NEW ENVIRONMENT>
```
Install fasttext using separate wheel. (the included wheel is for Python version 3.7. Wheels for other versions can be found here: https://pypi.bartbroe.re/fasttext/)
```
pip install fasttext-0.9.2-cp37-cp37m-win_amd64.whl
```
Install dependencies using requirements.txt
```
pip install -r requirements.txt
```

## Running the Code
Run ML_NLP.py and it will train all models and print the outputs.
```
python ML_NLP.py
```

All results will be logged in a separate log file (if the logger is uncommented out), and all plots will be tabulated on separate png files.

## About the files
fasttext-0.9.2-cp37-cp37m-win_amd64: fasttext whl for python version 3.7<br/>

Newsela_categorized: dataset for training models (and testing models on the same dataset). Five readability clases. File 0 is the most difficult class. File 4 is the easiest class.<br/>
newsela_features_cohesion_selected.csv: cohesive features extracted from Newsela.<br/>
weebit_features_cohesion_selected.csv: cohesive features extracted from Weebit.<br/><br/>
WeeBit-TextOnly_categorized_fortesting: dataset for testing models on a different dataset.<br/>

finetuned_BERT_for_newsela.pt : fine-tuned bert model<br/>
ML_NLP.py: main python file<br/><br/>
helper_functiond.py: helper functions<br/>
cohesive_indices.py: dealing with cohesive_indices<br/>
tfidf_plain.py: word vectorization<br/>
bert_related.py: bert related python file<br/>

ResultExample: stores result examples after running the projects.<br/>
results.log: recording the running results of the main python file.<br/>

## Results
The code was ran on our local machine and the results of the models were logged in the results.log file, and all plots were tabulated on separate png files.<br/><br/>
To log your own results, please uncomment out the following three lines of code in ML_NLP.py
```
# Log all results in a file
logger = open('results.log', 'a', encoding='utf-8', errors='ignore')
sys.stdout = logger
sys.stderr = logger
```
