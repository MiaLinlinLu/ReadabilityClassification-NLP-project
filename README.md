# Project Title: Readability Formulas and Text Coherence

Group members: Joon Suh Choi, Linlin Lu, Nahom Ogbazghi, Vinayak Renu Nair


## General Information
This project uses the Newsela corpus and different word vectorization and machine learning (ML) algorithms to produce models predicting text readability.<br/><br/>
Word vectorization is done through 1) TF-IDF, 2) FastText, and 3) Bert, and ML algorithms used include 1) SVM, 2) Multinomial Logistic Regression, 3) Naive Bayes, and 4) Transformers.<br/>Data is fed into these models in the formats of: plain text, ngram, and removed stopwords.

## Prerequisite
This project has been tested and installed on Windows. For optimal performance using a device with GPUs will significantly speed up BERT tuning. If the device used does not have GPUs the CPU will be used instead for tuning and that could potentially take hours.<br/>
<br/>Python version: 3.7-3.9The project was tested in Python 3.7.1. to match the Python version used in Google Colab.

## Getting Started
Create a new virtual environment and activate it by executing activate.bat
```
python -m venv <NAME OF NEW ENVIRONMENT>
```
Install fasttext using separate wheel. (the included wheel is for Python versions 3.7 and 3.8. Wheels for other versions can be found here: https://pypi.bartbroe.re/fasttext/)
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

All results will be logged in a separate log file, and all plots will be tabulated on separate png files.
