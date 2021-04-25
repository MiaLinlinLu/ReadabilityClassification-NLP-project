# Project Title: Readability Formulas and Text Coherence
CS8980 - Dr. Juan M. Banda\
Group members: Joon Suh Choi, Linlin Lu, Nahom Ogbazghi, Vinayak Renu Nair


## General Information
This project uses the Newsela corpus and different word vectorization and machine learning (ML) algorithms to produce models predicting text readability.<br/><br/>
Word vectorization is done through 1) TF-IDF, 2) FastText, and 3) Bert, and ML algorithms used include 1) SVM, 2) Multinomial Logistic Regression, 3) Naive Bayes, and 4) Transformers. Data is fed into these models in the formats of: plain text, ngram, and removed stopwords.

## Prerequisite
This project has been tested and installed on Windows. For optimal performance using a device with GPUs will significantly speed up BERT tuning. None of the devices used had GPUs and instead the CPU was used for tuning and that took hours.

## Getting Started
Create a new virtual environment and install dependencies using requirements.txt
```
pip install -r requirements.txt
```

## Running the Code
Run ML_NLP.py and it will train all models and print the outputs.
```
python ML_NLP.py
```
