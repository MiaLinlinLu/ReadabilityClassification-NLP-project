from helper_functions import LinearSVC, MultinomialNB, RandomForestClassifier, LogisticRegression, \
    model_result_save, scores


# Create and train a linear support vector classifier (LinearSVC)
def trainSVM(randomSeed, X, y):
    print('*'*80)
    print('train A SVM ')
    SVM = LinearSVC(random_state=randomSeed)
    SVM.fit(X, y)
    return SVM

# Create and train a multinomial naive bayes classifier (MultinomialNB)
def trainNB(randomSeed, X, y):
    print('*'*80)
    print("Train B NB")
    NB = MultinomialNB()
    NB.fit(X, y)
    return NB

# Create and train a random forest classifier
def trainRF(randomSeed, X, y):
    print('*'*80)
    print("Train C RF")
    forest = RandomForestClassifier(random_state=randomSeed)
    forest.fit(X, y)
    return forest

# Create and train a Logistic regression classifier
def trainLR(randomSeed, X, y):
    print('*'*80)
    print("Train D LR")
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs',random_state=randomSeed,max_iter=200)
    lr.fit(X, y)
    return lr

def svmResults(randomSeed, X, y, y_test, X_test, name):
    SVM = trainSVM(randomSeed, X, y)
    model_result_save(y_test,X_test,SVM, name)
    svm_scores = scores(name, SVM, y_test, X_test)
    return svm_scores

def nbResults(randomSeed, X, y, y_test, X_test, name):
    NB = trainNB(randomSeed, X, y)
    model_result_save(y_test, X_test, NB, name);
    nb_scores = scores(name, NB, y_test, X_test)
    return nb_scores

def rfResults(randomSeed, X, y, y_test, X_test, name):
    forest = trainRF(randomSeed, X, y)
    model_result_save(y_test,X_test,forest, name);
    forest_scores = scores(name, forest, y_test, X_test)
    return forest_scores

def lrResults(randomSeed, X, y, y_test, X_test, name):
    lr = trainLR(randomSeed, X, y)
    model_result_save(y_test, X_test, lr, name)
    lr_scores = scores(name, lr, y_test, X_test)
    return lr_scores

