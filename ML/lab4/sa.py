## Machine Learning
#  Lab assignment 4 | Spam Classification using logistic regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started with this
#  assignment. You will need to complete
#
#   emailFeatures in sa_utils.py
#
#  and develop the functions
#
#   train
#   predict
#   (can add them to sa_utils.py)
#
#  together with other code snippets in the current file; 
#  all identified with TODO

## Initialization
import numpy as np
import scipy.io
from sa_utils import *
import matplotlib.pyplot as plt
from os import listdir
import sys, getopt
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn import svm
from sklearn.model_selection import cross_validate

def get_args():
    def help():
        print("Example usage: python sa.py -c <float: regularization> -a <string: algorythm>")
        sys.exit(-1)

    result = {}
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:a:h:")
        for opt, arg in opts:
            if opt == '-c':
                result['C'] = float(arg) if arg else 1
            if opt in ("-a",):
                print(opt, arg)
                result['m'] = arg
    except getopt.GetoptError:
        print("WRONG ARGUMENTS!:")
        help()
        sys.exit(2)
    print(result)
    return result

## ==================== Part 1: Email Preprocessing ====================
#  To classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. Y

def test_preprocessing():
    print('\nPreprocessing sample email (samples/email1.txt)\n')
    # Extract Features
    file_contents = readFile('samples/email1.txt')

    print('Original email')
    print(file_contents)

    word_indices = processEmail(file_contents)
    print('Processed email')
    # Print Stats
    print('Word Indices: ', word_indices)
    

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.
def test_extract_features():
    print('\nExtracting features from sample email ( samples/email1.txt)\n')
    print('(this may take 1 to 2 secs) ...\n' % ())
    # Extract Features
    file_contents = readFile('samples/spam1.txt')
    word_indices = processEmail(file_contents)
    features = emailFeatures(word_indices)
    # Print Stats
    print('Length of feature vector: %d\n' % len(features))
    print('Number of non-zero entries: %d\n' % sum(features > 0))

    #pause()

## =========== Part 3: Train a classifier ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment

def get_classifier(classifier_code='lsvm', C=1):
    """ selects a classifier

    Args:
        classifier_code (str): code for the classifier.
        C (float): regulatization factor
    """
    classifier = None
    if classifier_code=='lr':
        classifier = LogisticRegression( max_iter=10000, C=C)
        print('Selected LOGISTIC REGRETION Classifier with C: {}'.format(C))
    elif classifier_code=='lsvm':
        classifier = svm.SVC(kernel='linear',C=C)
        print('Selected Linear SUPORT VECTOR MACHINE Classifier with C: {}'.format(C))
    elif classifier_code=='ssvm':
        classifier = svm.SVC(kernel='sigmoid',C=C)
        print('Selected Sigmoid SUPORT VECTOR MACHINE Classifier with C: {}'.format(C))
    elif classifier_code=='rbfsvm':
        classifier = svm.SVC(kernel='rbf',C=C)
        print('Selected RBF SUPORT VECTOR MACHINE Classifier with C: {}'.format(C))
    elif classifier_code=='b':
        classifier = naive_bayes.GaussianNB()
    return classifier

def predict(model, X):
    return model.predict(X)

def train_classifier(classifier, X, y):
    """ For a given classifier, performs crossvalidation for different C values 
    and writes the result to a CSV file

    Args:
        classifier_code (str): code for the classifier.
        X (ndarry): array (n, n_features)
        y (ndarray): array with the known output

    Returns:
        tuple: model, accuracy of the model on the training data
    """
    model = classifier.fit(X, y)     # TODO
    p = model.predict(X)      # TODO

    raw_accuracy =  np.mean(p == y)*100
    print('Training Test Accuracy: ', raw_accuracy)

    plt.plot(y, 'x', color='blue')
    plt.plot(p, '.', color='yellow')
    plt.xlim(0,100)
    #plt.show()
    return model, raw_accuracy

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
def test_classifier(model):
    test_data = scipy.io.loadmat('spamTest.mat')
    Xtest = test_data['Xtest']
    ytest =  test_data['ytest'].reshape(-1,)
    print('\nEvaluating the trained classifier on a test set ...\n')
    p_test = model.predict(Xtest)
    #print('Test Accuracy: %f\n', (mean(float(p == ytest)) * 100))
    accuracy = np.mean(p_test == ytest)*100
    print('Test Accuracy: ', np.mean(p_test == ytest)*100)

    return accuracy

## ================= Part 5: Top Predictors of Spam ====================
#  We can inspect the weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.

# Sort the weights and obtain the vocabulary list
def top_predictors_of_spam(model):
    vocabList = getVocabList()
    def get_best_indicators_of_spam():
        l = []
        for i, coef in enumerate(model.coef_[0]):
            if coef > 0.1:
                l.append([round(coef,2), vocabList[i]])
        return sorted(l, reverse=True)

    print('\nBest indicators of Spam: {}\n\n'.format( get_best_indicators_of_spam()[:10]))

    pause()
## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included samples/spam1.txt,
#  spamSample2.txt,  samples/email1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
#  samples/email1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
def extract_features(filename):
    """ Reads email from file and extracts features
    Args:
        filename (str): path to file.
    Returns:
        tuple: features, known y output
    """
    file_contents = readFile(filename)
    word_indices = processEmail(file_contents)
    x = emailFeatures(word_indices)
    spam = 'spam' in filename #0 not spam, 1 spam
    #print(filename, spam, p,  spam and p[0])
    return x, spam

def get_external_data_X_and_y(model):
    filenames = ['samples/{}'.format(f) for f in listdir('samples/')] 
    print('(1 indicates spam, 0 indicates not spam)')
    X = np.array([])
    y = np.array([]) 
    for filename in filenames:
        x, known_y = extract_features(filename)
        p = model.predict([x])
        print(filename, known_y, p)
        X = np.vstack((X, x)) if X.size else x
        y = np.append(y, int(known_y))
    return X, y

def test_with_external_data(model):
    X, y = get_external_data_X_and_y(model)
    p = model.predict(X)
    print(p)
    accuracy = np.mean(p==y)  
    print('Test accuracy for external data: {}'.format(accuracy))  
    print(cross_validate(model, X, y, cv=4))


##--------------------- MAIN ---------------------
def get_model(classifier_code, X, y, C):
    classifier = get_classifier(classifier_code, C)
    return train_classifier(classifier, X, y)

def estimate_C(classifier_code, X, y):
    """ For a given classifier, performs crossvalidation for different C values 
    and writes the result to a CSV file

    Args:
        classifier_code (str): code for the classifier.
        X (ndarry): array (n, n_features)
        y (ndarray): array with the known output
    """
    print('Estimating C parameter. Might take several minutes...')
    with open('{}.csv'.format(classifier_code), 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter='\t')
        for c in np.geomspace(0.0000001,10000):
            classifier = get_classifier(classifier_code, c)
            cv_results = cross_validate(classifier, X, y, cv=10)
            model, training_accuracy = get_model(classifier_code, X, y,c)
            test_accuracy = test_classifier(model)
            w.writerow([c, training_accuracy/100, test_accuracy/100, np.mean(cv_results['test_score'])])    



if __name__ == "__main__":
    args = get_args()
    classifier_code = args.get('m', 'lr')
    C = args.get('C')


    print(classifier_code)
    training_data = scipy.io.loadmat('spamTrain.mat')
    X = training_data['X']
    y =  training_data['y'].reshape(-1,)
    print(np.mean(y), np.std(y))

    test_preprocessing()
    test_extract_features()
    estimate_C(classifier_code, X, y)
    model, _ = get_model(classifier_code, X, y, C)
    test_classifier(model)
    #top_predictors_of_spam(model)
    test_with_external_data(model)

