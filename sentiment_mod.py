import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

# polling the results of all the classifiers
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # returning the sentiment of the sentence
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # returning the confidence value of the sentiment produced
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf*100


# loading word_features
load_word_features = open('word_features.pickle', 'rb')
word_features = pickle.load(load_word_features)
load_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# loading normal naive bayes
classifier_f = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

# loading MultinomialNB
MNB_f = open('MultinomialNB.pickle', 'rb')
MNB_classifier = pickle.load(MNB_f)
MNB_f.close()

# loading BernoulliNB
BNB_f = open('BernoulliNB.pickle', 'rb')
BNB_classifier = pickle.load(BNB_f)
BNB_f.close()

# loading LogisticRegression
LogisticRegression_f = open('LogisticRegression.pickle', 'rb')
LogisticRegression_classifier = pickle.load(LogisticRegression_f)
LogisticRegression_f.close()

# loading LinearSVC
LinearSVC_f = open('LinearSVC.pickle', 'rb')
LinearSVC_classifier = pickle.load(LinearSVC_f)
LinearSVC_f.close()

# polling the results of all the classifier
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier,)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

























