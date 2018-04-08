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

short_pos = open('pos.txt', 'r').read()
short_neg = open('neg.txt', 'r').read()

documents = []
all_words = []

# j is adjective, r is adverb, v is verb
allowed_word_types = ['J']

for p in short_pos.split('\n'):
    # appending the tuple: (review, sentiment) into documents
    documents.append((p, 'pos'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        # if the word is an adjective then add it into our word collection
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    # appending the tuple: (review, sentiment) into documents
    documents.append((p, 'neg'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        # if the word is an adjective then add it into our word collection
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

all_words = nltk.FreqDist(all_words)
#print((all_words.most_common(15)))
#print(all_words['stupid'])

# selecting the top 5k words as our feature words
word_features = list(all_words.keys())[:5000]

# saving documents
save_documents = open('documents.pickle', 'wb')
pickle.dump(documents, save_documents)
save_documents.close()

# saving word_features
save_word_features = open('word_features.pickle', 'wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# creating the feature set for each of our review and later saving it
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)
save_featuresets = open('featuresets.pickle', 'wb')
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

# splitting the featureset into training and testing
training_set = featuresets[:10000]
testing_set = featuresets[10000:]


# trainig and saving normal naive bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_f = open('naivebayes.pickle','wb')
pickle.dump(classifier, classifier_f)
classifier_f.close()
print('Original naive bayes algo accuracy:', (nltk.classify.accuracy(classifier, testing_set))*100)

# training and saving MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
MNB_f = open('MultinomialNB.pickle', 'wb')
pickle.dump(MNB_classifier, MNB_f)
MNB_f.close()
print('MNB_classifier accuracy:', (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# training and saving BernoulliNB
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
BNB_f = open('BernoulliNB.pickle', 'wb')
pickle.dump(BNB_classifier, BNB_f)
BNB_f.close()
print('BNB_classifier accuracy:', (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

# training and saving LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
LogisticRegression_f = open('LogisticRegression.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, LogisticRegression_f)
LogisticRegression_f.close()
print('LogisticRegression_classifier accuracy:', (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# training and saving LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
LinearSVC_f = open('LinearSVC.pickle', 'wb')
pickle.dump(LinearSVC_classifier, LinearSVC_f)
LinearSVC_f.close()
print('LinearSVC_classifier accuracy:', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# polling the results of all the classifier
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier,)
print('voted_classifier accuracy:', (nltk.classify.accuracy(voted_classifier, testing_set))*100)

























