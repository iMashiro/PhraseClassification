from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
stop_words = set(stopwords.words('portuguese'))

class NaiveBayes():
    def __init__(self):
        self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])

    def train(self, categories, train, test):
        for category in categories:
            print('Processing ' + str(category))
            self.pipeline.fit(train.sentence, train.category)
            prediction = self.pipeline.predict(test.sentence)
            print('Accuracy: ' + str(accuracy_score(test.category, prediction)))
