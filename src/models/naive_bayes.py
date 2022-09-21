from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

class NaiveBayes():
    def __init__(self, classes):
        self.classes = classes
        self.model = Pipeline([
                ('tfidf', CountVectorizer()),
                ('clf', OneVsRestClassifier(MultinomialNB(), n_jobs=1)),
            ])

    def train(self, x_train, x_test, y_train, y_test):
        for category in self.classes:
            print('Training... ' + str(category))
            self.model.fit(x_train, y_train)
            prediction = self.model.predict(x_test)
            print('Accuracy: ' + str(accuracy_score(y_test, prediction)))

        print(classification_report(y_test, prediction, target_names=self.classes))
