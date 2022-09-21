from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from src.models.model import Model

class Linear_SVC(Model):
    def __init__(self, classes):
        self.classes = classes
        self.model = Pipeline([
                ('tfidf', CountVectorizer()),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])


