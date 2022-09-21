from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


from src.models.model import Model

class Logistic_Regression(Model):
    def __init__(self, classes):
        self.classes = classes
        self.model = Pipeline([
                ('tfidf', CountVectorizer()),
                ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])


