import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset_exploration.dataset_exploration import DatasetExploration
from src.pre_processor.pre_processor import PreProcessor
from src.pre_processor.dataframe_format import DataframeFormat
from src.models.naive_bayes import NaiveBayes
from src.models.logistic_regression import Logistic_Regression
from src.models.linear_svc import Linear_SVC


class Classification():
    def __init__(self):
        self.path = 'data/dataset.csv'
        self.dataframe = pd.read_csv(self.path)
        self.categories = ['finanças', 'educação', 'indústrias', 'varejo', 'orgão público']

        self.pre_processor = PreProcessor()
        self.formatter = DataframeFormat()
        self.exploration = DatasetExploration(self.path)

        self.naive_bayes = NaiveBayes(self.categories)
        self.logistic_regression = Logistic_Regression(self.categories)
        self.linear_svc = Linear_SVC(self.categories)

    def test_case(self, model, name, sentence):
        print('-'*20)
        print('Test Case: ' + name)
        print(sentence)
        print('Labels: ' + str(self.categories))
        print(model.predict(sentence))



if __name__ == '__main__':
    classification = Classification()
    dataframe = classification.dataframe.copy()

    dataframe['processed_sentence'] = dataframe['sentence'].apply(classification.pre_processor.pipeline)
    dataframe[classification.categories] = dataframe.apply(classification.formatter.convert_columns,
                                                            args=[classification.categories],
                                                            axis=1, result_type='expand')

    X_train, X_test, y_train, y_test = train_test_split(dataframe['processed_sentence'],
                                                            dataframe[classification.categories],
                                                            test_size=0.3, random_state=42)

    classification.naive_bayes.train(X_train, X_test, y_train, y_test)
    classification.logistic_regression.train(X_train, X_test, y_train, y_test)
    classification.linear_svc.train(X_train, X_test, y_train, y_test)

    test_sentence = ['Curso de Técnico em Segurança do Trabalho por 32x R$ 161,03.']

    classification.test_case(classification.naive_bayes.model, 'Naive Bayes', test_sentence)
    classification.test_case(classification.logistic_regression.model, 'Logistic Regression', test_sentence)
    classification.test_case(classification.linear_svc.model, 'Linear SVC', test_sentence)

