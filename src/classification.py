import pandas as pd
from sklearn.model_selection import train_test_split

from src.pre_processor.pre_processor import PreProcessor
from src.models.naive_bayes import NaiveBayes

class Classification():
    def __init__(self):
        self.path = 'data/dataset.csv'
        self.dataframe = pd.read_csv(self.path)
        self.pre_processor = PreProcessor()

        self.naive_bayes = NaiveBayes()

        self.categories = ['finanças', 'educação', 'indústrias', 'varejo', 'orgão público']


if __name__ == '__main__':
    classification = Classification()
    train, test = train_test_split(classification.dataframe, random_state=42, test_size=0.3, shuffle=True)

    classification.naive_bayes.train(classification.categories, train, test)

    print(classification.naive_bayes.predict(['Curso de Técnico em Segurança do Trabalho por 32x R$ 161,03.']))