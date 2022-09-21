import csv
import pandas as pd

from src.pre_processor.dataframe_format import DataframeFormat

class DatasetExploration():
    def __init__(self, path='../data/dataset.csv'):
        self.path = path
        self.dataframe = pd.read_csv(self.path)
        self.formatter = DataframeFormat()
        self.labels = list(self.list_classes())

    def show_classes(self):
        print(self.dataframe['category'].value_counts())
        print('-'*20)

    def show_head(self):
        print(self.dataframe.head())
        print('-'*20)

    def list_classes(self):
        list_of_labels = self.dataframe['category'].tolist()
        data = set([label for label in list_of_labels if (',' not in label)])
        return data

if __name__ == '__main__':
    data_exploration = DatasetExploration()
    data_exploration.show_classes()


    data_exploration.dataframe[data_exploration.labels] = data_exploration.dataframe.apply(
                                                            data_exploration.formatter.convert_columns,
                                                            args=[data_exploration.labels],
                                                            axis=1, result_type='expand')



    data_exploration.show_head()