import csv
import pandas as pd

class DatasetExploration():
    def __init__(self):
        self.path = '../data/dataset.csv'
        self.dataframe = pd.read_csv(self.path)

    def show_classes(self, dataframe):
        print(dataframe['category'].value_counts())

if __name__ == '__main__':
    data_exploration = DatasetExploration()
    data_exploration.show_classes(data_exploration.dataframe)