import pandas as pd

class DataframeFormat():
    def convert_columns(self, data, categories):
        result_vector = []
        data = data['category'].split(',')
        for category in categories:
            if category in data:
                result_vector.append(1)
            else:
                result_vector.append(0)

        return result_vector