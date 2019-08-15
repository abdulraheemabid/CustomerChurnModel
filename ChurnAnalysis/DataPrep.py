from ConfigReader import ConfigReader
import pandas as pd
import numpy as np


class DataPrep:

    def __init__(self):
        pass

    @staticmethod
    def initiate():
        data, headers, labels = DataPrep.read_all_csv()
        data = DataPrep.add_headers(data, headers)
        data = DataPrep.add_ids(data)
        data = DataPrep.add_labels(data, labels)

        values_missing = DataPrep.check_if_missing_values(data)
        if values_missing:
            DataPrep.handle_missing_values()

        DataPrep.make_dtypes_consistent(data)
        values_missing = DataPrep.check_if_missing_values(data)

        if values_missing:
            DataPrep.handle_missing_values()

        return data

    @staticmethod
    def add_ids(data):
        data['id'] = np.arange(len(data))
        return data

    @staticmethod
    def add_headers(data, headers):
        transposed_headers = headers.transpose()
        data.columns = transposed_headers.iloc[0]
        return data

    @staticmethod
    def add_labels(data, labels):
        data['labels'] = labels.iloc[:, 0]
        return data

    @staticmethod
    def read_all_csv():
        # data_path = ConfigReader.readconfig("data-locations", "churn-data-file-path")
        # headers_path = ConfigReader.readconfig("data-locations", "headers-file-path")
        # label_path = ConfigReader.readconfig("data-locations", "labels-file-path")

        data = pd.read_csv("..\\Data\\train.csv", header=None)
        headers_data = pd.read_csv("..\\Data\\train_col_name.csv", header=None)
        label_data = pd.read_csv("..\\Data\\train_lable.csv", header=None)

        return data, headers_data, label_data

    @staticmethod
    def check_if_missing_values(data):
        return data.isnull().values.any()

    @staticmethod
    def make_dtypes_consistent(data):
        data = data.astype(int)

    @staticmethod
    def handle_missing_values():
        pass
