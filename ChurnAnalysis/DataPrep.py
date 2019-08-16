import pandas as pd
import numpy as np


class DataPrep:

    def __init__(self):
        pass

    @staticmethod
    def initiate():
        # read all csvs
        data, headers, labels = DataPrep.read_all_csv()

        # add headers in train data
        data = DataPrep.add_headers(data, headers)

        # add auto generated ID which might be useful onwards
        data = DataPrep.add_ids(data)

        # finally add the labels and make one consolidated csv
        data = DataPrep.add_labels(data, labels)

        # check for missing values. Note: did'nt find any.
        values_missing = DataPrep.check_if_missing_values(data)
        if values_missing:
            DataPrep.handle_missing_values()

        # make all the data types consistent. As we don't have any textual and floating type data,
        # we make sure all columns are integers.
        DataPrep.make_dtypes_consistent(data)

        # make sure by changing data types, we don't end up with missing values.
        values_missing = DataPrep.check_if_missing_values(data)

        if values_missing:
            DataPrep.handle_missing_values()

        # Finally return this clean and consolidated data set for further analysis.
        return data

    @staticmethod
    def add_ids(data):
        data['id'] = np.arange(len(data))
        return data

    @staticmethod
    def add_headers(data, headers):
        # headers are in separate csv file, so we fetch it, transpose it and assign it to pandas DF's columns
        transposed_headers = headers.transpose()
        data.columns = transposed_headers.iloc[0]
        return data

    @staticmethod
    def add_labels(data, labels):
        data['labels'] = labels.iloc[:, 0]
        return data

    @staticmethod
    def read_all_csv():
        # TODO: get these paths from config
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
        # we don't have any missing values in this case.
        # If we do in future, implement the mechanism to handle it here
        pass
