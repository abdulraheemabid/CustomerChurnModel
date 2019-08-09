import pandas as pd
from DataPrep import DataPrep

class Eda:
    def __init__(self):
        pass

    @staticmethod
    def initiate_eda(data):
        values_missing = Eda.check_if_missing_values(data)
        if values_missing:
            DataPrep.handle_missing_values()
        Eda.make_dtypes_consistent(data)
        values_missing = Eda.check_if_missing_values(data)
        if values_missing:
            DataPrep.handle_missing_values()

    @staticmethod
    def check_if_missing_values(data):
        return data.isnull().values.any()

    @staticmethod
    def make_dtypes_consistent(data):
        data = data.astype(int)
        print(data.dtypes)
