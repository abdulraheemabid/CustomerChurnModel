import matplotlib.pyplot as plt
import time


class Eda:
    def __init__(self):
        pass

    @staticmethod
    def initiate_eda(data):
        continuous_features, categorical_features = Eda.separate_categorical_continuous_features(data)
        print(continuous_features)
        print(categorical_features)
        Eda.analyze_continuous_features_distribution(data, continuous_features)

    @staticmethod
    def separate_categorical_continuous_features(data):
        unique_count_series = data.nunique()
        continuous_features = unique_count_series[unique_count_series > 2]
        continuous_features = continuous_features.drop(labels=['id'])
        categorical_features = unique_count_series[unique_count_series <= 2]
        return continuous_features, categorical_features

    @staticmethod
    def analyze_continuous_features_distribution(data, continuous_features):
        rows, cols = Eda.get_matrix_size_for_plots(continuous_features)
        fig, axs = plt.subplots(rows, cols)
        time.sleep(5)
        row_counter = 0
        col_counter = 0
        for feature in continuous_features.iteritems():
            # basic plot
            axs[row_counter, col_counter].boxplot(data[feature[0]])
            axs[row_counter, col_counter].set_title(feature[0])

            col_counter += 1
            if col_counter > (cols-1):
                row_counter += 1
                col_counter = 0


    @staticmethod
    def get_matrix_size_for_plots(columns_series):
        size = columns_series.size
        if (size % 2) == 0:
            return size/2, size/2
        else:
            rows = (size / 2) + 1
            cols = size - rows
            return rows, cols
