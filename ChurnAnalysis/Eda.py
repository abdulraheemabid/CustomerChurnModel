import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
from sklearn import linear_model
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split


class Eda:
    def __init__(self):
        pass

    @staticmethod
    def initiate(data):
        # First remove all the outliers.
        data = Eda.remove_outliers(data)

        # separate continuous and categorical variables
        continuous_features, categorical_features = Eda.separate_categorical_continuous_features(data)

        # analyze continuous feature's distribution
        Eda.analyze_continuous_features_distribution(data, continuous_features)

        # analyze categorical feature's distribution
        Eda.analyze_categorical_features_distribution(data, categorical_features)

        # change the dtype from int to categorical for all categorical features.
        data = Eda.convert_to_categorical_dtype(data, categorical_features)

        # Check categories distribution against label.
        # Eda.generate_count_plots(data)

        # do some feature engineering and select the final set of features we will use for ML.
        data = Eda.select_features(data)

        # finally return the data with selected features for ML.
        return data

    @staticmethod
    def select_features(data):
        # dropping the ID column as we will not need it now.
        data = data.drop(columns="id")

        # separating the label and features set,
        x = data.iloc[:, data.columns != "labels"].values
        y = data.iloc[:, data.columns == "labels"].values

        # dividing them into test and train data (70% - 30%),
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        # running a linear regression model to calculate the permutation importance of each feature against the label
        # for dimension reduction.
        reg = linear_model.LinearRegression()
        model = reg.fit(x_train, y_train)
        perm = PermutationImportance(reg, random_state=1).fit(x_train, y_train)

        # selecting those columns which do not increase the model's accuracy by more than 3%.
        non_important_cols = []
        for i in range(0, len(perm.feature_importances_)):
            if perm.feature_importances_[i] < 0.03:
                non_important_cols.append(data.columns[i])

        # remove the features identified unworthy.
        for feature in non_important_cols:
            data = data.drop(columns=feature)

        # below was a different approach of dimension reduction,
        # by removing those categorical features whose category frequency ratio was less than 10 %

        # data = Eda.remove_single_categorical_features(data)

        # features_to_remove = {key: value for (key, value) in frequency_ratios.items() if value < 10}
        # for feature in features_to_remove:
            # data = data.drop(columns=feature)

        # data = data.drop(columns="A2")
        # data = data.drop(columns="E")
        # data = data.drop(columns="B2")
        # data = data.drop(columns="A6")

        # finally return the data with selected features only.
        return data

    @staticmethod
    def remove_single_categorical_features(data):
        unique_count_series = data.nunique()
        single_categorical_vars = unique_count_series[unique_count_series == 1]
        for feature in single_categorical_vars.iteritems():
            data = data.drop(columns=feature[0])
        return data

    @staticmethod
    def convert_to_categorical_dtype(data, vars):
        for feature in vars.iteritems():
            data[feature[0]] = pd.Categorical(data[feature[0]])
        return data

    @staticmethod
    def separate_categorical_continuous_features(data):
        unique_count_series = data.nunique()
        continuous_features = unique_count_series[unique_count_series > 10]
        continuous_features = continuous_features.drop(labels=['id'])
        categorical_features = unique_count_series[unique_count_series <= 10]
        return continuous_features, categorical_features

    @staticmethod
    def analyze_continuous_features_distribution(data, continuous_features):
        rows, cols, size = Eda.get_matrix_size_for_plots(continuous_features, 2)

        fig, axs = plt.subplots(rows, cols)
        row_counter = 0
        col_counter = 0
        for feature in continuous_features.iteritems():
            # basic plot
            if rows == 1:
                axs[col_counter].boxplot(data[feature[0]])
                axs[col_counter].set_title(feature[0])
            else:
                axs[row_counter, col_counter].boxplot(data[feature[0]])
                axs[row_counter, col_counter].set_title(feature[0])

            col_counter += 1
            if col_counter > (cols-1):
                row_counter += 1
                col_counter = 0

            if rows == 1:
                axs[col_counter].hist(data[feature[0]], feature[1])
                axs[col_counter].set_title(feature[0])
            else:
                axs[row_counter, col_counter].hist(data[feature[0]], feature[1])
                axs[row_counter, col_counter].set_title(feature[0])
            # plt.show()

            col_counter += 1
            if col_counter > (cols - 1):
                row_counter += 1
                col_counter = 0

    @staticmethod
    def analyze_categorical_features_distribution(data, categorical_features):
        rows, cols, size = Eda.get_matrix_size_for_plots(categorical_features, 1)

        fig, axs = plt.subplots(rows, cols)
        row_counter = 0
        col_counter = 0
        frequency_ratios = {}
        for feature in categorical_features.iteritems():
            frequencies_of_categories = data[feature[0]].value_counts()
            frequency_ratio = (float(frequencies_of_categories.min()) / float(frequencies_of_categories.max())) * 100
            frequency_ratios[feature[0]] = frequency_ratio
            axs[row_counter, col_counter].hist(data[feature[0]], feature[1], edgecolor='white', linewidth=2)
            axs[row_counter, col_counter].set_title(str(feature[0]) + " - " + str(frequency_ratio) + "%")
            # plt.show()

            col_counter += 1
            if col_counter > (cols - 1):
                row_counter += 1
                col_counter = 0
        return frequency_ratios

    @staticmethod
    def generate_count_plots(data):
        for column in data.columns:
            sns.catplot(x=column, col="labels", data=data, kind="count", height=4, aspect=.7)

    @staticmethod
    def remove_outliers(data):
        # identified the outliers by calculating the z scores
        # and eliminating those rows where the absolute score was greater than 3
        return data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

    @staticmethod
    def get_matrix_size_for_plots(columns_series, number_of_charts):
        size = columns_series.size
        size = size * number_of_charts
        if size < 5:
            cols = size
            rows = 1
        else:
            cols = 5
            rows = int(math.ceil(size / 5.0))
        return rows, cols, size
