from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA


class Model:
    def __init__(self):
        pass

    @staticmethod
    def initiate(data):
        # Trying different techniques to further reduce the features and find the best accurate model.
        # RandomForest has been used with each technique
        Model.train_and_test_model(data)
        Model.train_and_test_using_pca(data)
        Model.train_and_test_using_pca_with_multiple_components(data)
        Model.train_model_with_forward_feature_selection(data)
        Model.train_model_with_reverse_feature_selection_method(data)
        data = Model.reduce_cols_identified_by_feature_addition_reduction(data)
        Model.train_and_test_model(data)

    @staticmethod
    def train_and_test_using_pca(data):
        # separating the labels and dividing data into training and testing
        x = data.iloc[:, data.columns != "labels"].values
        y = data.iloc[:, data.columns == "labels"].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        # making a principle component which describes 95 percent of the variance in all features.
        pca = PCA(.95)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

        # training and testing the model
        regressor = RandomForestRegressor(n_estimators=4, random_state=0)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)

        # printing the evaluations metrics.
        print("confusion matrix:")
        print(confusion_matrix(y_test, y_pred.round()))
        print("classification report:")
        print(classification_report(y_test, y_pred.round()))
        print("accuracy:")
        print(accuracy_score(y_test, y_pred.round()))

    @staticmethod
    def train_and_test_using_pca_with_multiple_components(data):
        # trying out same model with different number of principle components,
        # Turns out there is no major change if we increase the components as well
        accuracy_with_pca_variance = {}
        for i in range(1, 10):
            x = data.iloc[:, data.columns != "labels"].values
            y = data.iloc[:, data.columns == "labels"].values
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

            pca = PCA(n_components=i)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)

            regressor = RandomForestRegressor(n_estimators=4, random_state=0)
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)

            print("confusion matrix:")
            print(confusion_matrix(y_test, y_pred.round()))
            print("classification report:")
            print(classification_report(y_test, y_pred.round()))
            print("accuracy:")
            print(accuracy_score(y_test, y_pred.round()))
            accuracy_with_pca_variance[i] = accuracy_score(y_test, y_pred.round())
        print(accuracy_with_pca_variance)

    @staticmethod
    def train_and_test_model(data):
        x = data.iloc[:, data.columns != "labels"].values
        y = data.iloc[:, data.columns == "labels"].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        regressor = RandomForestRegressor(n_estimators=4, random_state=0)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)

        print("confusion matrix:")
        print(confusion_matrix(y_test, y_pred.round()))
        print("classification report:")
        print(classification_report(y_test, y_pred.round()))
        print("accuracy:")
        print(accuracy_score(y_test, y_pred.round()))

    @staticmethod
    def train_model_with_forward_feature_selection(data):
        # training the random forest by starting with 1 feature and adding features one by one
        # and checking on which number of features, the accuracy is highest.
        accuracy_for_features = {}
        for i in range(1, (len(data.columns) - 1)):
            x = data.iloc[:, data.columns != "labels"]
            x = data.iloc[:, 0:i].values
            y = data.iloc[:, data.columns == "labels"].values
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

            regressor = RandomForestRegressor(n_estimators=4, random_state=0)
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)

            print("confusion matrix:")
            print(confusion_matrix(y_test, y_pred.round()))
            print("classification report:")
            print(classification_report(y_test, y_pred.round()))
            print("accuracy:")
            print(accuracy_score(y_test, y_pred.round()))
            accuracy_for_features[i] = accuracy_score(y_test, y_pred.round())
        print(accuracy_for_features)

    @staticmethod
    def train_model_with_reverse_feature_selection_method(data):
        # training the random forest by starting with all feature and removing features one by one
        # and checking on which number of features, the accuracy is highest.
        accuracy_for_features = {}
        for i in range(0, len(data.columns)-2):
            x = data.iloc[:, data.columns != "labels"]
            x = data.iloc[:, i:len(data.columns)-2].values
            y = data.iloc[:, data.columns == "labels"].values
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

            regressor = RandomForestRegressor(n_estimators=4, random_state=0)
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)

            print("confusion matrix:")
            print(confusion_matrix(y_test, y_pred.round()))
            print("classification report:")
            print(classification_report(y_test, y_pred.round()))
            print("accuracy:")
            print(accuracy_score(y_test, y_pred.round()))
            accuracy_for_features[i] = accuracy_score(y_test, y_pred.round())
        print(accuracy_for_features)

    @staticmethod
    def reduce_cols_identified_by_feature_addition_reduction(data):
        # by running the forward and reverse feature selection method,
        # we are now removing the features identified by them.
        finalSet = data.iloc[:, 6:9]
        finalSet["labels"] = data["labels"]
        print(finalSet.columns)
        return finalSet