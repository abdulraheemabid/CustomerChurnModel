from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA


class Model:
    def __init__(self):
        pass

    @staticmethod
    def initiate(data):
        print("data col len: " + str(len(data.columns)))
        Model.train_and_test_model(data)
        Model.train_model_and_use_feature_addition_method(data)
        Model.train_model_and_use_feature_remove_method(data)

        data = Model.reduce_cols_identified_by_feature_addition_reduction(data)

        Model.train_and_test_model(data)

        Model.train_and_test_using_pca(data)
        Model.train_and_test_using_pca_with_multiple_variance(data)

        print("done")


    @staticmethod
    def train_and_test_using_pca(data):
        x = data.iloc[:, data.columns != "labels"].values
        y = data.iloc[:, data.columns == "labels"].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        pca = PCA(.95)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

        regressor = RandomForestRegressor(n_estimators=4, random_state=0)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)

        # print(confusion_matrix(y_test, y_pred.round()))
        # print(classification_report(y_test, y_pred.round()))
        print(accuracy_score(y_test, y_pred.round()))

    @staticmethod
    def train_and_test_using_pca_with_multiple_variance(data):

        accuracy_with_pca_variance = {}
        for i in range(1, 5):
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

            # print(confusion_matrix(y_test, y_pred.round()))
            # print(classification_report(y_test, y_pred.round()))
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

        # print(confusion_matrix(y_test, y_pred.round()))
        # print(classification_report(y_test, y_pred.round()))
        print(accuracy_score(y_test, y_pred.round()))

    @staticmethod
    def train_model_and_use_feature_addition_method(data):
        accuracy_for_features = {}
        for i in range(1, (len(data.columns) - 1)):
            x = data.iloc[:, data.columns != "labels"]
            x = data.iloc[:, 0:i].values
            y = data.iloc[:, data.columns == "labels"].values
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

            regressor = RandomForestRegressor(n_estimators=4, random_state=0)
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)

            #print(confusion_matrix(y_test, y_pred.round()))
            #print(classification_report(y_test, y_pred.round()))
            #print(accuracy_score(y_test, y_pred.round()))
            accuracy_for_features[i] = accuracy_score(y_test, y_pred.round())
        print(accuracy_for_features)

    @staticmethod
    def train_model_and_use_feature_remove_method(data):
        accuracy_for_features = {}
        for i in range(0, len(data.columns)-2):
            x = data.iloc[:, data.columns != "labels"]
            x = data.iloc[:, i:len(data.columns)-2].values
            y = data.iloc[:, data.columns == "labels"].values
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

            regressor = RandomForestRegressor(n_estimators=4, random_state=0)
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)

            # print(confusion_matrix(y_test, y_pred.round()))
            # print(classification_report(y_test, y_pred.round()))
            # print(accuracy_score(y_test, y_pred.round()))
            accuracy_for_features[i] = accuracy_score(y_test, y_pred.round())
        print(accuracy_for_features)

    @staticmethod
    def reduce_cols_identified_by_feature_addition_reduction(data):
        finalSet = data.iloc[:, 6:9]
        finalSet["labels"] = data["labels"]
        print(finalSet.columns)
        return finalSet


