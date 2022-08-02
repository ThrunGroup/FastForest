from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingTreeRegressor

from sklearn.ensemble import RandomForestClassifier as RFC_sklearn
from sklearn.ensemble import RandomForestRegressor as RFR_sklearn

from experiments.datasets import data_loader

from utils.constants import FLIGHT, AIR, APS, BLOG, SKLEARN_REGRESSION, MNIST_STR, HOUSING, COVTYPE, KDD, GPU

def main():
    SUBSAMPLE_SIZE = 1000
    X_train, y_train, X_test, y_test = data_loader.fetch_data(MNIST_STR)
    X_train = X_train[:SUBSAMPLE_SIZE]
    y_train = y_train[:SUBSAMPLE_SIZE]

    sklearn_RFC = RFC_sklearn(n_estimators=1, max_depth=None, min_samples_split=2, random_state=0)
    sklearn_RFC.fit(X_train, y_train)
    sklearn_RFC.predict(X_test)
    print(sklearn_RFC.score(X_test, y_test))

    # Hoeffding Tree for classification
    ht_classifier = HoeffdingTreeClassifier()
    ht_classifier.fit(X_train, y_train)
    ht_classifier.predict(X_test)
    print(ht_classifier.score(X_test, y_test))

    # # Hoeffding Tree for regression
    # ht_regressor = HoeffdingTreeRegressor()
    # ht_regressor.fit(X, y)
    # ht_regressor.predict(X)
    # ht_regressor.predict_proba(X)
    # ht_regressor.score(X, y)

if __name__ == '__main__':
    main()