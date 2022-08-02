from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingTreeRegressor

from sklearn.ensemble import RandomForestClassifier as RFC_sklearn
from sklearn.ensemble import RandomForestRegressor as RFR_sklearn

from experiments.datasets import data_loader

from utils.constants import (
    FLIGHT,
    AIR,
    APS,
    BLOG,
    SKLEARN_REGRESSION,
    MNIST_STR,
    HOUSING,
    COVTYPE,
    KDD,
    GPU,
    BATCH_SIZE,
)


def main():
    SUBSAMPLE_SIZE = 60000
    X_train, y_train, X_test, y_test = data_loader.fetch_data(MNIST_STR)
    X_train = X_train[:SUBSAMPLE_SIZE]
    y_train = y_train[:SUBSAMPLE_SIZE]

    sklearn_RFC = RFC_sklearn(
        n_estimators=1,
        # defaults
        criterion="gini",  # default
        max_depth=None,  # default
        min_samples_split=2,  # default
        min_samples_leaf=1,  # default
        random_state=0,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )
    sklearn_RFC.fit(X_train, y_train)
    sklearn_RFC.predict(X_test)
    print(sklearn_RFC.score(X_test, y_test))

    # Hoeffding Tree for classification
    ht_classifier = HoeffdingTreeClassifier(
        max_byte_size=float("inf"),
        memory_estimate_period=float("inf"),
        grace_period=10,
        split_criterion="gini",
        split_confidence=0.1,
        # defaults
        tie_threshold=0.05,  # default
        binary_split=True,  # default
        stop_mem_management=False,  # default
        remove_poor_atts=False,  # default
        no_preprune=False,  # default
        leaf_prediction="nba",  # default
        nb_threshold=0,  # default
    )
    ht_classifier.fit(X_train, y_train)
    ht_classifier.predict(X_test)
    print(ht_classifier.score(X_test, y_test))
    print(ht_classifier.get_rules_description())

    # # Hoeffding Tree for regression
    # ht_regressor = HoeffdingTreeRegressor()
    # ht_regressor.fit(X, y)
    # ht_regressor.predict(X)
    # ht_regressor.predict_proba(X)
    # ht_regressor.score(X, y)


if __name__ == "__main__":
    main()
