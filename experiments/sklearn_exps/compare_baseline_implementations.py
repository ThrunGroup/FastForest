import os
import ast

from sklearn.ensemble import RandomForestClassifier as RFC_sklearn
from sklearn.ensemble import ExtraTreesClassifier as ERFC_sklearn

from sklearn.ensemble import RandomForestRegressor as RFR_sklearn
from sklearn.ensemble import ExtraTreesRegressor as ERFR_sklearn
from sklearn.ensemble import GradientBoostingRegressor as GBRFR_sklearn

from experiments.exp_utils import *
from utils.constants import GINI, BEST, EXACT, MSE

from data_structures.wrappers.random_forest_classifier import (
    RandomForestClassifier as RFC_ours,
)
from data_structures.wrappers.extremely_random_forest_classifier import (
    ExtremelyRandomForestClassifier as ERFC_ours,
)


from data_structures.wrappers.random_forest_regressor import (
    RandomForestRegressor as RFR_ours,
)
from data_structures.wrappers.extremely_random_forest_regressor import (
    ExtremelyRandomForestRegressor as ERFR_ours,
)
from data_structures.wrappers.gradient_boosted_random_forest_regressor import (
    GradientBoostedRandomForestRegressor as GBRFR_ours,
)


def compare_accuracies(
    compare: str = "RFC",
    train_data: np.ndarray = None,
    train_targets: np.ndarray = None,
    test_data: np.ndarray = None,
    test_targets: np.ndarray = None,
    num_seeds: int = 20,
) -> bool:
    our_train_accs = []
    our_test_accs = []
    their_train_accs = []
    their_test_accs = []
    for seed in range(num_seeds):
        # Ok to have n_jobs = -1 throughout?
        if compare == "RFC":
            our_model = RFC_ours(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                verbose=False,
            )
            their_model = RFC_sklearn(
                n_estimators=5,
                criterion="gini",
                max_depth=5,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                n_jobs=-1,
                random_state=seed,
                verbose=0,
            )
        elif compare == "ERFC":
            our_model = ERFC_ours(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                num_bins=1,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                verbose=False,
            )
            their_model = ERFC_sklearn(
                n_estimators=5,
                criterion="gini",
                max_depth=5,
                min_samples_split=2,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=False,
                n_jobs=-1,
                random_state=seed,
                verbose=0,
            )
        elif compare == "RFR":
            our_model = RFR_ours(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                verbose=False,
            )
            their_model = RFR_sklearn(
                n_estimators=5,
                # criterion="squared_error",
                max_depth=5,
                min_samples_split=2,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                n_jobs=-1,
                random_state=seed,
                verbose=0,
            )
        elif compare == "ERFR":
            our_model = ERFR_ours(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                num_bins=1,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                verbose=False,
            )
            their_model = ERFR_sklearn(
                n_estimators=5,
                # criterion="squared_error",
                max_depth=5,
                min_samples_split=2,
                max_features="auto",
                min_impurity_decrease=0.0,
                bootstrap=False,
                n_jobs=-1,
                random_state=seed,
                verbose=False,
            )
        elif compare == "GBRFR":
            our_model = GBRFR_ours(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                bootstrap=False,  # Override for RFR, since sklearn GBR doesn't support bootstrapping
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                with_replacement=False,
                boosting_lr=0.1,
                random_state=seed,
                verbose=False,
            )
            their_model = GBRFR_sklearn(
                n_estimators=5,
                # loss="squared_error",
                max_depth=5,
                learning_rate=0.1,
                # criterion="squared_error",
                min_samples_split=2,
                min_samples_leaf=1,
                min_impurity_decrease=0.0,
                max_features="sqrt",
                random_state=seed,
                verbose=False,
            )

        elif compare == "GBHRFR":
            # TODO(@motiwari): fill out
            our_model = ERFR_ours(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                num_bins=None,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                verbose=False,
            )
            their_model = ERFR_sklearn(
                n_estimators=1,
                # criterion="squared_error",
                max_depth=5,
                min_samples_split=2,
                max_features="auto",
                min_impurity_decrease=0.0,
                bootstrap=False,
                n_jobs=-1,
                random_state=seed,
                verbose=False,
            )
        else:
            raise NotImplementedError("Need to decide what models to compare")

        our_model.fit()
        their_model.fit(train_data, train_targets)

        if compare == "RFC" or compare == "ERFC":
            is_classification = True
            our_train_acc = np.mean(
                our_model.predict_batch(train_data)[0] == train_targets
            )
            our_test_acc = np.mean(
                our_model.predict_batch(test_data)[0] == test_targets
            )
            their_train_acc = np.mean(their_model.predict(train_data) == train_targets)
            their_test_acc = np.mean(their_model.predict(test_data) == test_targets)
        elif (
            compare == "RFR"
            or compare == "ERFR"
            or compare == "GBRFR"
            or compare == "GBHRFR"
        ):
            is_classification = False
            our_train_acc = np.mean(
                (our_model.predict_batch(train_data) - train_targets) ** 2
            )
            our_test_acc = np.mean(
                (our_model.predict_batch(test_data) - test_targets) ** 2
            )
            their_train_acc = np.mean(
                (their_model.predict(train_data) - train_targets) ** 2
            )
            their_test_acc = np.mean(
                (their_model.predict(test_data) - test_targets) ** 2
            )

        metric = "accuracy" if is_classification else "MSE"
        print("Trial", seed)
        print(f"(Ours) Train {metric}:", our_train_acc)
        print(f"(Ours) Test {metric}:", our_test_acc)
        print(f"(Theirs) Train {metric}:", their_train_acc)
        print(f"(Theirs) Test {metric}:", their_test_acc)
        print("-" * 30)

        our_train_accs.append(our_train_acc)
        our_test_accs.append(our_test_acc)
        their_train_accs.append(their_train_acc)
        their_test_accs.append(their_test_acc)

    our_avg_train = np.mean(our_train_accs)
    our_std_train = np.std(our_train_accs)

    our_avg_test = np.mean(our_test_accs)
    our_std_test = np.std(our_test_accs)

    their_avg_train = np.mean(their_train_accs)
    their_std_train = np.std(their_train_accs)

    their_avg_test = np.mean(their_test_accs)
    their_std_test = np.std(their_test_accs)

    # See if confidence intervals overlap
    overlap = np.abs(their_avg_test - our_avg_test) < their_std_test + our_std_test
    results = {
        "overlap": overlap,
        "our_train_accs": our_train_accs,
        "our_avg_train": our_avg_train,
        "our_std_train": our_std_train,
        "our_test_accs": our_test_accs,
        "our_avg_test": our_avg_test,
        "our_std_test": our_std_test,
        "their_train_accs": their_train_accs,
        "their_avg_train": their_avg_train,
        "their_std_train": their_std_train,
        "their_test_accs": their_test_accs,
        "their_avg_test": their_avg_test,
        "their_std_test": their_std_test,
    }
    filename = str(compare) + "_dict"
    if os.path.exists(filename):
        with open(filename, "r+") as fin:
            prev_results = ast.literal_eval(fin.read())
            print(f"prev_results: {prev_results}")
            if prev_results == results:
                print(f"{filename} is successfully reproduced")
                return results
    print(f"Write a new {filename}")
    with open(filename, "w+") as fout:
        fout.write(str(results))
    return (
        overlap,
        our_avg_train,
        our_std_train,
        our_avg_test,
        our_std_test,
        their_avg_train,
        their_std_train,
        their_avg_test,
        their_std_test,
    )


def main():
    # Classification
    pca_train_vecs, train_labels, pca_test_vecs, test_labels, classes = load_pca_ng()
    # print("Performing Experiment: Random Forest Classifier")
    # print(
    #     compare_accuracies(
    #         "RFC", pca_train_vecs, train_labels, pca_test_vecs, test_labels
    #     )
    # )

    print("Performing Experiment: Extremely Random Forest Classifier")
    print(
        compare_accuracies(
            "ERFC", pca_train_vecs, train_labels, pca_test_vecs, test_labels
        )
    )

    # Regression
    train_data, train_targets, test_data, test_targets = load_housing()
    # Subsample the data because training on 20k points (the full housing dataset) takes too long for RFR
    train_data_subsampled = train_data[:1000]
    train_targets_subsampled = train_targets[:1000]

    print("Performing Experiment: Random Forest Regression")
    print(
        compare_accuracies(
            "RFR",
            train_data_subsampled,
            train_targets_subsampled,
            test_data,
            test_targets,
        )
    )

    print("Performing Experiment: Extremely Random Forest Regression")
    print(
        compare_accuracies("ERFR", train_data, train_targets, test_data, test_targets)
    )


if __name__ == "__main__":
    main()
