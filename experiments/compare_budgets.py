import cProfile
import pstats

import time
from typing import Any
import pprint

from sklearn.datasets import load_diabetes, make_classification, make_regression


from experiments.exp_utils import *
from utils.constants import CLASSIFICATION_MODELS, REGRESSION_MODELS
from utils.constants import (
    GINI,
    BEST,
    EXACT,
    MAB,
    MSE,
    DEFAULT_NUM_BINS,
    DEFAULT_MIN_IMPURITY_DECREASE,
)

from mnist import MNIST


# Classification #
# Vanilla random forest + H + GB + GBH
from data_structures.wrappers.random_forest_classifier import (
    RandomForestClassifier as RFC,
)
from data_structures.wrappers.histogram_random_forest_classifier import (
    HistogramRandomForestClassifier as HRFC,
)
from data_structures.wrappers.gradient_boosted_random_forest_classifier import (
    GradientBoostedRandomForestClassifier as GBRFC,
)
from data_structures.wrappers.gradient_boosted_histogram_random_forest_classifier import (
    GradientBoostedHistogramRandomForestClassifier as GBHRFC,
)

# Extremely random forest + GB (already histogrammed)
from data_structures.wrappers.extremely_random_forest_classifier import (
    ExtremelyRandomForestClassifier as ERFC,
)
from data_structures.wrappers.gradient_boosted_extremely_random_forest_classifier import (
    GradientBoostedExtremelyRandomForestClassifier as GBERFC,
)

# Random patches + H + GB + GBH
from data_structures.wrappers.random_patches_classifier import (
    RandomPatchesClassifier as RPC,
)
from data_structures.wrappers.histogram_random_patches_classifier import (
    HistogramRandomPatchesClassifier as HRPC,
)
from data_structures.wrappers.gradient_boosted_random_patches_classifier import (
    GradientBoostedRandomPatchesClassifier as GBRPC,
)
from data_structures.wrappers.histogram_random_patches_classifier import (
    HistogramRandomPatchesClassifier as HBRPC,
)


# Regression #
# Vanilla random forest + H + GB + GBH
from data_structures.wrappers.random_forest_regressor import (
    RandomForestRegressor as RFR,
)
from data_structures.wrappers.histogram_random_forest_regressor import (
    HistogramRandomForestRegressor as HRFR,
)
from data_structures.wrappers.gradient_boosted_random_forest_regressor import (
    GradientBoostedRandomForestRegressor as GBRFR,
)
from data_structures.wrappers.gradient_boosted_histogram_random_forest_regressor import (
    GradientBoostedHistogramRandomForestRegressor as GBHRFR,
)

# Extremely random forest + GB (already histogrammed)
from data_structures.wrappers.extremely_random_forest_regressor import (
    ExtremelyRandomForestRegressor as ERFR,
)
from data_structures.wrappers.gradient_boosted_extremely_random_forest_regressor import (
    GradientBoostedExtremelyRandomForestRegressor as GBERFR,
)

# Random patches + H + GB + GBH
from data_structures.wrappers.random_patches_regressor import (
    RandomPatchesRegressor as RPR,
)
from data_structures.wrappers.histogram_random_patches_regressor import (
    HistogramRandomPatchesRegressor as HRPR,
)
from data_structures.wrappers.gradient_boosted_random_patches_regressor import (
    GradientBoostedRandomPatchesRegressor as GBRPR,
)
from data_structures.wrappers.gradient_boosted_histogram_random_patches_regressor import (
    GradientBoostedHistogramRandomPatchesRegressor as GBHRPR,
)
from data_structures.wrappers.histogram_random_patches_regressor import (
    HistogramRandomPatchesRegressor as HBRPR,
)


def compare_budgets(
    compare: str = "HRFC",
    train_data: np.ndarray = None,
    train_targets: np.ndarray = None,
    test_data: np.ndarray = None,
    test_targets: np.ndarray = None,
    num_seeds: int = 1,
    predict: bool = True,
    run_theirs: bool = True,
    filename: str = None,
    verbose: bool = False,
):  # TODO(@motiwari) add return typehint
    assert filename is not None, "Need to pass filename_prefix"
    print("\n\n", "Running comparison for:", compare)

    # Runtimes
    our_train_times = []
    their_train_times = []

    # For accuracies
    our_train_accs = []
    our_test_accs = []
    their_train_accs = []
    their_test_accs = []

    # params
    default_alpha_N = 0.5
    default_alpha_F = 0.5
    default_max_depth = 5
    default_n_estimators = 5
    default_min_samples_split = 2
    default_boosting_lr = 0.1

    for seed in range(num_seeds):
        if compare == "HRFC":
            our_model = HRFC(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                verbose=False,
            )
            their_model = HRFC(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                verbose=False,
            )
        elif compare == "ERFC":
            our_model = ERFC(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=None,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
            their_model = ERFC(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=None,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
        elif compare == "HRPC":
            our_model = HRPC(
                data=train_data,
                labels=train_targets,
                alpha_N=default_alpha_N,
                alpha_F=default_alpha_F,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
            their_model = HRPC(
                data=train_data,
                labels=train_targets,
                alpha_N=default_alpha_N,
                alpha_F=default_alpha_F,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=GINI,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
        elif compare == "ERFR":
            our_model = ERFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=None,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
            their_model = ERFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=None,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
        elif compare == "GBERFR":
            our_model = GBERFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=None,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=default_boosting_lr,
            )
            their_model = GBERFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=None,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=default_boosting_lr,
            )
        elif compare == "HRFR":
            our_model = HRFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
            their_model = HRFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
        elif compare == "GBHRFR":
            our_model = GBHRFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=default_boosting_lr,
            )
            their_model = GBHRFR(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=default_boosting_lr,
            )
        elif compare == "HRPR":
            our_model = HRPR(
                data=train_data,
                labels=train_targets,
                alpha_N=default_alpha_N,
                alpha_F=default_alpha_F,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
            their_model = HRPR(
                data=train_data,
                labels=train_targets,
                alpha_N=default_alpha_N,
                alpha_F=default_alpha_F,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
            )
        elif compare == "GBHRPR":
            our_model = GBHRPR(
                data=train_data,
                labels=train_targets,
                alpha_N=default_alpha_N,
                alpha_F=default_alpha_F,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=default_boosting_lr,
            )
            their_model = GBHRPR(
                data=train_data,
                labels=train_targets,
                alpha_N=default_alpha_N,
                alpha_F=default_alpha_F,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=default_boosting_lr,
            )
        else:
            raise NotImplementedError("Need to decide what models to compare")

        assert (
            "sklearn" not in their_model.__module__
        ), "Cannot use sklearn models for runtime comparisons"

        our_model.fit()

        if run_theirs:
            their_model.fit()

        if compare in CLASSIFICATION_MODELS:
            is_classification = True
            our_train_acc = np.mean(
                our_model.predict_batch(train_data)[0] == train_targets
            )
            if predict:
                our_test_acc = np.mean(
                    our_model.predict_batch(test_data)[0] == test_targets
                )
            if run_theirs:
                their_train_acc = np.mean(
                    their_model.predict_batch(train_data)[0] == train_targets
                )
                if predict:
                    their_test_acc = np.mean(
                        their_model.predict_batch(test_data)[0] == test_targets
                    )
        elif compare in REGRESSION_MODELS:
            is_classification = False
            our_train_acc = np.mean(
                (our_model.predict_batch(train_data) - train_targets) ** 2
            )
            if predict:
                our_test_acc = np.mean(
                    (our_model.predict_batch(test_data) - test_targets) ** 2
                )
            if run_theirs:
                their_train_acc = np.mean(
                    (their_model.predict_batch(train_data) - train_targets) ** 2
                )
                if predict:
                    their_test_acc = np.mean(
                        (their_model.predict_batch(test_data) - test_targets) ** 2
                    )
        else:
            raise Exception("Invalid model choice.")

        metric = "accuracy" if is_classification else "MSE"
        if verbose:
            print(f"(Ours) Train {metric}:", our_train_acc)
        our_train_accs.append(our_train_acc)
        if predict:
            if verbose:
                print(f"(Ours) Test {metric}:", our_test_acc)
            our_test_accs.append(our_test_acc)
        if verbose:
            print("*" * 30)
            print("(Ours) Num queries:", our_model.num_queries)

        if run_theirs:
            if verbose:
                print(f"(Theirs) Train {metric}:", their_train_acc)
            their_train_accs.append(their_train_acc)
            if predict:
                if verbose:
                    print(f"(Theirs) Test {metric}:", their_test_acc)
                their_test_accs.append(their_test_acc)
            if verbose:
                print("-" * 30)
                print("(Theirs) Num queries:", their_model.num_queries)

        print("/" * 30)

    # For accuracies
    our_avg_train = np.mean(our_train_accs)
    our_std_train = np.std(our_train_accs) / np.sqrt(num_seeds)

    if predict:
        our_avg_test = np.mean(our_test_accs)
        our_std_test = np.std(our_test_accs) / np.sqrt(num_seeds)

    if run_theirs:
        their_avg_train = np.mean(their_train_accs)
        their_std_train = np.std(their_train_accs) / np.sqrt(num_seeds)

        if predict:
            their_avg_test = np.mean(their_test_accs)
            their_std_test = np.std(their_test_accs) / np.sqrt(num_seeds)

        their_avg_train_time = np.mean(their_train_times)
        their_std_train_time = np.std(their_train_times) / np.sqrt(num_seeds)

        # See if confidence intervals overlap
        overlap = np.abs(their_avg_test - our_avg_test) < their_std_test + our_std_test

    results = {
        "overlap": overlap if run_theirs else None,
        "our_train_accs": our_train_accs,
        "our_avg_train": our_avg_train,
        "our_std_train": our_std_train,
        "our_test_accs": our_test_accs if predict else None,
        "our_avg_test": our_avg_test if predict else None,
        "our_std_test": our_std_test if predict else None,
        "their_train_accs": their_train_accs if run_theirs else None,
        "their_avg_train": their_avg_train if run_theirs else None,
        "their_std_train": their_std_train if run_theirs else None,
        "their_test_accs": their_test_accs if run_theirs and predict else None,
        "their_avg_test": their_avg_test if run_theirs and predict else None,
        "their_std_test": their_std_test if run_theirs and predict else None,
        "our_train_times": our_train_times,
        "our_avg_train_time": our_avg_train_time,
        "our_std_train_time": our_std_train_time,
        "their_train_times": their_train_times if run_theirs else None,
        "their_avg_train_time": their_avg_train_time if run_theirs else None,
        "their_std_train_time": their_std_train_time if run_theirs else None,
    }
    with open(filename, "w+") as fout:
        fout.write(str(results))

    return results


def main():
    ################## LIST OF MODELS
    # Classification
    ## Extremely Random -- already histogrammed
    # ERFC

    ## Forest
    # HRFC

    ## Patches
    # HRPC

    # Regression
    ## Extremely Random -- already histogrammed
    # ERFR
    # GBERFR

    ## Forest
    # HRFR
    # GBHRFR

    ## Patches
    # HRPR
    # GBHRPR

    # Not implemented
    # GBERFC
    # GBHRFC
    # GBHRPC

    # Can't because no binning
    # RFC
    # RPC
    # GBRFC (also not implemented)
    # GBRPC (also not implemented)

    # RFR
    # RPR
    # GBRFR
    # GBRPR

    ########################################### PARAMS
    pp = pprint.PrettyPrinter(indent=2)
    NUM_SEEDS = 5

    ############### Regression
    # train_data, train_targets, test_data, test_targets = load_housing()
    # # Subsample the data because training on 20k points (the full housing dataset) takes too long for RFR
    #
    # train_data, train_targets = make_huge(train_data, train_targets)
    # train_data_subsampled = train_data
    # train_targets_subsampled = train_targets
    # print(len(train_data_subsampled), len(train_targets_subsampled))

    params = {
        "data_size": 1000000,
        "n_features": 50,
        "informative_ratio": 0.06,
        "seed": 1,
        "epsilon": 0.01,
        "use_dynamic_epsilon": False,
        "use_logarithmic split point": True,
    }

    n_informative = int(params["n_features"] * params["informative_ratio"])
    full_data, full_targets = make_regression(
        params["data_size"],
        n_features=params["n_features"],
        n_informative=n_informative,
        random_state=params["seed"],
    )

    train_test_split = int(0.8 * params["data_size"])
    train_data = full_data[:train_test_split]
    train_targets = full_targets[:train_test_split]

    test_data = full_data[train_test_split:]
    test_targets = full_targets[train_test_split:]

    ## Random Forests
    pp.pprint(
        compare_runtimes(
            compare="HRFR",
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="HRFR_dict",
            verbose=True,
        )
    )

    pp.pprint(
        compare_runtimes(
            compare="GBHRFR",
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="GBHRFR_dict",
            verbose=True,
        )
    )

    ## Extremely Random Forests
    pp.pprint(
        compare_runtimes(
            compare="ERFR",
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="ERFR_dict",
            verbose=True,
        )
    )

    pp.pprint(
        compare_runtimes(
            compare="GBERFR",
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="GBERFR_dict",
            verbose=True,
        )
    )

    ## Random Patches
    pp.pprint(
        compare_runtimes(
            compare="HRPR",
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="HRPR_dict",
            verbose=True,
        )
    )

    pp.pprint(
        compare_runtimes(
            compare="GBHRPR",
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="GBHRPR_dict",
            verbose=True,
        )
    )

    ############### Classification
    mndata = MNIST("mnist/")

    train_images, train_labels = mndata.load_training()
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images, test_labels = mndata.load_testing()
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    ## Random Forests
    pp.pprint(
        compare_runtimes(
            compare="HRFC",
            train_data=train_images,
            train_targets=train_labels,
            test_data=test_images,
            test_targets=test_labels,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="HRFC_dict",
            verbose=True,
        )
    )

    ## Extremely Random Forests
    pp.pprint(
        compare_runtimes(
            compare="ERFC",
            train_data=train_images,
            train_targets=train_labels,
            test_data=test_images,
            test_targets=test_labels,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="ERFC_dict",
            verbose=True,
        )
    )

    ## Random Patches
    pp.pprint(
        compare_runtimes(
            compare="HRPC",
            train_data=train_images,
            train_targets=train_labels,
            test_data=test_images,
            test_targets=test_labels,
            num_seeds=NUM_SEEDS,
            predict=True,
            run_theirs=True,
            filename="HRPC_dict",
            verbose=True,
        )
    )


if __name__ == "__main__":
    main()
