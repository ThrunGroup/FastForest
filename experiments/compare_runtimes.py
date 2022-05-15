import time
from typing import Any, Tuple

from experiments.exp_utils import *
from utils.constants import CLASSIFICATION_MODELS, REGRESSION_MODELS
from utils.constants import GINI, BEST, EXACT, MAB, MSE

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
from data_structures.wrappers.histogram_random_patches_regressor import (
    HistogramRandomPatchesRegressor as HBRPR,
)


def time_measured_fit(
    model: Any,
) -> float:
    """
    Returns wall clock time of training the model, in seconds.

    Has a side effect: trains the model.
    """
    start = time.time()
    model.fit()
    end = time.time()
    return end - start


def compare_runtimes(
    compare: str = "HRFC",
    train_data: np.ndarray = None,
    train_targets: np.ndarray = None,
    test_data: np.ndarray = None,
    test_targets: np.ndarray = None,
    num_seeds: int = 1,
) -> bool:
    # Runtimes
    our_train_times = []
    their_train_times = []

    # For accuracies
    our_train_accs = []
    our_test_accs = []
    their_train_accs = []
    their_test_accs = []
    for seed in range(num_seeds):
        # Ok to have n_jobs = -1 throughout?
        if compare == "HRFC":
            our_model = HRFC(
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
                solver=MAB,
                random_state=seed,
                verbose=False,
            )
            their_model = HRFC(
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
        elif compare == "ERFC":
            raise NotImplementedError("Need to decide what models to compare")
        elif compare == "HRPC":
            raise NotImplementedError("Need to decide what models to compare")
        elif compare == "ERFR":
            raise NotImplementedError("Need to decide what models to compare")
        elif compare == "GBERFR":
            raise NotImplementedError("Need to decide what models to compare")
        elif compare == "HRFR":
            raise NotImplementedError("Need to decide what models to compare")
        elif compare == "GBHRFR":
            raise NotImplementedError("Need to decide what models to compare")
        elif compare == "HRPR":
            raise NotImplementedError("Need to decide what models to compare")
        elif compare == "GBHRPR":
            raise NotImplementedError("Need to decide what models to compare")
        else:
            raise NotImplementedError("Need to decide what models to compare")

        assert (
            "sklearn" not in their_model.__module__
        ), "Cannot use sklearn models for runtime comparisons"

        our_runtime = time_measured_fit(our_model)
        our_train_times.append(our_runtime)
        print("Ours fitted", our_runtime)

        their_runtime = time_measured_fit(their_model)
        their_train_times.append(their_runtime)
        print("Theirs fitted", their_runtime)
        print()

        if compare in CLASSIFICATION_MODELS:
            our_train_acc = np.mean(
                our_model.predict_batch(train_data)[0] == train_targets
            )
            our_test_acc = np.mean(
                our_model.predict_batch(test_data)[0] == test_targets
            )
            their_train_acc = np.mean(
                their_model.predict_batch(train_data)[0] == train_targets
            )
            their_test_acc = np.mean(
                their_model.predict_batch(test_data)[0] == test_targets
            )
        elif compare in REGRESSION_MODELS:
            raise NotImplementedError("Need to decide what models to compare")
        else:
            raise Exception("Invalid model choice.")

        print("(Ours) Train accuracy:", our_train_acc)
        print("(Ours) Test accuracy:", our_test_acc)
        print("(Theirs) Train accuracy:", their_train_acc)
        print("(Theirs) Test accuracy:", their_test_acc)
        print("*" * 30)
        print("(Ours) Runtime:", our_runtime)
        print("(Theirs) Runtime:", their_runtime)
        print("-" * 30)

        our_train_accs.append(our_train_acc)
        our_test_accs.append(our_test_acc)
        their_train_accs.append(their_train_acc)
        their_test_accs.append(their_test_acc)

    # For accuracies
    our_avg_train = np.mean(our_train_accs)
    our_std_train = np.std(our_train_accs)

    our_avg_test = np.mean(our_test_accs)
    our_std_test = np.std(our_test_accs)

    their_avg_train = np.mean(their_train_accs)
    their_std_train = np.std(their_train_accs)

    their_avg_test = np.mean(their_test_accs)
    their_std_test = np.std(their_test_accs)

    # For runtimes
    our_avg_train_time = np.mean(our_train_times)
    our_std_train_time = np.std(our_train_times)

    their_avg_train_time = np.mean(their_train_times)
    their_std_train_time = np.std(their_train_times)

    # See if confidence intervals overlap
    overlap = np.abs(their_avg_test - our_avg_test) < their_std_test + our_std_test
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
        our_avg_train_time,
        our_std_train_time,
        their_avg_train_time,
        their_std_train_time,
    )


def main():
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

    # Classification
    mndata = MNIST("mnist/")

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = np.array(train_images)[:2000]
    train_labels = np.array(train_labels)[:2000]
    test_images = np.array(test_images)[:2000]
    test_labels = np.array(test_labels)[:2000]

    compare_runtimes("HRFC", train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    main()
