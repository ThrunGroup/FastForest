import time
from typing import Any, Tuple
import cProfile
import pstats

from experiments.exp_utils import *
from utils.constants import CLASSIFICATION_MODELS, REGRESSION_MODELS
from utils.constants import GINI, BEST, EXACT, MAB, MSE, DEFAULT_NUM_BINS

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
    full_train_data: np.ndarray = None,
    full_train_targets: np.ndarray = None,
    full_test_data: np.ndarray = None,
    full_test_targets: np.ndarray = None,
    starting_seed: int = 0,
    num_seeds: int = 1,
    predict: bool = True,
    run_theirs: bool = True,
    profile_name: str = "profile",
    C_SUBSAMPLE_SIZE: int = 60000,
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
        seed = (
            seed + starting_seed
        )  # Does not "skip" any in the loop above, loop condition precomputes range

        # Subsample data
        np.random.seed(seed)
        idcs = np.random.choice(60000, size=C_SUBSAMPLE_SIZE, replace=True)
        train_data = np.array(full_train_data)[idcs]
        train_targets = np.array(full_train_data)[idcs]

        if compare == "HRFC":
            our_model = HRFC(
                data=train_data,
                labels=train_targets,
                n_estimators=1,
                max_depth=1,
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
                n_estimators=1,
                max_depth=1,
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
            our_model = ERFC(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                num_bins=None,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                n_estimators=5,
                max_depth=5,
                num_bins=None,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                alpha_N=0.5,
                alpha_F=0.5,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                alpha_N=0.5,
                alpha_F=0.5,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                n_estimators=5,
                max_depth=5,
                num_bins=None,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                with_replacement=False,
                verbose=False,
            )
        elif compare == "GBERFR":
            our_model = GBERFR(
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
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=0.1,
            )
            their_model = GBERFR(
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
                with_replacement=False,
                verbose=False,
                boosting_lr=0.1,
            )
        elif compare == "HRFR":
            our_model = HRFR(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=0.1,
            )
            their_model = GBHRFR(
                data=train_data,
                labels=train_targets,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=0.1,
            )
        elif compare == "HRPR":
            our_model = HRPR(
                data=train_data,
                labels=train_targets,
                alpha_N=0.5,
                alpha_F=0.5,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                alpha_N=0.5,
                alpha_F=0.5,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
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
                alpha_N=0.5,
                alpha_F=0.5,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=MAB,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=0.1,
            )
            their_model = GBHRPR(
                data=train_data,
                labels=train_targets,
                alpha_N=0.5,
                alpha_F=0.5,
                n_estimators=5,
                max_depth=5,
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=2,
                min_impurity_decrease=0,
                max_leaf_nodes=None,
                budget=None,
                criterion=MSE,
                splitter=BEST,
                solver=EXACT,
                random_state=seed,
                with_replacement=False,
                verbose=False,
                boosting_lr=0.1,
            )
        else:
            raise NotImplementedError("Need to decide what models to compare")

        assert (
            "sklearn" not in their_model.__module__
        ), "Cannot use sklearn models for runtime comparisons"

        our_prof = cProfile.Profile()
        our_prof.enable()
        our_runtime = time_measured_fit(our_model)
        our_prof.disable()
        our_stats = pstats.Stats(our_prof).strip_dirs().sort_stats("tottime")
        our_stats.dump_stats(profile_name + "_ours_" + str(seed))

        our_train_times.append(our_runtime)
        print("Ours fitted", our_runtime)

        if run_theirs:
            their_prof = cProfile.Profile()
            their_prof.enable()
            their_runtime = time_measured_fit(their_model)
            their_prof.disable()
            their_stats = pstats.Stats(their_prof).strip_dirs().sort_stats("tottime")
            their_stats.dump_stats(profile_name + "_theirs_" + str(seed))

            their_train_times.append(their_runtime)
            print("Theirs fitted", their_runtime)
            print()

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
        print(f"(Ours) Train {metric}:", our_train_acc)
        our_train_accs.append(our_train_acc)
        if predict:
            print(f"(Ours) Test {metric}:", our_test_acc)
            our_test_accs.append(our_test_acc)
        print("*" * 30)
        print("(Ours) Runtime:", our_runtime)
        print("(Ours) Num queries:", our_model.num_queries)

        if run_theirs:
            print(f"(Theirs) Train {metric}:", their_train_acc)
            their_train_accs.append(their_train_acc)
            if predict:
                print(f"(Theirs) Test {metric}:", their_test_acc)
                their_test_accs.append(their_test_acc)
            print("-" * 30)
            print("(Theirs) Runtime:", their_runtime)

        print("/" * 30)

    # For accuracies
    our_avg_train = np.mean(our_train_accs)
    our_std_train = np.std(our_train_accs)

    if predict:
        our_avg_test = np.mean(our_test_accs)
        our_std_test = np.std(our_test_accs)

    # For runtimes
    our_avg_train_time = np.mean(our_train_times)
    our_std_train_time = np.std(our_train_times)

    if run_theirs:
        their_avg_train = np.mean(their_train_accs)
        their_std_train = np.std(their_train_accs)

        if predict:
            their_avg_test = np.mean(their_test_accs)
            their_std_test = np.std(their_test_accs)

        their_avg_train_time = np.mean(their_train_times)
        their_std_train_time = np.std(their_train_times)

        # See if confidence intervals overlap
        overlap = np.abs(their_avg_test - our_avg_test) < their_std_test + our_std_test
    return (
        overlap if run_theirs else None,
        our_avg_train,
        our_std_train,
        our_avg_test if predict else None,
        our_std_test if predict else None,
        their_avg_train if run_theirs else None,
        their_std_train if run_theirs else None,
        their_avg_test if run_theirs and predict else None,
        their_std_test if run_theirs and predict else None,
        our_avg_train_time,
        our_std_train_time,
        their_avg_train_time if run_theirs else None,
        their_std_train_time if run_theirs else None,
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

    ############### Classification
    mndata = MNIST("mnist/")

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # compare_runtimes("HRFC", train_images, train_labels, test_images, test_labels)
    # compare_runtimes("ERFC", train_images, train_labels, test_images, test_labels)
    # compare_runtimes("HRPC", train_images, train_labels, test_images, test_labels)

    ############### Regression
    # train_data, train_targets, test_data, test_targets = load_housing()
    # # Subsample the data because training on 20k points (the full housing dataset) takes too long for RFR
    # R_SUBSAMPLE_SIZE = 3000
    # train_data_subsampled = train_data[:R_SUBSAMPLE_SIZE]
    # train_targets_subsampled = train_targets[:R_SUBSAMPLE_SIZE]
    # print(len(train_data_subsampled), len(train_targets_subsampled))

    # compare_runtimes(
    #     "ERFR", train_data_subsampled, train_targets_subsampled, test_data, test_targets
    # )
    # compare_runtimes(
    #     "GBERFR",
    #     train_data_subsampled,
    #     train_targets_subsampled,
    #     test_data,
    #     test_targets,
    # )
    # compare_runtimes(
    #     "HRFR", train_data_subsampled, train_targets_subsampled, test_data, test_targets
    # )
    # compare_runtimes(
    #     "GBHRFR",
    #     train_data_subsampled,
    #     train_targets_subsampled,
    #     test_data,
    #     test_targets,
    # )
    # compare_runtimes(
    #     "HRPR", train_data_subsampled, train_targets_subsampled, test_data, test_targets
    # )
    # compare_runtimes(
    #     "GBHRPR",
    #     train_data_subsampled,
    #     train_targets_subsampled,
    #     test_data,
    #     test_targets,
    # )


if __name__ == "__main__":
    main()
