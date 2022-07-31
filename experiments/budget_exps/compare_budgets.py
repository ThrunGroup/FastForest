from typing import Any
import pprint
import os

from experiments.datasets import data_loader

from experiments.exp_utils import *
from experiments.exp_constants import (
    BUDGET_REGRESSION,
    BUDGET_CLASSIFICATION,
    BUDGET_MAX_DEPTH,
    BUDGET_ALPHA_F,
    BUDGET_ALPHA_N,
    BUDGET_SAMPLE_SIZE,
    BUDGET_NUM_SEEDS,
)

from utils.constants import CLASSIFICATION_MODELS, REGRESSION_MODELS
from utils.constants import FLIGHT, AIR, APS, BLOG, SKLEARN_REGRESSION, MNIST_STR, COVTYPE, KDD, GPU
from utils.constants import (
    GINI,
    BEST,
    EXACT,
    MAB,
    MSE,
    DEFAULT_NUM_BINS,
    DEFAULT_ALPHA_F,
    DEFAULT_ALPHA_N,
)

from mnist import MNIST


# Classification #########################
# Vanilla random forest + H + GB + GBH
from data_structures.wrappers.histogram_random_forest_classifier import (
    HistogramRandomForestClassifier as HRFC,
)

# Extremely random forest + GB (already histogrammed)
from data_structures.wrappers.extremely_random_forest_classifier import (
    ExtremelyRandomForestClassifier as ERFC,
)

# Random patches + H + GB + GBH
from data_structures.wrappers.histogram_random_patches_classifier import (
    HistogramRandomPatchesClassifier as HRPC,
)

# Regression #########################
# Vanilla random forest + H + GB + GBH
from data_structures.wrappers.histogram_random_forest_regressor import (
    HistogramRandomForestRegressor as HRFR,
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
from data_structures.wrappers.histogram_random_patches_regressor import (
    HistogramRandomPatchesRegressor as HRPR,
)
from data_structures.wrappers.gradient_boosted_histogram_random_patches_regressor import (
    GradientBoostedHistogramRandomPatchesRegressor as GBHRPR,
)


def compare_budgets(
    compare: str = "HRFC",
    train_data: np.ndarray = None,
    train_targets: np.ndarray = None,
    original_test_data: np.ndarray = None,
    test_targets: np.ndarray = None,
    num_seeds: int = 1,
    predict: bool = True,
    run_theirs: bool = True,
    filename: str = None,
    verbose: bool = False,
    default_budget: int = None,  # Must be tuned per task
    depth_override: int = None,
    alpha_N_override: float = None,
    alpha_F_override: float = None,
):  # TODO(@motiwari) add return typehint
    assert filename is not None, "Need to pass filename_prefix"
    print("\n\n", "Running budget comparison for:", compare)
    # Query counts
    our_num_queries = []
    their_num_queries = []
    our_num_trees = []
    their_num_trees = []

    # For accuracies
    our_train_accs = []
    our_test_accs = []
    their_train_accs = []
    their_test_accs = []

    # params
    default_alpha_N = (
        alpha_N_override if alpha_N_override is not None else DEFAULT_ALPHA_N
    )
    default_alpha_F = (
        alpha_F_override if alpha_F_override is not None else DEFAULT_ALPHA_F
    )
    # Different from compare_runtimes
    default_max_depth = depth_override if depth_override is not None else 2
    default_n_estimators = 100
    default_min_samples_split = 2
    default_boosting_lr = 0.1

    # TODO(@motiwari): Change default min_impurity decrease for regression and classification
    default_min_impurity_decrease = 100  # New as opposed to compare_runtimes

    for seed in range(num_seeds):
        if compare == "HRFC":
            our_model = HRFC(
                data=train_data,
                labels=train_targets,
                n_estimators=default_n_estimators,
                max_depth=default_max_depth,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                num_bins=DEFAULT_NUM_BINS,
                min_samples_split=default_min_samples_split,
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
                min_impurity_decrease=default_min_impurity_decrease,
                max_leaf_nodes=None,
                budget=default_budget,
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
        our_num_trees.append(len(our_model.trees))
        our_num_queries.append(our_model.num_queries)
        print("Ours fitted", our_model.num_queries)
        print("Our Trees", len(our_model.trees))

        our_test_data = original_test_data
        our_measured_train_data = train_data
        their_test_data = original_test_data
        their_measured_train_data = train_data

        if run_theirs:
            their_model.fit()
            their_num_trees.append(len(their_model.trees))
            their_num_queries.append(their_model.num_queries)
            print("Theirs fitted", their_model.num_queries)
            print("Their Trees", len(their_model.trees))
            print()

        if compare in CLASSIFICATION_MODELS:
            is_classification = True
            our_train_acc = np.mean(
                our_model.predict_batch(our_measured_train_data)[0] == train_targets
            )
            if predict:
                our_test_acc = np.mean(
                    our_model.predict_batch(our_test_data)[0] == test_targets
                )
            if run_theirs:
                their_train_acc = np.mean(
                    their_model.predict_batch(their_measured_train_data)[0]
                    == train_targets
                )
                if predict:
                    their_test_acc = np.mean(
                        their_model.predict_batch(their_test_data)[0] == test_targets
                    )
        elif compare in REGRESSION_MODELS:
            is_classification = False
            our_train_acc = np.mean(
                (our_model.predict_batch(our_measured_train_data) - train_targets) ** 2
            )
            if predict:
                our_test_acc = np.mean(
                    (our_model.predict_batch(our_test_data) - test_targets) ** 2
                )
            if run_theirs:
                their_train_acc = np.mean(
                    (
                        their_model.predict_batch(their_measured_train_data)
                        - train_targets
                    )
                    ** 2
                )
                if predict:
                    their_test_acc = np.mean(
                        (their_model.predict_batch(their_test_data) - test_targets) ** 2
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

    # For queries
    our_avg_num_queries = np.mean(our_num_queries)
    our_std_num_queries = np.std(our_num_queries) / np.sqrt(num_seeds)

    our_avg_num_trees = np.mean(our_num_trees)
    our_std_num_trees = np.std(our_num_trees) / np.sqrt(num_seeds)

    if run_theirs:
        their_avg_train = np.mean(their_train_accs)
        their_std_train = np.std(their_train_accs) / np.sqrt(num_seeds)

        if predict:
            their_avg_test = np.mean(their_test_accs)
            their_std_test = np.std(their_test_accs) / np.sqrt(num_seeds)

        their_avg_num_queries = np.mean(their_num_queries)
        their_std_num_queries = np.std(their_num_queries) / np.sqrt(num_seeds)

        their_avg_num_trees = np.mean(their_num_trees)
        their_std_num_trees = np.std(their_num_trees) / np.sqrt(num_seeds)

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
        "our_num_queries": our_num_queries,
        "our_avg_num_queries": our_avg_num_queries,
        "our_std_num_queries": our_std_num_queries,
        "our_num_trees": our_num_trees,
        "our_avg_num_trees": our_avg_num_trees,
        "our_std_num_trees": our_std_num_trees,
        "their_num_queries": their_num_queries if run_theirs else None,
        "their_avg_num_queries": their_avg_num_queries if run_theirs else None,
        "their_std_num_queries": their_std_num_queries if run_theirs else None,
        "their_num_trees": their_num_trees if run_theirs else None,
        "their_avg_num_trees": their_avg_num_trees if run_theirs else None,
        "their_std_num_trees": their_std_num_trees if run_theirs else None,
    }
    print(f"Write a new {filename}")
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", filename), "w+") as fout:
        fout.write(str(results))
    return results


def main():
    pp = pprint.PrettyPrinter(indent=2)
    ############### Regression ###############
    for dataset in [GPU]:  # SKLEARN_REGRESSION, AIR
        if dataset == SKLEARN_REGRESSION or dataset == AIR:
            budget = BUDGET_REGRESSION * 32
        else:
            BUDGET_REGRESSION * 32  # Default

        train_data, train_targets, test_data, test_targets = data_loader.fetch_data(dataset)
        regression_models = ["HRFR", "HRPR", "ERFR"]
        for r_m in regression_models:
            pp.pprint(
                compare_budgets(
                    compare=r_m,
                    train_data=train_data,
                    train_targets=train_targets,
                    original_test_data=test_data,
                    test_targets=test_targets,
                    num_seeds=BUDGET_NUM_SEEDS,
                    predict=True,
                    run_theirs=True,
                    filename=dataset + "_" + r_m + "_dict",
                    verbose=True,
                    # TODO(@motiwari): May need to jiggle this. Was *12 for RP, *12 for ER, *10 for RF
                    default_budget=budget,
                    depth_override=BUDGET_MAX_DEPTH,
                )
            )

    ############### Classification ###############
    # for dataset in [FLIGHT, COVTYPE, APS]:  # MNIST_STR
    #     if dataset == COVTYPE or dataset == APS:
    #         budget = int(BUDGET_CLASSIFICATION * 3)
    #     elif dataset == FLIGHT:
    #         budget = int(BUDGET_CLASSIFICATION * 2.6)
    #     elif dataset == MNIST_STR:
    #         budget = int(BUDGET_CLASSIFICATION * 2.6)
    #     else:
    #         budget = int(BUDGET_CLASSIFICATION * 2.6)  # Default
    #
    #     train_images, train_labels, test_images, test_labels = data_loader.get_mnist()
    #     classification_models = ["HRFC", "HRPC", "ERFC"]
    #     for c_m in classification_models:
    #         pp.pprint(
    #             compare_budgets(
    #                 compare=c_m,
    #                 train_data=train_images,
    #                 train_targets=train_labels,
    #                 original_test_data=test_images,
    #                 test_targets=test_labels,
    #                 num_seeds=BUDGET_NUM_SEEDS,
    #                 predict=True,
    #                 run_theirs=True,
    #                 filename=dataset + "_" + c_m + "_dict",
    #                 verbose=True,
    #                 default_budget=budget,
    #                 depth_override=BUDGET_MAX_DEPTH,
    #             )
    #         )


if __name__ == "__main__":
    main()
