import sklearn.datasets
import numpy as np
import math
import os
import pandas as pd

from permutation import PermutationImportance
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from utils.constants import (
    EXACT,
    MAB,
    FOREST_UNIT_BUDGET_DIGIT,
    JACCARD,
    SPEARMAN,
    KUNCHEVA,
    FOREST_UNIT_BUDGET_DIABETES,
    MAX_SEED,
)
from experiments.heart.fit_heart import append_dict_as_row


def test_contrived_dataset() -> None:
    # contrived dataset where the first three features are most important
    X, Y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )
    PI = PermutationImportance(
        data=X,
        labels=Y,
        num_forests=10,
        num_trees_per_forest=10,
    )
    results = PI.get_importance_array()
    print("importance array", results)
    print("stability", PI.get_stability(results))


def test_stability_with_budget(seed: int = 1) -> None:
    np.random.seed(seed)
    # digits = sklearn.datasets.load_digits()
    # data, labels = digits.data, digits.target
    diabetes = sklearn.datasets.load_diabetes()
    # data, labels = diabetes.data, diabetes.target
    data_size=10000
    num_features=40
    num_informative=5
    data, labels = make_regression(n_samples=data_size, n_features=num_features, n_informative=num_informative)
    num_forests = 5
    num_trees_per_feature = 20
    best_k_features = 7
    max_depth = 5
    max_leaf_nodes = 24
    feature_subsampling = "SQRT"
    epsilon = 0.00
    budget = 500000
    PI_exact = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=budget,
        solver=EXACT,
        is_classification=False,
        feature_subsampling=feature_subsampling,
        max_leaf_nodes=max_leaf_nodes,
        epsilon=epsilon,
    )
    stability_exact = PI_exact.run_baseline(best_k_features)
    print("stability for exact", stability_exact)
    print("\n\n")

    PI_mab = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=budget,
        solver=MAB,
        is_classification=False,
        feature_subsampling=feature_subsampling,
        max_leaf_nodes=max_leaf_nodes,
        epsilon=epsilon,
    )
    stability_mab = PI_mab.run_baseline(best_k_features)
    print("stability for mab", stability_mab)

    if stability_mab > stability_exact:
        print("MAB IS MORE STABLE!!!!")
    else:
        print("EXACT is more stable :((")

    log_dict = {
        "stability_diff": stability_mab - stability_exact,
        "dataset": "make_regression",
        "data_size": data_size,
        "budget": budget,
        "num_forests": num_forests,
        "num_trees": num_trees_per_feature,
        "best_k": best_k_features,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "feature_subsampling": feature_subsampling,
        "epsilon": epsilon,
        "seed": seed,
        "num_features": num_features,
        "num_informative": num_informative

    }
    dir_name = "stability_log"
    log_filename = os.path.join(dir_name, "stability_log.csv")
    if not os.path.exists(log_filename):
        os.makedirs(dir_name, exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())


def run_stability_make_regressions(
    seed: int = 0,
    data_size=10000,
    num_features=40,
    num_informative=5,
    num_trials: int = 30,
    num_forests: int = 5,
    max_depth: int = 8,
    max_leaf_nodes=50,
    num_trees_per_feature: int = 20,
    feature_subsampling: str = "SQRT",
    best_k_feature: int = 7,
    epsilon=0.0,
    budget=800000,
) -> None:
    exact_stab_array = []
    mab_stab_array = []
    data, labels = make_regression(
        n_samples=data_size, n_features=num_features, n_informative=num_informative
    )
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    for trial in range(num_trials):
        print("TRIALS NUM: ", trial)
        exact_seed, mab_seed = rng.integers(0, MAX_SEED), rng.integers(0, MAX_SEED)
        exact = PermutationImportance(
            seed=exact_seed,
            data=data,
            labels=labels,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            num_forests=num_forests,
            num_trees_per_forest=num_trees_per_feature,
            budget_per_forest=budget,
            solver=EXACT,
            feature_subsampling=feature_subsampling,
            epsilon=epsilon,
            is_classification=False,
        )
        exact_stab_array.append(exact.run_baseline(best_k_feature))

        mab = PermutationImportance(
            seed=mab_seed,
            data=data,
            labels=labels,
            max_depth=max_depth,
            num_forests=num_forests,
            num_trees_per_forest=num_trees_per_feature,
            budget_per_forest=budget,
            solver=MAB,
            feature_subsampling=feature_subsampling,
            epsilon=epsilon,
            is_classification=False,
        )
        mab_stab_array.append(mab.run_baseline(best_k_feature))

    # compute confidence intervals
    conf_multiplier = 3
    exact_stab_array = np.asarray(exact_stab_array)
    e_avg = np.mean(exact_stab_array)
    e_std = np.std(exact_stab_array) / math.sqrt(num_trials)
    exact_CI = [e_avg - e_std * conf_multiplier, e_avg + e_std * conf_multiplier]

    mab_sim_array = np.asarray(mab_stab_array)
    m_avg = np.mean(mab_sim_array)
    m_std = np.std(mab_sim_array) / math.sqrt(num_trials)
    mab_CI = [m_avg - m_std * conf_multiplier, m_avg + m_std * conf_multiplier]
    is_overlap = exact_CI[1] >= mab_CI[0]

    print("confidence interval for exact: ", exact_CI)
    print("\n")
    print("confidence interval for mab: ", mab_CI)
    log_dict = {
        "stability_diff": m_avg - e_avg,
        "mab stability": m_avg,
        "is_overlap": is_overlap,
        "dataset": "make_regression",
        "budget": budget,
        "num_forests": num_forests,
        "num_trees": num_trees_per_feature,
        "best_k": best_k_feature,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "feature_subsampling": feature_subsampling,
        "epsilon": epsilon,
        "seed": seed,
    }
    dir_name = "stat_test_stability_log"
    log_filename = os.path.join(dir_name, "statistics_log.csv")
    if not os.path.exists(log_filename):
        os.makedirs(dir_name, exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())

#
# def run_stability_baseline_digits(
#     seed: int = 0,
#     num_trials: int = 10,
#     num_forests: int = 5,
#     max_depth: int = 3,
#     num_trees_per_feature: int = 20,
#     best_k_feature: int = 10,
# ) -> None:
#     exact_sim_array = []
#     mab_sim_array = []
#     digits = sklearn.datasets.load_digits()
#     data, labels = digits.data, digits.target
#     rng = np.random.default_rng(seed)
#
#     for trial in range(num_trials):
#         print("TRIALS NUM: ", trial)
#         exact_seed, mab_seed = rng.integers(0, MAX_SEED), rng.integers(0, MAX_SEED)
#         exact = PermutationImportance(
#             seed=exact_seed,
#             data=data,
#             labels=labels,
#             max_depth=max_depth,
#             num_forests=num_forests,
#             num_trees_per_forest=num_trees_per_feature,
#             budget_per_forest=FOREST_UNIT_BUDGET_DIGIT,
#             solver=EXACT,
#         )
#         exact_sim_array.append(exact.run_baseline(best_k_feature))
#
#         mab = PermutationImportance(
#             seed=mab_seed,
#             data=data,
#             labels=labels,
#             max_depth=max_depth,
#             num_forests=num_forests,
#             num_trees_per_forest=num_trees_per_feature,
#             budget_per_forest=FOREST_UNIT_BUDGET_DIGIT,
#             solver=MAB,
#         )
#         mab_sim_array.append(mab.run_baseline(best_k_feature))
#
#     # compute confidence intervals
#     exact_sim_array = np.asarray(exact_sim_array)
#     e_avg = np.mean(exact_sim_array)
#     e_std = np.std(exact_sim_array) / math.sqrt(num_trials)
#     exact_CI = [e_avg - e_std, e_avg + e_std]
#
#     mab_sim_array = np.asarray(mab_sim_array)
#     m_avg = np.mean(mab_sim_array)
#     m_std = np.std(mab_sim_array) / math.sqrt(num_trials)
#     mab_CI = [m_avg - m_std, m_avg + m_std]
#
#     print("confidence interval for exact: ", exact_CI)
#     print("\n")
#     print("confidence interval for mab: ", mab_CI)
#     log_dict = {
#         "stability_diff": stability_mab - stability_exact,
#         "dataset": "digits",
#         "budget": FOREST_UNIT_BUDGET_DIGIT,
#         "num_forests": num_forests,
#         "num_trees": num_trees_per_feature,
#         "best_k": best_k_features,
#         "max_depth": max_depth,
#         "max_leaf_nodes": max_leaf_nodes,
#         "feature_subsampling": feature_subsampling,
#         "epsilon": epsilon,
#     }
#     dir_name = "stability_log"
#     log_filename = os.path.join(dir_name, "stability_log.csv")
#     if not os.path.exists(log_filename):
#         os.makedirs(dir_name, exist_ok=True)
#         df = pd.DataFrame(columns=log_dict.keys())
#         df.to_csv(log_filename, index=False)
#     append_dict_as_row(log_filename, log_dict, log_dict.keys())


if __name__ == "__main__":
    # test_contrived_dataset()
    test_stability_with_budget(22)
    run_stability_make_regressions(231)
    # run_stability_baseline_digits()
