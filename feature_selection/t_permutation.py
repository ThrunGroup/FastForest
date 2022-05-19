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


def test_stability_with_budget(seed: int = 0) -> None:
    np.random.seed(seed)
    # digits = sklearn.datasets.load_digits()
    # data, labels = digits.data, digits.target
    diabetes = sklearn.datasets.load_diabetes()
    # data, labels = diabetes.data, diabetes.target
    data, labels = make_regression(100000, n_features=30, n_informative=3)
    num_forests = 5
    num_trees_per_feature = 20
    best_k_features = 10
    best_k_features = 2
    max_depth = 5
    max_leaf_nodes = 24
    feature_subsampling = None
    epsilon = 0.00
    budget = 5000000
    PI_exact = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=FOREST_UNIT_BUDGET_DIGIT,
        budget_per_forest=budget,
        solver=EXACT,
        #is_classification=False,
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
        "dataset": "digits",
        "budget": FOREST_UNIT_BUDGET_DIGIT,
        "num_forests": num_forests,
        "num_trees": num_trees_per_feature,
        "best_k": best_k_features,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "feature_subsampling": feature_subsampling,
        "epsilon": epsilon,
    }
    dir_name = "feature_selection/stability_log"
    log_filename = os.path.join(dir_name, "stability_log.csv")
    if not os.path.exists(log_filename):
        os.makedirs(dir_name, exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())


def run_stability_baseline_digits(
    seed: int = 0,
    num_trials: int = 10,
    num_forests: int = 5,
    max_depth: int = 3,
    num_trees_per_feature: int = 20,
    best_k_feature: int = 10,
) -> None:
    exact_sim_array = []
    mab_sim_array = []
    digits = sklearn.datasets.load_digits()
    data, labels = digits.data, digits.target
    rng = np.random.default_rng(seed)

    for trial in range(num_trials):
        print("TRIALS NUM: ", trial)
        exact_seed, mab_seed = rng.integers(0, MAX_SEED), rng.integers(0, MAX_SEED)
        exact = PermutationImportance(
            seed=exact_seed,
            data=data,
            labels=labels,
            max_depth=max_depth,
            num_forests=num_forests,
            num_trees_per_forest=num_trees_per_feature,
            budget_per_forest=FOREST_UNIT_BUDGET_DIGIT,
            solver=EXACT,
        )
        exact_sim_array.append(exact.run_baseline(best_k_feature))

        mab = PermutationImportance(
            seed=mab_seed,
            data=data,
            labels=labels,
            max_depth=max_depth,
            num_forests=num_forests,
            num_trees_per_forest=num_trees_per_feature,
            budget_per_forest=FOREST_UNIT_BUDGET_DIGIT,
            solver=MAB,
        )
        mab_sim_array.append(mab.run_baseline(best_k_feature))

    # compute confidence intervals
    exact_sim_array = np.asarray(exact_sim_array)
    e_avg = np.mean(exact_sim_array)
    e_std = np.std(exact_sim_array) / math.sqrt(num_trials)
    exact_CI = [e_avg - e_std, e_avg + e_std]

    mab_sim_array = np.asarray(mab_sim_array)
    m_avg = np.mean(mab_sim_array)
    m_std = np.std(mab_sim_array) / math.sqrt(num_trials)
    mab_CI = [m_avg - m_std, m_avg + m_std]

    print("confidence interval for exact: ", exact_CI)
    print("\n")
    print("confidence interval for mab: ", mab_CI)


if __name__ == "__main__":
    # test_contrived_dataset()
    test_stability_with_budget(2000000)
    # run_stability_baseline_digits()
