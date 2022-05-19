import sklearn.datasets
import numpy as np
import math
import os
import pandas as pd

from permutation import PermutationImportance
from utils.constants import (
    MAB,
    EXACT,
    FOREST_UNIT_BUDGET_DIGIT,
    FOREST_UNIT_BUDGET_DIABETES,
    MAX_SEED,
)


def test_stability_with_budget_digit(
    seed: int,
    max_depth: int = 3,
    num_forests: int = 5,
    num_trees_per_feature: int = 20,
    best_k_features: int = 10,
) -> None:
    np.random.seed(seed)
    digits = sklearn.datasets.load_digits()
    data, labels = digits.data, digits.target

    exact = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=FOREST_UNIT_BUDGET_DIGIT,
        solver=EXACT,
    )
    stability_exact = exact.run_baseline(best_k_features)

    mab = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=FOREST_UNIT_BUDGET_DIGIT,
        solver=MAB,
    )
    stability_mab = mab.run_baseline(best_k_features)
    assert stability_mab > stability_exact, "MAB is NOT more stable for classification"


def test_stability_with_budget_regression(
    seed: int = 0,
    max_depth = 5,
    max_leaf_nodes = 24,
    num_forests = 5,
    num_trees_per_feature = 20,
    feature_subsampling = "SQRT",
    epsilon = 0.00,
    best_k_features = 2,
    budget = FOREST_UNIT_BUDGET_REGRESSION,
) -> None:
    np.random.seed(seed)
    data, labels = sklearn.datasets.make_regression(10000, n_features=100, n_informative=10)
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


def run_stability_baseline_digits(
    seed: int,
    num_trials: int = 10,
    max_depth: int = 3,
    num_forests: int = 5,
    num_trees_per_forest: int = 20,
    best_k_feature: int = 10,
) -> None:
    mab_sim_array = []
    exact_sim_array = []
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
            num_trees_per_forest=num_trees_per_forest,
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
            num_trees_per_forest=num_trees_per_forest,
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

    # print results
    print("exact CIs: ", exact_CI)
    print("mab CIs: ", mab_CI)
    assert (
        exact_CI[0] < mab_CI[1]
    ), "EXACT and MAB have overlapping confidence intervals. This should not be the case."


def run_stability_baseline_regression(
    seed: int = 0,
    num_trials: int = 10,
    max_depth=5,
    max_leaf_nodes=24,
    num_forests=5,
    num_trees_per_feature=20,
    feature_subsampling="SQRT",
    epsilon=0.00,
    best_k_feature=5,
    budget=FOREST_UNIT_BUDGET_REGRESSION,
) -> None:
    mab_sim_array = []
    exact_sim_array = []
    data, labels = sklearn.datasets.make_regression(10000, n_features=100, n_informative=10)
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
            budget_per_forest=budget,
            solver=EXACT,
            is_classification=False,
            feature_subsampling=feature_subsampling,
            max_leaf_nodes=max_leaf_nodes,
            epsilon=epsilon,
        )
        exact_sim_array.append(exact.run_baseline(best_k_feature))

        mab = PermutationImportance(
            seed=mab_seed,
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

    print("exact and mab avg: ", (e_avg, m_avg))
    print("exact  CI: ", exact_CI)
    print("mab CIs: ", mab_CI)

    assert (
        exact_CI[1] < mab_CI[0]
    ), "EXACT and MAB have overlapping confidence intervals. This should not be the case."


if __name__ == "__main__":
    test_stability_with_budget_digit(0)
    run_stability_baseline_digits(0)

    test_stability_with_budget_regression(0)
    run_stability_baseline_regression(0)


