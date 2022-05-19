import sklearn.datasets
import numpy as np
import math

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


def test_stability_with_budget_diabetes(
    seed: int,
    max_depth: int = 3,
    num_forests: int = 5,
    num_trees_per_feature: int = 20,
    best_k_features: int = 5,
) -> None:
    np.random.seed(seed)
    diabetes = sklearn.datasets.load_diabetes()
    data, labels = diabetes.data, diabetes.target

    exact = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=FOREST_UNIT_BUDGET_DIABETES,
        solver=EXACT,
        is_classification=False,
    )
    stability_exact = exact.run_baseline(best_k_features)

    mab = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=FOREST_UNIT_BUDGET_DIABETES,
        solver=MAB,
        is_classification=False,
    )
    stability_mab = mab.run_baseline(best_k_features)
    print(stability_exact, stability_mab)
    assert stability_mab > stability_exact, "MAB is NOT more stable for regression"


def run_stability_baseline_digits(
    seed: int,
    num_trials: int = 10,
    max_depth: int = 3,
    num_forests: int = 5,
    num_trees_per_feature: int = 20,
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
    exact_CI_upper = e_avg + e_std

    mab_sim_array = np.asarray(mab_sim_array)
    m_avg = np.mean(mab_sim_array)
    m_std = np.std(mab_sim_array) / math.sqrt(num_trials)
    mab_CI_lower = m_avg - m_std

    assert (
        exact_CI_upper < mab_CI_lower
    ), "EXACT and MAB have overlapping confidence intervals. This should not be the case."


def run_stability_baseline_diabetes(
    seed: int,
    num_trials: int = 10,
    max_depth: int = 3,
    num_forests: int = 5,
    num_trees_per_feature: int = 20,
    best_k_feature: int = 5,
) -> None:
    mab_sim_array = []
    exact_sim_array = []
    diabetes = sklearn.datasets.load_diabetes()
    data, labels = diabetes.data, diabetes.target
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
            budget_per_forest=FOREST_UNIT_BUDGET_DIABETES,
            solver=EXACT,
            is_classification=False,
        )
        exact_sim_array.append(exact.run_baseline(best_k_feature))

        mab = PermutationImportance(
            seed=mab_seed,
            data=data,
            labels=labels,
            max_depth=max_depth,
            num_forests=num_forests,
            num_trees_per_forest=num_trees_per_feature,
            budget_per_forest=FOREST_UNIT_BUDGET_DIABETES,
            solver=EXACT,
            is_classification=False,
        )
        mab_sim_array.append(mab.run_baseline(best_k_feature))

    # compute confidence intervals
    exact_sim_array = np.asarray(exact_sim_array)
    e_avg = np.mean(exact_sim_array)
    e_std = np.std(exact_sim_array) / math.sqrt(num_trials)
    exact_CI_upper = e_avg + e_std

    mab_sim_array = np.asarray(mab_sim_array)
    m_avg = np.mean(mab_sim_array)
    m_std = np.std(mab_sim_array) / math.sqrt(num_trials)
    mab_CI_lower = m_avg - m_std

    assert (
        exact_CI_upper < mab_CI_lower
    ), "EXACT and MAB have overlapping confidence intervals. This should not be the case."


if __name__ == "__main__":
    # test_stability_with_budget_digit(0)
    test_stability_with_budget_diabetes(0)
    # run_stability_baseline_digits(0)
    # run_stability_baseline_diabetes(0)
