import sklearn.datasets
import numpy as np
import math
import os
import pandas as pd
from sklearn.datasets import make_regression, make_classification

from permutation import PermutationImportance
from utils.constants import (
    MAB,
    EXACT,
    FOREST_UNIT_BUDGET_DIGIT,
    FOREST_UNIT_BUDGET_DIABETES,
    MAX_SEED,
)
from experiments.heart.fit_heart import append_dict_as_row


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


def test_stability_with_budget(
    seed: int = 1,
    data_size=10000,
    num_features=40,
    num_forests=5,
    num_trees_per_feature=20,
    best_k_features=7,
    max_depth=5,
    max_leaf_nodes=24,
    feature_subsampling="SQRT",
    epsilon=0.00,
    budget=500000,
    importance_score="impurity",
    num_informative=5,
    is_classification=False,
) -> None:
    np.random.seed(seed)
    data_name = "make_regression" if not is_classification else "make_classification"
    if not is_classification:
        data, labels = make_regression(
            n_samples=data_size, n_features=num_features, n_informative=num_informative
        )
    else:
        data, labels = make_classification(
            n_samples=data_size, n_features=num_features, n_informative=num_informative
        )
    PI_exact = PermutationImportance(
        seed=seed,
        data=data,
        labels=labels,
        max_depth=max_depth,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=budget,
        solver=EXACT,
        is_classification=is_classification,
        feature_subsampling=feature_subsampling,
        max_leaf_nodes=max_leaf_nodes,
        epsilon=epsilon,
        importance_score=importance_score,
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
        is_classification=is_classification,
        feature_subsampling=feature_subsampling,
        max_leaf_nodes=max_leaf_nodes,
        epsilon=epsilon,
        importance_score=importance_score,
    )
    stability_mab = PI_mab.run_baseline(best_k_features)
    print("stability for mab", stability_mab)

    if stability_mab > stability_exact:
        print("MAB IS MORE STABLE!!!!")
    else:
        print("EXACT is more stable :((")

    log_dict = {
        "stability_diff": stability_mab - stability_exact,
        "dataset": data_name,
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
        "num_informative": num_informative,
        "importance_score": importance_score,
    }
    dir_name = "../stability_log"
    log_filename = os.path.join(dir_name, "stability_log.csv")
    if not os.path.exists(log_filename):
        os.makedirs(dir_name, exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())
    assert stability_mab > stability_exact, "stability of exact is greater than or equal to stability mab"


def run_stability_make_regressions(
    seed: int = 0,
    data_size=10000,
    num_features=30,
    num_informative=6,
    num_trials: int = 30,
    num_forests: int = 10,
    max_depth: int = 6,
    max_leaf_nodes=40,
    num_trees_per_feature: int = 10,
    feature_subsampling: str = "SQRT",
    best_k_feature: int = 6,
    epsilon=0.0,
    budget=350000,
    importance_score="impurity",
    is_classification=True,
) -> None:
    data_name = "make_regression" if not is_classification else "make_classification"
    exact_stab_array = []
    mab_stab_array = []
    if not is_classification:
        data, labels = make_regression(
            n_samples=data_size, n_features=num_features, n_informative=num_informative
        )
    else:
        data, labels = make_classification(
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
            is_classification=is_classification,
            importance_score=importance_score,
        )
        exact_stab_array.append(exact.run_baseline(best_k_feature))
        print(exact_stab_array[-1])

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
            is_classification=is_classification,
            importance_score=importance_score,
        )
        mab_stab_array.append(mab.run_baseline(best_k_feature))
        print(mab_stab_array[-1])

    # compute confidence intervals
    conf_multiplier = 1.96
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
        "dataset": data_name,
        "budget": budget,
        "num_forests": num_forests,
        "num_trees": num_trees_per_feature,
        "best_k": best_k_feature,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "feature_subsampling": feature_subsampling,
        "epsilon": epsilon,
        "seed": seed,
        "conf_multiplier": conf_multiplier,
        "num_trials": num_trials,
        "lb_exact": exact_CI[0],
        "ub_exact": exact_CI[1],
        "lb_mab": mab_CI[0],
        "ub_mab": mab_CI[0],
    }
    dir_name = "../stat_test_stability_log"
    log_filename = os.path.join(dir_name, "statistics_log.csv")
    if not os.path.exists(log_filename):
        os.makedirs(dir_name, exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())
    assert not is_overlap, "Exact and MABs stability overlaps"


if __name__ == "__main__":
    run_stability_make_regressions(213)
