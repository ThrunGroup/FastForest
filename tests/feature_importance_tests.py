import sklearn.datasets
import numpy as np
import math
import os
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from typing import Tuple

from data_structures.permutation import PermutationImportance
from utils.constants import (
    MAB,
    EXACT,
    MAX_SEED,
    t5_l1_args,
    t5_l2_args,
    t5_l3_args,
    t5_l4_args,
)
from experiments.heart.fit_heart import append_dict_as_row
from experiments.exp_constants import FI_EPSILON


def test_stability_with_budget(
    seed: int = 0,
    data_size=10000,
    num_features=30,
    num_informative=6,
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
    data_name: str = None,
    file_name: str = None,
    verbose: bool = False,
    verbose_test: bool = True,
    is_log: bool = False,
) -> None:
    np.random.seed(seed)
    if data_name is None:
        data_name = (
            "make_regression" if not is_classification else "make_classification"
        )
    if data_name == "make_regression":
        data, labels = make_regression(
            n_samples=data_size,
            n_features=num_features,
            n_informative=num_informative,
            random_state=seed,
        )
    elif data_name == "make_classification":
        data, labels = make_classification(
            n_samples=data_size,
            n_features=num_features,
            n_informative=num_informative,
            random_state=seed,
        )
    elif data_name == "digits":
        digits = sklearn.load_digits()
        data, labels = digits.data, digits.target
    elif data_name == "diabetes":
        diabetes = sklearn.load_diabetes()
        data, labels = diabetes.data, diabetes.target
    else:
        raise NotImplementedError(f"{data_name} is not implemented")
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
        verbose=verbose,
    )
    stability_exact = PI_exact.run_baseline(best_k_feature)
    if verbose_test:
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
        verbose=verbose,
    )
    stability_mab = PI_mab.run_baseline(best_k_feature)
    if verbose_test:
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
        "best_k": best_k_feature,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "feature_subsampling": feature_subsampling,
        "epsilon": epsilon,
        "seed": seed,
        "num_features": num_features,
        "num_informative": num_informative,
        "importance_score": importance_score,
    }
    if is_log:
        dir_name = "stability_log"
        log_filename = "stability_log.csv" if file_name is None else file_name
        if not os.path.exists(log_filename):
            os.makedirs(dir_name, exist_ok=True)
            df = pd.DataFrame(columns=log_dict.keys())
            df.to_csv(log_filename, index=False)
        append_dict_as_row(log_filename, log_dict, log_dict.keys())
    assert (
        stability_mab > stability_exact
    ), "stability of exact is greater than or equal to stability mab"


def run_stability_stats_test(
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
    conf_multiplier: float = 1.96,
    data_name: str = None,
    file_name: str = None,
    verbose: bool = False,
    verbose_test: bool = True,
    is_log: bool = False,
    csv_log: bool = True,
) -> Tuple[float, float, float, float]:
    if data_name is None:
        data_name = (
            "make_regression" if not is_classification else "make_classification"
        )
    exact_stab_array = []
    mab_stab_array = []
    if data_name == "make_regression":
        data, labels = make_regression(
            n_samples=data_size,
            n_features=num_features,
            n_informative=num_informative,
            random_state=seed,
        )
    elif data_name == "make_classification":
        data, labels = make_classification(
            n_samples=data_size,
            n_features=num_features,
            n_informative=num_informative,
            random_state=seed,
        )
    elif data_name == "digits":
        digits = sklearn.datasets.load_digits()
        data, labels = digits.data, digits.target
    elif data_name == "diabetes":
        diabetes = sklearn.datasets.load_diabetes()
        data, labels = diabetes.data, diabetes.target
    else:
        raise NotImplementedError(f"{data_name} is not implemented")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    for trial in range(num_trials):
        if verbose_test:
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
            verbose=verbose,
        )
        exact_stab_array.append(exact.run_baseline(best_k_feature))
        if verbose_test:
            print(f"Exact stability is {exact_stab_array[-1]}")

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
            verbose=verbose,
        )
        mab_stab_array.append(mab.run_baseline(best_k_feature))
        if verbose_test:
            print(f"Mab Stability is {mab_stab_array[-1]}")

    # compute confidence intervals
    conf_multiplier = conf_multiplier
    exact_stab_array = np.asarray(exact_stab_array)
    e_avg = np.mean(exact_stab_array)
    e_std = np.std(exact_stab_array) / math.sqrt(num_trials)
    exact_CI = [e_avg - e_std * conf_multiplier, e_avg + e_std * conf_multiplier]

    mab_sim_array = np.asarray(mab_stab_array)
    m_avg = np.mean(mab_sim_array)
    m_std = np.std(mab_sim_array) / math.sqrt(num_trials)
    mab_CI = [m_avg - m_std * conf_multiplier, m_avg + m_std * conf_multiplier]
    is_overlap = exact_CI[1] >= mab_CI[0]
    if verbose_test:
        print("confidence interval for exact: ", exact_CI)
        print("confidence interval for mab: ", mab_CI)
    if is_log:
        log_dict = {
            "stability_diff": m_avg - e_avg,
            "mab stability": m_avg,
            "is_overlap": is_overlap,
            "dataset": data_name,
            "data_size": data_size,
            "n_features": num_features,
            "n_informative": num_informative,
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
            "ub_mab": mab_CI[1],
            "f_importance": importance_score,
        }
        dir_name = "stat_test_stability_log"
        if csv_log:
            file_name = "statistics_log_tables.csv" if file_name is None else file_name
            log_filename = os.path.join(dir_name, file_name)
            if not os.path.exists(log_filename):
                os.makedirs(dir_name, exist_ok=True)
                df = pd.DataFrame(columns=log_dict.keys())
                df.to_csv(log_filename, index=False)
            append_dict_as_row(log_filename, log_dict, log_dict.keys())
        else:
            with open(file_name, "w+") as fout:
                fout.write(str(log_dict))

    assert not is_overlap, "Exact and MABs stability overlaps"
    return exact_CI[0], exact_CI[1], mab_CI[0], mab_CI[1]


def reproduce_stability():
    file_path = os.path.join("stat_test_stability_log", "reproduce_stability.csv")
    stability_data = df = pd.read_csv(file_path)
    lb_exact_list, ub_exact_list, lb_mab_list, ub_mab_list = (
        stability_data["lb_exact"],
        stability_data["ub_exact"],
        stability_data["lb_mab"],
        stability_data["ub_mab"],
    )
    epsilon = FI_EPSILON
    print("=" * 30)
    print("Reproduce new Table 5\n")
    lb_exact, ub_exact, lb_mab, ub_mab = run_stability_stats_test(**t5_l1_args)
    assert (
        abs(lb_exact_list[0] - lb_exact) < epsilon
        and abs(ub_exact_list[0] - ub_exact) < epsilon
        and abs(lb_mab_list[0] - lb_mab) < epsilon
        and abs(ub_mab_list[0] - ub_mab) < epsilon
    )
    print("Table 5 line 1 is successfully reproduced!")
    print("-" * 30)
    lb_exact, ub_exact, lb_mab, ub_mab = run_stability_stats_test(**t5_l2_args)
    assert (
        abs(lb_exact_list[1] - lb_exact) < epsilon
        and abs(ub_exact_list[1] - ub_exact) < epsilon
        and abs(lb_mab_list[1] - lb_mab) < epsilon
        and abs(ub_mab_list[1] - ub_mab) < epsilon
    )
    print("Table 5 line 2 is successfully reproduced!")
    print("-" * 30)
    lb_exact, ub_exact, lb_mab, ub_mab = run_stability_stats_test(**t5_l3_args)
    assert (
        abs(lb_exact_list[2] - lb_exact) < epsilon
        and abs(ub_exact_list[2] - ub_exact) < epsilon
        and abs(lb_mab_list[2] - lb_mab) < epsilon
        and abs(ub_mab_list[2] - ub_mab) < epsilon
    )
    print("Table 5 line 3 is successfully reproduced!")
    print("-" * 30)
    lb_exact, ub_exact, lb_mab, ub_mab = run_stability_stats_test(**t5_l4_args)
    assert (
        abs(lb_exact_list[3] - lb_exact) < epsilon
        and abs(ub_exact_list[3] - ub_exact) < epsilon
        and abs(lb_mab_list[3] - lb_mab) < epsilon
        and abs(ub_mab_list[3] - ub_mab) < epsilon
    )
    print("Table 5 line 4 is successfully reproduced!")
    print("-" * 30)


def produce_stability():
    # RF + MID
    print("=" * 30)
    print("Reproduce new Table 5\n")

    def produce_test(line_idx, args, file_name):
        run_stability_stats_test(
            **args, file_name=file_name, is_log=True, csv_log=False
        )
        print(f"Table 5 line {line_idx} is successfully produced")
        print("-" * 30)

    args = [t5_l1_args, t5_l2_args, t5_l3_args, t5_l4_args]
    file_names = ["HRFC+MID_dict", "HRFR+MID_dict", "HRFC+Perm_dict", "HRFR+Perm_dict"]
    for idx in range(4):
        produce_test(idx + 1, args[idx], file_names[idx])


if __name__ == "__main__":
    produce_stability()
