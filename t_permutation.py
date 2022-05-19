import sklearn.datasets
import numpy as np
from permutation import PermutationImportance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.constants import EXACT, MAB, FOREST_UNIT_BUDGET, JACCARD, SPEARMAN, KUNCHEVA


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


def test_stability_with_budget(seed: int) -> None:
    np.random.seed(seed)
    digits = sklearn.datasets.load_digits()
    data, labels = digits.data, digits.target
    print(data.shape)

    stability_metric = JACCARD
    num_forests = 5
    num_trees_per_feature = 30
    PI_exact = PermutationImportance(
        data=data,
        labels=labels,
        max_depth=3,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=FOREST_UNIT_BUDGET,
        solver=EXACT,
        stability_metric=stability_metric,
    )
    res_exact = PI_exact.get_importance_array()
    stability_exact = PI_exact.get_stability(res_exact)
    print("stability for exact", stability_exact)
    print("\n\n")

    PI_mab = PermutationImportance(
        data=data,
        labels=labels,
        max_depth=3,
        num_forests=num_forests,
        num_trees_per_forest=num_trees_per_feature,
        budget_per_forest=FOREST_UNIT_BUDGET,
        solver=MAB,
        stability_metric=stability_metric,
    )
    res_mab = PI_mab.get_importance_array()
    stability_mab = PI_mab.get_stability(res_mab)
    print("stability for mab", stability_mab)

    if stability_mab > stability_exact:
        print("MAB IS MORE STABLE!!!!")
    else:
        print("EXACT is more stable :((")


def run_stability_baseline_digits(
        num_trials: int = 10,
        num_forests: int = 5,
        max_depth: int = 3,
        num_trees_per_feature: int = 30,
        stability_metric : str = JACCARD,
) -> None:
    exact_sim_array = []
    mab_sim_array = []
    digits = sklearn.datasets.load_digits()
    data, labels = digits.data, digits.target

    for trial in range(num_trials):
        print("TRIALS NUM: ", trial)
        exact = PermutationImportance(
            data=data,
            labels=labels,
            max_depth=max_depth,
            num_forests=num_forests,
            num_trees_per_forest=num_trees_per_feature,
            budget_per_forest=FOREST_UNIT_BUDGET,
            solver=EXACT,
            stability_metric=stability_metric,
        )
        exact_sim_array.append(exact.run_baseline())

        mab = PermutationImportance(
            data=data,
            labels=labels,
            max_depth=max_depth,
            num_forests=num_forests,
            num_trees_per_forest=num_trees_per_feature,
            budget_per_forest=FOREST_UNIT_BUDGET,
            solver=MAB,
            stability_metric=stability_metric,
        )
        mab_sim_array.append(mab.run_baseline())

    # compute confidence intervals
    exact_sim_array = np.asarray(exact_sim_array)
    e_avg = np.mean(exact_sim_array)
    e_std = np.std(exact_sim_array) / math.sqrt(num_trials)
    exact_CI = [e_avg-e_std, e_avg+e_std]

    mab_sim_array = np.asarray(mab_sim_array)
    m_avg = np.mean(mab_sim_array)
    m_std = np.std(mab_sim_array) / math.sqrt(num_trials)
    mab_CI = [m_avg - m_std, m_avg + m_std]

    print("confidence interval for exact: ", exact_CI)
    print("\n")
    print("confidence interval for mab: ", mab_CI)


if __name__ == "__main__":
    #test_contrived_dataset()
    #test_stability_with_budget()
    run_stability_baseline_digits()
    

