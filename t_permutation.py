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


def test_stability_with_budget() -> None:
    np.random.seed(0)
    digits = sklearn.datasets.load_digits()
    data, labels = digits.data, digits.target
    print(data.shape)

    stability_metric = SPEARMAN
    PI_exact = PermutationImportance(
        data=data,
        labels=labels,
        max_depth=5,
        num_forests=5,
        num_trees_per_forest=50,
        budget_per_forest=FOREST_UNIT_BUDGET,
        solver=EXACT,
        stability_metric=stability_metric,
    )
    res_exact = PI_exact.get_importance_array()
    stability_exact = PI_exact.get_stability(res_exact)
    print("stability for exact", stability_exact)

    PI_mab = PermutationImportance(
        data=data,
        labels=labels,
        max_depth=5,
        num_forests=5,
        num_trees_per_forest=50,
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


if __name__ == "__main__":
    #test_contrived_dataset()
    test_stability_with_budget()
    

