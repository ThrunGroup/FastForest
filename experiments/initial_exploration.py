import numpy as np
import time

from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier_sklearn
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sklearn

from data_structures.tree_classifier import TreeClassifier as TreeClassifier_ours
from utils.constants import EXACT, MAB
from experiments.exp_utils import load_pca_ng, make_huge


def sklearn_ng_perf(pca_train_vecs, train_labels, pca_test_vecs, test_labels):
    dt = DecisionTreeClassifier_sklearn(random_state=0, max_depth=5)
    dt.fit(pca_train_vecs, train_labels)
    print(
        "sklearn Decision Train Accuracy:",
        np.mean(dt.predict(pca_train_vecs) == train_labels),
    )
    print(
        "sklearn Decision Tree Accuracy:",
        np.mean(dt.predict(pca_test_vecs) == test_labels),
    )

    rf = RandomForestClassifier_sklearn(random_state=0, max_depth=5)
    rf.fit(pca_train_vecs, train_labels)
    print(
        "sklearn Random Forest Train Accuracy:",
        np.mean(rf.predict(pca_train_vecs) == train_labels),
    )
    print(
        "sklearn Random Forest Test Accuracy:",
        np.mean(rf.predict(pca_test_vecs) == test_labels),
    )
    print("-" * 30)


def ours_ng_perf(
    pca_train_vecs,
    train_labels,
    pca_test_vecs,
    test_labels,
    classes,
    max_depth=5,
    solver=EXACT,
    verbose=False,
):
    tc = TreeClassifier_ours(
        data=pca_train_vecs,
        labels=train_labels,
        max_depth=max_depth,
        classes=classes,
        solver=solver,
        verbose=verbose,
        random_state=0,
    )
    start = time.time()
    tc.fit()
    end = time.time()
    print(
        "Our Train accuracy:",
        np.mean(tc.predict_batch(pca_train_vecs)[0] == train_labels),
    )
    print(
        "Our Test accuracy:", np.mean(tc.predict_batch(pca_test_vecs)[0] == test_labels)
    )
    print("Num queries:", tc.num_queries)
    print(solver + " Runtime:", end - start)
    print("-" * 30)


def main():
    pca_train_vecs, train_labels, pca_test_vecs, test_labels, classes = load_pca_ng()

    # Print sklearn performance for both Tree and Forest. Forest perf should be better than Tree
    sklearn_ng_perf(pca_train_vecs, train_labels, pca_test_vecs, test_labels)

    # Print our performance for just Trees. Should be similar to sklearn tree, and MAB should (counterintuitively)
    # take longer than EXACT
    ours_ng_perf(
        pca_train_vecs,
        train_labels,
        pca_test_vecs,
        test_labels,
        classes,
        max_depth=5,
        solver=EXACT,
        verbose=True,
    )

    ours_ng_perf(
        pca_train_vecs,
        train_labels,
        pca_test_vecs,
        test_labels,
        classes,
        max_depth=5,
        solver=MAB,
        verbose=True,
    )

    pca_train_vecs_huge, pca_train_labels_huge = make_huge(pca_train_vecs, train_labels)

    # Print our performance for just Trees. Should be similar to sklearn tree, and MAB should now be faster than EXACT
    ours_ng_perf(
        pca_train_vecs_huge,
        pca_train_labels_huge,
        pca_test_vecs,
        test_labels,
        classes,
        max_depth=2,
        solver=EXACT,
        verbose=True,
    )

    ours_ng_perf(
        pca_train_vecs_huge,
        pca_train_labels_huge,
        pca_test_vecs,
        test_labels,
        classes,
        max_depth=2,
        solver=MAB,
        verbose=True,
    )


if __name__ == "__main__":
    main()
