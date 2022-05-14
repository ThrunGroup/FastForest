import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.datasets import *

from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier_sklearn
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sklearn
import copy

from data_structures.tree_classifier import TreeClassifier as TreeClassifier_ours
import utils.utils
import time

from utils.constants import EXACT, MAB


def load_pca_ng(n_components: int = 100):
    """
    Loads the 20 newgroups dataset, choosing just 2 subcategories, and PCA transforms their TF-IDF vectors

    :param n_components: Number of components to use in PCA
    :return: tuple of PCA train vectors, train labels, PCA test vectors, test labels, and classes dict
    """
    # Download the data from two categories
    cats = ["alt.atheism", "sci.space"]
    ng_train = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes"), categories=cats
    )
    ng_test = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes"), categories=cats
    )

    vectorizer = TfidfVectorizer()
    _trans = vectorizer.fit(ng_train.data)
    train_vectors = vectorizer.transform(ng_train.data)
    test_vectors = vectorizer.transform(ng_test.data)
    print("Number of datapoints: ", len(ng_train.data))
    print("Number of features: ", train_vectors.shape[1])
    print(
        "Balance: ", np.sum(ng_train.target) / len(ng_train.target)
    )  # 55-45, roughly balanced

    pca = PCA(n_components=n_components)
    pca.fit(train_vectors.toarray())

    classes_arr = np.unique(ng_train.target)
    classes = utils.utils.class_to_idx(classes_arr)

    pca_train_vecs = pca.transform(train_vectors.toarray())
    pca_test_vecs = pca.transform(test_vectors.toarray())

    return pca_train_vecs, ng_train.target, pca_test_vecs, ng_test.target, classes


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


def make_huge(pca_train_vecs, train_labels, doublings: int = 4):
    pca_train_vecs_huge = copy.deepcopy(pca_train_vecs)
    pca_train_labels_huge = copy.deepcopy(train_labels)
    print(pca_train_vecs_huge.shape)
    for i in range(doublings):
        pca_train_vecs_huge = np.concatenate((pca_train_vecs_huge, pca_train_vecs_huge))
        pca_train_labels_huge = np.concatenate(
            (pca_train_labels_huge, pca_train_labels_huge)
        )
    return pca_train_vecs_huge, pca_train_labels_huge


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
