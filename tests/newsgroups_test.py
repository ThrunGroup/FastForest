import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from data_structures.tree_classifier import TreeClassifier
from utils.constants import EXACT, BEST, MAB, ENTROPY
import utils.utils


def preprocess_news(verbose: bool = False):
    """
    Returns training and test dataest of sklearn newsgroups dataset. Reduce the size of dataset using PCA and only
    considering 2 target values.
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
    trans = vectorizer.fit(ng_train.data)
    train_vectors = vectorizer.transform(ng_train.data)
    test_vectors = vectorizer.transform(ng_test.data)
    if verbose:
        print(train_vectors.shape)
        print("Number of datapoints: ", len(ng_train.data))
        print("Number of features: ", train_vectors.shape[1])
        print(
            "Balance: ", np.sum(ng_train.target) / len(ng_train.target)
        )  # 55-45, roughly balanced

    N_COMPONENTS = 100
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(train_vectors.toarray())
    pca_train_vecs = pca.transform(train_vectors.toarray())
    pca_test_vecs = pca.transform(test_vectors.toarray())
    return pca_train_vecs, ng_train.target, pca_test_vecs, ng_test.target


def test_tree_news(
    seed: int = 1,
    verbose: bool = False,
    solver: str = MAB,
    with_replacement: bool = False,
    print_sklearn: bool = False,
):
    X_train, Y_train, X_test, Y_test = preprocess_news()
    if print_sklearn:
        dt = DecisionTreeClassifier(max_depth=5, random_state=0)
        dt.fit(X_train, Y_train)
        print(
            "sklearn Decision Tree Train Accuracy:",
            np.mean(dt.predict(X_train) == Y_train),
        )
        print(
            "sklearn Decision Tree Test Accuracy:",
            np.mean(dt.predict(X_test) == Y_test),
        )

    classes_arr = np.unique(Y_train)
    classes = utils.utils.class_to_idx(classes_arr)
    tc = TreeClassifier(
        forest=None,
        data=X_train,
        labels=Y_train,
        max_depth=4,
        classes=classes,
        verbose=False,
        bin_type="",
        random_state=seed,
        solver=solver,
        with_replacement=with_replacement,
        criterion=ENTROPY,
    )
    tc.fit()
    train_acc = np.mean(tc.predict_batch(X_train)[0] == Y_train)
    test_acc = np.mean(tc.predict_batch(X_test)[0] == Y_test)
    if verbose:
        print("--Experiment FastTree with Newsgroups dataset--")
        print(f"Seed : {seed}")
        print(f"Solver: {solver}")
        print(f"Sample with replacement: {with_replacement}")
        print("Train accuracy:", train_acc)
        print("Test accuracy:", test_acc)
        print("Num queries:", tc.num_queries, "\n")
    return tc.num_queries, train_acc
