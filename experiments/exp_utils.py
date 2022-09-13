import copy
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.datasets import *

import utils.utils


def load_pca_ng(n_components: int = 100):
    """
    Loads the 20 newgroups datasets, choosing just 2 subcategories, and PCA transforms their TF-IDF vectors

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
        "Balance: ", np.sum(ng_train.target) / len(ng_train.target), "\n"
    )  # 55-45, roughly balanced

    pca = PCA(n_components=n_components)
    pca.fit(train_vectors.toarray())

    classes_arr = np.unique(ng_train.target)
    classes = utils.utils.class_to_idx(classes_arr)

    pca_train_vecs = pca.transform(train_vectors.toarray())
    pca_test_vecs = pca.transform(test_vectors.toarray())

    return pca_train_vecs, ng_train.target, pca_test_vecs, ng_test.target, classes


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


def load_housing():
    seed = 0
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    data, targets = fetch_california_housing(return_X_y=True)
    random_idcs = rng.choice(len(data), size=len(data), replace=False)
    data = data[random_idcs]
    targets = targets[random_idcs]

    TRAIN_TEST_SPLIT = 16000
    train_data = data[:TRAIN_TEST_SPLIT]
    train_targets = targets[:TRAIN_TEST_SPLIT]

    test_data = data[TRAIN_TEST_SPLIT:]
    test_targets = targets[TRAIN_TEST_SPLIT:]
    return train_data, train_targets, test_data, test_targets
