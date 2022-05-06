import numpy as np
import random
from typing import Tuple, DefaultDict

from data_structures.tree_classifier import TreeClassifier
from data_structures.classifier import Classifier
from utils.utils import class_to_idx, data_to_discrete


class Forest(Classifier):
    """
    Class for vanilla random forest model, which averages each tree's predictions
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = None,
        bootstrap: bool = True,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0,
        max_leaf_nodes: int = 0,
        bin_type="linear",
    ) -> None:
        self.data = data
        self.num_features = len(data[0])
        self.labels = labels
        self.trees = []
        self.n_estimators = n_estimators
        self.feature_subsampling = "SQRT"
        self.classes: dict = class_to_idx(
            np.unique(labels)
        )  # a dictionary that maps class name to class index
        self.n_classes = len(self.classes)

        # Same parameters as sklearn.ensembleRandomForestClassifier. We won't need all of them.
        # See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        self.criterion = "gini"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0.0
        self.max_features = "auto"
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = True
        self.oob_score = False
        self.n_jobs = None
        self.random_state = None
        self.verbose = 0
        self.warm_start = False
        self.class_weight = None
        self.ccp_alpha = 0.0
        self.max_samples = None
        self.num_queries = 0
        self.bootstrap = bootstrap
        self.bin_type = bin_type

        # Need this to do remapping when features are shuffled
        self.tree_feature_idcs = {}

        self.discrete_features: DefaultDict = data_to_discrete(
            data, n=10
        )  # TODO: Fix this hard-coding

    def fit(self, verbose=True) -> None:
        """
        Fit the random forest classifier by training trees, where each tree is trained with only a subset of the
        available features

        :return: None
        """
        self.trees = []

        for i in range(self.n_estimators):
            if self.feature_subsampling == "SQRT":
                feature_idcs = np.random.choice(
                    self.num_features, size=int(np.ceil(np.sqrt(self.num_features)))
                )
            else:
                raise Exception("Bad feature subsampling method")

            if verbose:
                print("Fitting tree", i)

            self.tree_feature_idcs[i] = feature_idcs

            if self.bootstrap:
                N = len(self.labels)
                idcs = np.random.choice(N, size=N, replace=True)
                new_data = self.data[idcs, :]
                new_labels = self.labels[idcs]
            else:
                new_data = self.data
                new_labels = self.labels
            tree = TreeClassifier(
                data=new_data[
                    :, feature_idcs
                ],  # Randomly choose a subset of the available features
                labels=new_labels,
                max_depth=self.max_depth,
                classes=self.classes,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_leaf_nodes=self.max_leaf_nodes,
                discrete_features=self.discrete_features,
                bin_type=self.bin_type,
            )
            tree.fit()
            self.num_queries += tree.num_queries
            self.trees.append(tree)

    def predict(self, datapoint: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Generate a prediction for the given datapoint by averaging over all trees' predictions
        :param datapoint: datapoint to predict
        :return: a tuple containing class label, probability of class label
        """
        T = len(self.trees)
        agg_preds = np.empty((T, self.n_classes))

        for tree_idx, tree in enumerate(self.trees):
            agg_preds[tree_idx] = tree.predict(
                datapoint[self.tree_feature_idcs[tree_idx]]
            )[1]

        avg_preds = agg_preds.mean(axis=0)
        label_pred = list(self.classes.keys())[avg_preds.argmax()]
        return label_pred, avg_preds
