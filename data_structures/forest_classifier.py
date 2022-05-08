import numpy as np
from typing import Tuple, DefaultDict

from data_structures.tree_classifier import TreeClassifier
from data_structures.boosted_tree_classifier import BoostedTreeClassifier
from data_structures.classifier import Classifier
from utils.constants import BUFFER, MAB, LINEAR, GINI, SQRT, BEST
from utils.utils import class_to_idx, data_to_discrete


class ForestClassifier(Classifier):
    """
    Class for vanilla random forest model, which averages each tree's predictions
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        n_estimators: int = 100,
        max_depth: int = None,
        bootstrap: bool = True,
        feature_subsampling: str = SQRT,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0,
        max_leaf_nodes: int = None,
        bin_type: str = LINEAR,
        budget: int = None,
        criterion: str = GINI,
        splitter: str = BEST,
        solver: str = MAB,
        verbose: bool = True,
        erf_k: str = SQRT,

        # parameters specific for boosting
        use_boosting: bool = False,
        loss_type: str = "CELoss"
    ) -> None:
        self.data = data
        self.num_features = len(data[0])
        self.labels = labels
        self.trees = []
        self.n_estimators = n_estimators

        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.feature_subsampling = feature_subsampling
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.bin_type = bin_type
        self.erf_k = erf_k

        self.classes: dict = class_to_idx(
            np.unique(labels)
        )  # a dictionary that maps class name to class index
        self.n_classes = len(self.classes)

        self.remaining_budget = budget
        self.num_queries = 0

        self.criterion = criterion
        self.solver = solver
        self.splitter = splitter
        self.verbose = verbose

        # Same parameters as sklearn.ensembleRandomForestClassifier. We won't need all of them.
        # See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0.0
        self.max_features = None
        self.oob_score = False
        self.n_jobs = None
        self.random_state = None
        self.warm_start = False
        self.class_weight = None
        self.ccp_alpha = 0.0
        self.max_samples = None

        # Need this to do remapping when features are shuffled
        self.tree_feature_idcs = {}

        self.discrete_features: DefaultDict = data_to_discrete(
            data, n=10
        )  # TODO: Fix this hard-coding

        self.use_boosting = use_boosting
        self.loss_type = loss_type

    def check_both_or_neither(
        self, data: np.ndarray = None, labels: np.ndarray = None
    ) -> bool:
        if data is None:
            if labels is not None:
                raise Exception("Need to pass both data and labels to .fit()")
        else:
            if labels is None:
                raise Exception("Need to pass both data and labels to .fit()")

        # Either (data and labels) or (not data and not labels)
        return True

    def fit(self, data: np.ndarray = None, labels: np.ndarray = None) -> None:
        """
        Fit the random forest classifier by training trees, where each tree is trained with only a subset of the
        available features

        :return: None
        """

        self.check_both_or_neither(data, labels)
        if data is not None:
            self.data = data
            self.labels = labels

        self.trees = []
        new_labels = [self.labels]
        for i in range(self.n_estimators):
            if self.remaining_budget is not None and self.remaining_budget <= 0:
                break

            if self.feature_subsampling == SQRT:
                feature_idcs = np.random.choice(
                    self.num_features, size=int(np.ceil(np.sqrt(self.num_features)))
                )
            else:
                raise Exception("Bad feature subsampling method")

            if self.verbose:
                print("Fitting tree", i)

            self.tree_feature_idcs[i] = feature_idcs

            if self.bootstrap:
                N = len(self.labels)
                idcs = np.random.choice(N, size=N, replace=True)
                new_data = self.data[idcs, :]
                new_labels[0] = self.labels[idcs]
            else:
                new_data = self.data
                new_labels[0] = self.labels

            if self.use_boosting:
                tree = BoostedTreeClassifier(
                    data=new_data[
                         :, feature_idcs
                         ],  # Randomly choose a subset of the available features
                    labels=new_labels[0],
                    max_depth=self.max_depth,
                    classes=self.classes,
                    budget=self.remaining_budget,
                    verbose=self.verbose,
                    min_samples_split=self.min_samples_split,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_leaf_nodes=self.max_leaf_nodes,
                    discrete_features=self.discrete_features,
                    bin_type=self.bin_type,
                    solver=self.solver,
                    erf_k=self.erf_k,
                    loss_type=self.loss_type
                )
            else:
                tree = TreeClassifier(
                    data=new_data[
                        :, feature_idcs
                    ],  # Randomly choose a subset of the available features
                    labels=new_labels[0],
                    max_depth=self.max_depth,
                    classes=self.classes,
                    budget=self.remaining_budget,
                    verbose=self.verbose,
                    min_samples_split=self.min_samples_split,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_leaf_nodes=self.max_leaf_nodes,
                    discrete_features=self.discrete_features,
                    bin_type=self.bin_type,
                    solver=self.solver,
                    erf_k=self.erf_k,
                )
            tree.fit()
            if self.use_boosting:
                new_labels[0] = tree.update_next_labels()
            self.trees.append(tree)

            # Bookkeeping
            self.num_queries += tree.num_queries
            if self.remaining_budget is not None:
                self.remaining_budget -= tree.num_queries
                assert self.remaining_budget > -BUFFER, "Error: went over budget"

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
