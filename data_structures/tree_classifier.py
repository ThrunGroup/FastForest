from data_structures.classifier import Classifier
from data_structures.tree_base import TreeBase
from typing import DefaultDict
from collections import defaultdict

import numpy as np


class TreeClassifier(TreeBase, Classifier):
    """
    TreeClassifier object. Contains a node attribute, the root, as well as fitting parameters that are global
    to the tree (i.e., are used in splitting the nodes)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        max_depth: int,
        classes: dict,
        budget: int = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = -1e-6,
        max_leaf_nodes: int = None,
        discrete_features: DefaultDict = defaultdict(list),
        bin_type: str = "linear",
        erf_k: str = "",
        verbose: bool = True,
    ):
        self.classes = classes  # dict from class name to class index
        self.idx_to_class = {value: key for key, value in classes.items()}
        super().__init__(
            data=data,
            labels=labels,
            max_depth=max_depth,
            classes=classes,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            discrete_features=discrete_features,
            bin_type=bin_type,
            is_classification=True,
            budget=budget,
            verbose=verbose,
            erf_k=erf_k
        )
