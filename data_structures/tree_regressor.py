from data_structures.regressor import Regressor
from data_structures.tree_base import TreeBase
from typing import DefaultDict
from collections import defaultdict

import numpy as np


class TreeRegressor(TreeBase, Regressor):
    """
    Regression tree object. Contains a node attribute, the root, as well as fitting parameters that are global
    to the tree.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        max_depth: int,
        budget: int = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = -1e-5,
        max_leaf_nodes: int = None,
        discrete_features: DefaultDict = defaultdict(list),
        bin_type: str = "linear",
        verbose: bool = True,
    ):
        super().__init__(
            data=data,
            labels=labels,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            discrete_features=discrete_features,
            bin_type=bin_type,
            is_classification=False,
            budget=budget,
            verbose=verbose,
        )
