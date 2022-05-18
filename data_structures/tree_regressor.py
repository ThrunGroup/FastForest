import numpy as np
from typing import DefaultDict, Union, Tuple
from collections import defaultdict

from data_structures.regressor import Regressor
from data_structures.tree_base import TreeBase
from utils.constants import MAB, LINEAR, BEST, MSE, DEFAULT_NUM_BINS


class TreeRegressor(TreeBase, Regressor):
    """
    Regression tree object. Contains a node attribute, the root, as well as fitting parameters that are global
    to the tree.
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        max_depth: int = 100,
        feature_subsampling: Union[str, int] = None,
        tree_global_feature_subsampling: bool = False,
        min_samples_split: int = 2,
        min_impurity_decrease: float = -1e-5,
        max_leaf_nodes: int = None,
        discrete_features: DefaultDict = None,
        bin_type: str = LINEAR,
        num_bins: int = DEFAULT_NUM_BINS,
        budget: int = None,
        criterion: str = MSE,
        splitter: str = BEST,
        solver: str = MAB,
        random_state: int = 0,
        with_replacement: bool = False,
        verbose: bool = False,
        make_discrete: bool = False,
        minmax: Tuple[np.ndarray, np.ndarray] = None,
    ):
        super().__init__(
            data=data,
            labels=labels,
            max_depth=max_depth,
            feature_subsampling=feature_subsampling,
            tree_global_feature_subsampling=tree_global_feature_subsampling,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            discrete_features=discrete_features,
            bin_type=bin_type,
            num_bins=num_bins,
            budget=budget,
            is_classification=False,
            criterion=criterion,
            splitter=splitter,
            solver=solver,
            random_state=random_state,
            with_replacement=with_replacement,
            verbose=verbose,
            make_discrete=make_discrete,
            minmax=minmax,
        )
