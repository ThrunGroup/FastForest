import numpy as np

from data_structures.forest_base import ForestBase
from data_structures.regressor import Regressor
from utils.constants import MAB, LINEAR, GINI, SQRT, BEST, DEFAULT_NUM_BINS


class ForestRegressor(ForestBase, Regressor):
    """
    Class for vanilla random forest regression model, which averages each tree's predictions
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        n_estimators: int = 100,
        max_depth: int = None,
        bootstrap: bool = True,
        feature_subsampling: str = None,
        tree_global_feature_subsampling: bool = False,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0,
        max_leaf_nodes: int = None,
        bin_type: str = LINEAR,
        num_bins: int = DEFAULT_NUM_BINS,
        budget: int = None,
        criterion: str = GINI,
        splitter: str = BEST,
        solver: str = MAB,
        random_state: int = 0,
        with_replacement: bool = False,
        verbose: bool = False,
        boosting: bool = False,
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            feature_subsampling=feature_subsampling,
            tree_global_feature_subsampling=tree_global_feature_subsampling,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            bin_type=bin_type,
            num_bins=num_bins,
            budget=budget,
            criterion=criterion,
            splitter=splitter,
            solver=solver,
            is_classification=False,
            random_state=random_state,
            with_replacement=with_replacement,
            verbose=verbose,
            boosting=boosting,
        )
