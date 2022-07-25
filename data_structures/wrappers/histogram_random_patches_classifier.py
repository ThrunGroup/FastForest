import math
import numpy as np

from data_structures.forest_classifier import ForestClassifier
from utils.constants import LINEAR, DEFAULT_NUM_BINS, GINI, BEST, EXACT, SQRT


class HistogramRandomPatchesClassifier(ForestClassifier):
    """
    A HistogramRandomPatchesClassifier, which is a ForestClassifier with the following settings with subsampled data and
    features.

    bootstrap: bool = False,
    feature_subsampling: str = None,
    tree_global_feature_subsampling: bool = True,
    bin_type: str = LINEAR,
    num_bins: int = DEFAULT_NUM_BINS, (default value, not fixed)
    solver: str = EXACT (default value, not fixed)
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        alpha_N: float = 1.0,
        alpha_F: float = 1.0,
        n_estimators: int = 100,
        max_depth: int = None,
        num_bins: int = DEFAULT_NUM_BINS,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0,
        max_leaf_nodes: int = None,
        budget: int = None,
        criterion: str = GINI,
        splitter: str = BEST,
        solver: str = EXACT,
        random_state: int = 0,
        with_replacement: bool = False,
        verbose: bool = False,
    ) -> None:
        if alpha_N is None or alpha_F is None:
            raise Exception("Need to pass alpha_N and alpha_F to RP objects")
        super().__init__(
            data=data,  # Fixed
            labels=labels,  # Fixed
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=True,  # Fixed
            feature_subsampling=SQRT,  # Fixed
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            bin_type=LINEAR,  # Fixed
            num_bins=num_bins,
            budget=budget,
            criterion=criterion,
            splitter=splitter,
            solver=solver,
            random_state=random_state,
            with_replacement=with_replacement,
            verbose=verbose,
            alpha_N=alpha_N,
            alpha_F=alpha_F,
        )
