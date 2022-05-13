import math
import numpy as np

from data_structures.forest_classifier import ForestClassifier
from utils.constants import IDENTITY, GINI, BEST, EXACT


class GradientBoostedHistogramRandomPatchesClassifier(ForestClassifier):
    """
    A GradientBoostedHistogramRandomPatchesClassifier, which is a ForestClassifier with the following settings with subsampled data and
    features.

    bootstrap: bool = False,
    feature_subsampling: str = None,
    tree_global_feature_subsampling: bool = True,
    bin_type: str = LINEAR,
    num_bins: int = DEFAULT_NUM_BINS, (default value, not fixed)
    solver: str = EXACT (default value, not fixed)
    boosting: bool = True,
    boosting_lr: float = parameter
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        alpha_N: float = 1.0,
        alpha_F: float = 1.0,
        n_estimators: int = 100,
        max_depth: int = None,
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
        boosting_lr: float = None,
    ) -> None:
        N = len(data)
        F = len(data[0])
        data_idcs = np.random.choice(N, math.ceil(alpha_N * N), replace=False)
        feature_idcs = np.random.choice(F, math.ceil(alpha_F * F), replace=False)

        self.data = data[data_idcs, feature_idcs]
        self.labels = labels[data_idcs]
        super().__init__(
            data=self.data,  # Fixed
            labels=self.labels,  # Fixed
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=False,  # Fixed
            feature_subsampling=None,  # Fixed
            tree_global_feature_subsampling=True,  # Fixed
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            bin_type=IDENTITY,  # Fixed
            num_bins=None,  # Fixed
            budget=budget,
            criterion=criterion,
            splitter=splitter,
            solver=solver,
            random_state=random_state,
            with_replacement=with_replacement,
            verbose=verbose,
            boosting=True,  # Fixed
            boosting_lr=boosting_lr,
        )
