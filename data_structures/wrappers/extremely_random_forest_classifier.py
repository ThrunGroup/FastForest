import numpy as np

from data_structures.forest_classifier import ForestClassifier
from utils.constants import RANDOM, GINI, BEST, EXACT, DEFAULT_NUM_BINS, SQRT


class ExtremelyRandomForestClassifier(ForestClassifier):
    """
    A ExtremelyRandomForestClassifier, which is a ForestClassifier with the following settings:

    bootstrap: bool = False,
    feature_subsampling: str = SQRT,
    tree_global_feature_subsampling: bool = False,
    bin_type: str = RANDOM,
    num_bins: int = None,
    solver: str = EXACT (default value, not fixed)
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
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
        super().__init__(
            data=data,
            labels=labels,
            n_estimators=n_estimators,
            max_depth=max_depth,
            # https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees doesn't suggest
            # bootstraping
            bootstrap=False,  # Fixed
            # For ExtraTreesClassifier, feature_subsapling default is SQRT.
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            # For ExtraTreesRegressor, feature_subsampling is 'all'
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor
            feature_subsampling=SQRT,  # Fixed
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            bin_type=RANDOM,  # Fixed
            num_bins=num_bins,  # Fixed
            budget=budget,
            criterion=criterion,
            splitter=splitter,
            solver=solver,
            random_state=random_state,
            with_replacement=with_replacement,
            verbose=verbose,
        )
