import numpy as np

from data_structures.forest_classifier import ForestClassifier
from utils.constants import SQRT, RANDOM, GINI, BEST, EXACT, DEFAULT_ERF_K


class ExtremelyRandomForestClassifier(ForestClassifier):
    """
    A RandomForestClassifier, which is a ForestClassifier with the following settings:

    bootstrap: bool = True,
    feature_subsampling: str = SQRT,
    tree_global_feature_subsampling: bool = False,
    bin_type: str = RANDOM,
    num_bins: int = None,
    erf_k: str = DEFAULT_ERF_K (default value, not fixed)
    solver: str = EXACT (default value, not fixed)
    """

    def __init__(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0,
        max_leaf_nodes: int = None,
        budget: int = None,
        criterion: str = GINI,
        splitter: str = BEST,
        solver: str = EXACT,
        erf_k: int = DEFAULT_ERF_K,
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            n_estimators=n_estimators,
            max_depth=max_depth,
            # https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees suggests bootstrapping
            bootstrap=True,  # Fixed.
            feature_subsampling=SQRT,  # Fixed
            tree_global_feature_subsampling=False,  # Fixed.
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            bin_type=RANDOM,  # Fixed
            num_bins=None,  # Fixed
            budget=budget,
            criterion=criterion,
            splitter=splitter,
            solver=solver,
            erf_k=erf_k,  # Fixed
            random_state=random_state,
            verbose=verbose,
        )
