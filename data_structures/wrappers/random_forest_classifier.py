import numpy as np

from data_structures.forest_classifier import ForestClassifier
from utils.constants import SQRT, IDENTITY, GINI, BEST, EXACT


class RandomForestClassifier(ForestClassifier):
    """
    A RandomForestClassifier, which is a ForestClassifier with the following settings:

    bootstrap: bool = True,
    feature_subsampling: str = SQRT,
    bin_type: str = IDENTITY,
    num_bins: int = None,
    solver: str = EXACT (default value, not fixed, but cannot use MAB because there's no binning)
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
        random_state: int = 0,
        with_replacement: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=True,  # Fixed
            feature_subsampling=SQRT,  # Fixed
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
        )
