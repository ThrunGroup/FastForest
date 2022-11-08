import cProfile
import pstats

import time
from typing import Any
import pprint
import os

from experiments.datasets import data_loader

from experiments.exp_utils import *
from experiments.exp_constants import (
    MAX_DEPTH,
    RUNTIME_ALPHA_N,
    RUNTIME_ALPHA_F,
    RUNTIME_NUM_SEEDS,
    RUNTIME_MAX_DEPTH
)

from utils.constants import CLASSIFICATION_MODELS, REGRESSION_MODELS
from utils.constants import FLIGHT, AIR, APS, BLOG, SKLEARN_REGRESSION, MNIST_STR, HOUSING, COVTYPE, KDD, GPU
from utils.constants import (
    GINI,
    BEST,
    EXACT,
    MAB,
    MSE,
    DEFAULT_NUM_BINS,
    DEFAULT_MIN_IMPURITY_DECREASE,
)

from data_structures.wrappers.histogram_random_forest_classifier import (
    HistogramRandomForestClassifier as HRFC,
)

from experiments.runtime_exps.compare_runtimes import time_measured_fit

def main():
    dataset = MNIST_STR
    max_depth = 1

    train_images, train_labels, test_images, test_labels = data_loader.fetch_data(dataset)

    our_model = HRFC(
        data=train_data,
        labels=train_targets,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=default_min_samples_split,
        min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
        max_leaf_nodes=max_leaf_nodes,
        budget=None,
        criterion=GINI,
        splitter=BEST,
        solver=MAB,
        random_state=seed,
        verbose=verbose,
    )
    their_model = HRFC(
        data=train_data,
        labels=train_targets,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=default_min_samples_split,
        min_impurity_decrease=DEFAULT_MIN_IMPURITY_DECREASE,
        max_leaf_nodes=max_leaf_nodes,
        budget=None,
        criterion=GINI,
        splitter=BEST,
        solver=RANDOM_SOLVER,
        random_state=seed,
        verbose=verbose,
    )

if __name__ == "__main__":
    main()