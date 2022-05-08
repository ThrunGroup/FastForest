import numpy as np
from typing import DefaultDict
from collections import defaultdict

from data_structures.classifier import Classifier
from data_structures.tree_base import TreeBase
from utils.constants import MAB, LINEAR, GINI, BEST


class BoostedTreeClassifier(TreeBase, Classifier):
    """
    BoostedTreeClassifier object. Contains a node attribute, the root, fitting parameters that are global
    to the tree (i.e., are used in splitting the nodes), and the weight calculations to find g_t and h_t
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        max_depth: int,
        classes: dict,
        min_samples_split: int = 2,
        min_impurity_decrease: float = -1e-6,
        max_leaf_nodes: int = None,
        discrete_features: DefaultDict = defaultdict(list),
        bin_type: str = LINEAR,
        budget: int = None,
        criterion: str = GINI,
        splitter: str = BEST,
        solver: str = MAB,
        erf_k: str = "",
        verbose: bool = True,
        loss_type: str = "CELoss"
    ):
        self.classes = classes  # dict from class name to class index
        self.idx_to_class = {value: key for key, value in classes.items()}
        self.loss_type = loss_type
        self.labels = labels
        self.smoothing = 1e-5
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
            budget=budget,
            is_classification=True,
            criterion=criterion,
            splitter=splitter,
            solver=solver,
            verbose=verbose,
            erf_k=erf_k,
        )

    def find_gradient(self, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the gradient for the given loss function using numpy broadcasting
        ex) gradient instance for Cross-Entropy Loss:
            d_loss_d_pred = -label/pred

        :return: the gradient matrix of size len(labels)
        """
        if self.loss_type == "CELoss":
            return -(self.labels + self.smoothing) / (predictions + self.smoothing)
        else:
            NotImplementedError("Invalid choice of loss function")

    def find_hessian(self, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the hessian for the given loss function using numpy broadcasting
        ex) hessian instance for Cross-Entropy Loss:
            d_loss_d_pred = label/pred^2

        :return: the gradient matrix of size len(labels)
        """
        if self.loss_type == "CELoss":
            return (self.labels + self.smoothing) / (np.square(predictions) + self.smoothing)
        else:
            NotImplementedError("Invalid choice of loss function")

    def update_next_labels(self) -> np.ndarray:
        """
        This function updates the labels for the next iteration of boosting.
        The resulting new training set will look like {X, -grad/hessian}.
        It does so by following these steps:
            - get the predictions array by calling predict
            - compute the labels for the next iteration.

        NOTE: this function assumes tree is already fitted
        :return: the new updated labels
        """
        _, prediction_probs = self.predict_batch(self.data)
        mean_probs = np.mean(prediction_probs, axis=1)
        return -self.find_gradient(mean_probs) / self.find_hessian(mean_probs)



