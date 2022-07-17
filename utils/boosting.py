import numpy as np

from utils.constants import (
    DEFAULT_GRAD_SMOOTHING_VAL,
    DEFAULT_CLASSIFIER_LOSS,
)


def find_gradient(
    loss_type: str,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Computes the gradient for the given loss function w.r.t the prediction target
    ex) gradient for cross-entropy loss:
        d_loss_d_pred = -target/pred

    :return: the gradient matrix of size len(targets)
    """
    if loss_type == DEFAULT_CLASSIFIER_LOSS:
        return -(targets + DEFAULT_GRAD_SMOOTHING_VAL) / (
            predictions + DEFAULT_GRAD_SMOOTHING_VAL
        )
    else:
        NotImplementedError("Invalid choice of loss function")


def find_hessian(
    loss_type: str, predictions: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """
    Computes the hessian for the given loss function w.r.t the prediction target
    ex) hessian for cross-entropy loss:
        d_loss_d_pred = target/pred^2

    :return: the gradient matrix of size len(targets)
    """
    if loss_type == DEFAULT_CLASSIFIER_LOSS:
        return (targets + DEFAULT_GRAD_SMOOTHING_VAL) / (
            np.square(predictions) + DEFAULT_GRAD_SMOOTHING_VAL
        )
    else:
        NotImplementedError("Invalid choice of loss function")


def get_next_targets(
    loss_type: str,
    is_classification,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Updates the targets for the next iteration of boosting. For classification, the resulting new training set will
    look like {X, -grad/hessian} and for regression, it will look like {X, ensemble_residuals}. This function assumes
    the tree is already fitted.

    :return: the new targets
    """
    if is_classification:
        return -find_gradient(loss_type, predictions, targets) / find_hessian(
            loss_type, predictions, targets
        )
    else:
        return targets - predictions
