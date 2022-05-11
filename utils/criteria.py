from typing import Tuple, List, Callable, Union
import scipy
import numpy as np

from data_structures.histogram import Histogram
from utils.constants import GINI, ENTROPY, VARIANCE, MSE


def get_gini(
    counts: np.ndarray, ret_var: bool = False, pop_size: int = None,
) -> Union[Tuple[float, float], float]:
    """
    Compute the Gini impurity for a given node, where the node is represented by the number of counts of each class
    label. The Gini impurity is equal to 1 - sum_{i=1}^k (p_i^2)

    :param counts: 1d array of counts where ith element is the number of counts on the ith class(label).
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :return: the Gini impurity of the node, as well as its estimated variance if ret_var
    """
    n = np.sum(counts)
    if n == 0:
        if ret_var:
            return 0, 0
        return 0
    p = counts / n
    V_p = p * (1 - p) / n
    if pop_size is not None:
        assert pop_size >= n, "Sample size is greater than the population size"
        if pop_size <= 1:
            return 0, 0
        # Use FPC for variance calculation, see
        # https://stats.stackexchange.com/questions/376417/sampling-from-finite-population-with-replacement
        V_p *= (pop_size - n) / (pop_size - 1)

    G = 1 - np.dot(p, p)
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        dG_dp = (
            -2 * p[:-1] + 2 * p[-1]
        )  # Note: len(dG_dp) is len(p) - 1 since p[-1] is dependent variable on p[:-1]
        V_G = np.dot(dG_dp ** 2, V_p[:-1])
        return float(G), float(V_G)
    return float(G)


def get_entropy(counts: np.ndarray, ret_var=False, pop_size: int = None) -> Union[Tuple[float, float], float]:
    """
    Compute the entropy impurity for a given node, where the node is represented by the number of counts of each class
    label. The entropy impurity is equal to - sum{i=1}^k (p_i * log_2 p_i)

    :param counts: 1d array of counts where ith element is the number of counts on the ith class(label)
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :return: the entropy impurity of the node, as well as its estimated variance if ret_var
    """
    n = np.sum(counts)
    if n == 0:
        if ret_var:
            return 0, 0
        return 0

    p = counts / n
    V_p = p * (1 - p) / n
    if pop_size is not None:
        assert pop_size >= n, "Sample size is greater than the population size"
        if pop_size <= 1:
            return 0, 0
        # Use FPC for variance calculation, see
        # https://stats.stackexchange.com/questions/376417/sampling-from-finite-population-with-replacement
        V_p *= (pop_size - n) / (pop_size - 1)
    log_p = np.zeros(len(p))
    for i in range(len(p)):
        if p[i] != 0:
            log_p[i] = np.log(p[i])
    E = np.dot(-log_p, p)  # Note: when p -> 0, (-log(p) * p ) -> 0
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        dE_dp = (
            -log_p[:-1] + log_p[-1]
        )  # Note: len(dE_dp) is len(p) - 1 since p[-1] is dependent variable on p[:-1]
        V_E = np.dot(dE_dp ** 2, V_p[:-1])
        return float(E), float(V_E)
    return float(E)


def get_variance(
    counts: np.ndarray, ret_var=False
) -> Union[Tuple[float, float], float]:
    """
    Compute the variance for a given node, where the node is represented by the number of counts of each class
    label.

    :param counts: 1d array of counts where ith element is the number of counts on the ith class(label)
    :param ret_var: Whether to return the variance of the estimate
    :return: the variance of the node, as well as its estimated variance if ret_var
    """
    raise NotImplementedError("Not implemented until we do regression trees")
    n = np.sum(counts)
    if n == 0:
        if ret_var:
            return 0, 0
        return 0
    p = counts / n
    V_p = p * (1 - p) / n
    y_values = np.arange(len(counts))  # ith class label is assigned to the integer i
    V_target = np.sum((y_values ** 2) * p) - np.sum(y_values * p) ** 2
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        dV_target_dp = y_values[:-1] ** 2 - 2 * np.dot(y_values, p) * (
            y_values[:-1] - n
        )  # Note: len(dV_target_dp) is len(p) - 1 since p[-1] is dependent variable on p[:-1]
        V_V_target = np.dot(dV_target_dp, V_p[:-1])
        return float(V_target), float(V_V_target)
    return float(V_target)


def get_mse(
    targets_pile: List,
    ret_var: bool = False,
    pop_size: int = None,
) -> Union[Tuple[float, float], float]:
    """
    Compute the MSE for a given node, where the node is represented by the pile of all target values. Also Compute the
    confidence bound of our estimation by using Hoeffding's inequality for bounded values

    :param targets_pile: An array of all target values in the node
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :return: the mse(variance) of the node, as well as its estimated variance if ret_var
    """
    assert len(np.array(targets_pile).shape) == 1, "Invalid pile of target values"
    n = len(targets_pile)
    if n <= 1:
        if ret_var:
            return 0, 0
        return 0
    mse = float(
        scipy.stats.moment(targets_pile, 2)
    )  # 2nd central moment is mse with mean as a predicted value
    fourth_moment = float(scipy.stats.moment(targets_pile, 4))
    pop_var = mse
    if n == 2:
        # This variance comes from the variance of sample variance, see
        # https://math.stackexchange.com/questions/72975/variance-of-sample-variance
        # Use sample variance as an estimation of population variance.
        V_mse = (fourth_moment - pop_var ** 2) / 4
    else:
        # This variance comes from the variance of sample variance, see
        # https://en.wikipedia.org/wiki/Variance#Distribution_of_the_sample_variance
        # Use sample variance as an estimation of population variance.
        V_mse = (fourth_moment - (pop_var ** 2) * (n - 3) / (n - 1)) / n

    if pop_size is not None:
        assert pop_size >= n, "Sample size is greater than the population size"
        V_mse *= (pop_size - n) / (pop_size - 1)
    if ret_var:
        return mse, V_mse
    return mse


def get_impurity_fn(impurity_measure: str) -> Callable:
    if impurity_measure == GINI:
        get_impurity: Callable = get_gini
    elif impurity_measure == ENTROPY:
        get_impurity: Callable = get_entropy
    elif impurity_measure == VARIANCE:
        get_impurity: Callable = get_variance
    elif impurity_measure == MSE:
        get_impurity: Callable = get_mse
    else:
        Exception(
            "Did not assign any measure for impurity calculation in get_impurity_reduction function"
        )
    return get_impurity


def get_impurity_reductions(
    is_classification: bool,
    histogram: Histogram,
    bin_edge_idcs: List[int],
    ret_vars: bool = False,
    impurity_measure: str = "",
    pop_size: int = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Given a histogram of counts for each bin, compute the impurity reductions if we were to split a node on any of the
    histogram's bin edges.

    Impurity is measured either by Gini index or entropy

    :param is_classification: Whether the problem is a classification problem(True) or a regression problem(False)
    :returns: Impurity reduction when splitting node by bins in bin_edge_idcs
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :returns: Impurity reduction when splitting node by bins in _bin_edge_idcs
    """
    if impurity_measure == "":
        impurity_measure = GINI if is_classification else MSE
    get_impurity = get_impurity_fn(impurity_measure)

    h = histogram
    b = len(bin_edge_idcs)
    assert (
        b <= h.num_bins
    ), "len(bin_edges) whose impurity reductions we want to calculate is greater than len(total_bin_edges)"
    impurities_left = np.zeros(b)
    impurities_right = np.zeros(b)
    V_impurities_left = np.zeros(b)
    V_impurities_right = np.zeros(b)

    if is_classification:
        n = np.sum(h.left[0, :]) + np.sum(h.right[0, :])
    else:
        n = len(h.left_pile[0]) + len(h.right_pile[0])
    for i in range(b):
        b_idx = bin_edge_idcs[i]

        # Impurity is weighted by population of each node during a split
        if is_classification:
            left_weight = np.sum(h.left[b_idx, :])
            right_weight = np.sum(h.right[b_idx, :])
        else:
            left_weight = len(h.left_pile[b_idx])
            right_weight = len(h.right_pile[b_idx])

        # Population of left and right node is approximated by left_weight and right_weight
        left_size = None if pop_size is None else int(left_weight * pop_size / n)
        right_size = None if pop_size is None else int(right_weight * pop_size / n)
        left_weight /= n
        right_weight /= n
        if is_classification:
            IL, V_IL = get_impurity(h.left[b_idx, :], ret_var=True, pop_size=left_size)
            IR, V_IR = get_impurity(h.right[b_idx, :], ret_var=True, pop_size=right_size)
        else:
            IL, V_IL = get_impurity(h.left_pile[b_idx], ret_var=True, pop_size=left_size)
            IR, V_IR = get_impurity(h.right_pile[b_idx], ret_var=True, pop_size=right_size)

        impurities_left[i], V_impurities_left[i] = (
            float(left_weight * IL),
            float((left_weight ** 2) * V_IL),
        )
        impurities_right[i], V_impurities_right[i] = (
            float(right_weight * IR),
            float((right_weight ** 2) * V_IR),
        )

    if is_classification:
        impurity_curr, V_impurity_curr = get_impurity(
            h.left[0, :] + h.right[0, :], ret_var=True, pop_size=pop_size
        )
    else:
        impurity_curr, V_impurity_curr = get_impurity(
            h.left_pile[0] + h.right_pile[0], ret_var=True, pop_size=pop_size
        )
    impurity_curr = float(impurity_curr)
    V_impurity_curr = float(V_impurity_curr)
    # TODO(@motiwari): Might not need to subtract off impurity_curr
    #  since it doesn't affect reduction in a single feature?
    # (once best feature is determined)
    impurity_reductions = (impurities_left + impurities_right) - impurity_curr

    if ret_vars:
        # Note the last plus because Var(X-Y) = Var(X) + Var(Y) if X, Y are independent (this is an UNDERestimate)
        impurity_vars = V_impurities_left + V_impurities_right + V_impurity_curr
        return impurity_reductions, impurity_vars
    return impurity_reductions  # Jay: we can change the type of impurity_reductions to Tuple[np.ndarray] whose each array has size 1
