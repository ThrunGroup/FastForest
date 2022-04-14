from typing import Tuple, Union, List
import math
import numpy as np


def get_gini(
    counts: np.ndarray, ret_var: bool = False
) -> Union[Tuple[float, float], float]:
    """
    Compute the Gini impurity for a given node, where the node is represented by the number of counts of each class
    label. The Gini impurity is equal to 1 - sum_{i=1}^k (p_i^2)

    :param counts: 1d array of counts where ith element is the number of counts on the ith class(label).
    :param ret_var: Whether to the variance of the estimate
    :return: the Gini impurity of the node, as well as its estimated variance if ret_var
    """
    n = np.sum(counts)
    if n == 0:
        if ret_var:
            return 0, 0
        return 0
    p = counts / n
    V_p = p * (1 - p) / n
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


def get_entropy(counts: np.ndarray, ret_var=False) -> Union[Tuple[float, float], float]:
    """
    Compute the entropy impurity for a given node, where the node is represented by the number of counts of each class
    label. The entropy impurity is equal to - sum{i=1}^k (p_i * log_2 p_i)

    :param counts: 1d array of counts where ith element is the number of counts on the ith class(label)
    :param ret_var: Whether to the variance of the estimate
    :return: the entropy impurity of the node, as well as its estimated variance if ret_var
    """
    n = np.sum(counts)
    if n == 0:
        if ret_var:
            return 0, 0
        return 0

    p = counts / n
    V_p = p * (1 - p) / n
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
    :param ret_var: Whether to the variance of the estimate
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
