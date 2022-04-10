from typing import Tuple, Union
import math


def get_gini(zero_count: int, one_count: int, ret_var: bool = False) -> Union[Tuple[float, float], float]:
    """
    Compute the Gini impurity for a given node, where the node is represented by the number of counts of each class
    label. The Gini impurity is equal to 1 - \sum_{i=1}^k (p_i^2)

    :param zero_count: Number of zeros in the node
    :param one_count: Number of ones in the node
    :param ret_var: Whether to the variance of the estimate
    :return: the Gini impurity of the node, as well as its estimated variance if ret_var
    """
    # When p0 = 0 or p1 = 0, gini impurity and its variance should be equal to 0
    if zero_count == 0 or one_count == 0:
        if ret_var:
            return 0, 0  # We have to think about its variance as 0 variance means we have no confidence bound
        return 0
    n = zero_count + one_count
    p0 = zero_count / n
    p1 = one_count / n
    V_p0 = p0 * (1 - p0) / n  # Assuming the independence
    G = 1 - p0 ** 2 - p1 ** 2
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        V_G = (-2 * p0 + 2 * p1) ** 2 * V_p0
        return G, V_G
    return G


def get_entropy(zero_count: int, one_count: int, ret_var=False) -> Union[Tuple[float, float], float]:
    """
    Compute the entropy impurity for a given node, where the node is represented by the number of counts of each class
    label. The entropy impurity is equal to - \sum{i=1}^k (p_i * \log_2 p_i)

    :param zero_count: Number of zeros in the node
    :param one_count: Number of ones in the node
    :param ret_var: Whether to the variance of the estimate
    :return: the entropy impurity of the node, as well as its estimated variance if ret_var
    """
    if zero_count == 0 or one_count == 0:
        if ret_var:
            return 0, 0  # We have to think about its variance as 0 variance means we have no confidence bound
        return 0
    n = zero_count + one_count
    p0 = zero_count / n
    p1 = one_count / n
    V_p0 = p0 * (1 - p0) / n
    I = - math.log(x=p0) * p0 - math.log(x=p1) * p1
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        V_I = (- math.log(p0) + math.log(p1)) ** 2 * V_p0
        return I, V_I
    return I


def get_variance(zero_count: int, one_count: int, ret_var=False) -> Union[Tuple[float, float], float]:
    """
    Compute the variance for a given node, where the node is represented by the number of counts of each class
    label.

    :param zero_count: Number of zeros in the node
    :param one_count: Number of ones in the node
    :param ret_var: Whether to the variance of the estimate
    :return: the variance of the node, as well as its estimated variance if ret_var
    """
    if zero_count == 0 or one_count == 0:
        if ret_var:
            return 0, 0  # We have to think about its variance as 0 variance means we have no confidence bound
        return 0
    n = zero_count + one_count
    p0 = zero_count / n
    p1 = one_count / n
    V_target = p0 * (1 - p0)  # Assume that each target is from bernoulli distribution
    # This variance comes from propagation of error formula, see
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
    if ret_var:
        V_V_target = (1 - 2 * p0) ** 2 * V_target
        return V_target, V_V_target
    return V_target
