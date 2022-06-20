from typing import Tuple, List, Callable, Union
import scipy
import numpy as np
from numba import jit

from data_structures.histogram import Histogram
from utils.constants import GINI, ENTROPY, VARIANCE, MSE, KURTOSIS


@jit
def get_gini(
    counts: np.ndarray, ret_var: bool = False, pop_size: int = None, n: int = None,
) -> Union[Tuple[float, float], float]:
    """
    Compute the Gini impurity for a given node, where the node is represented by the number of counts of each class
    label. The Gini impurity is equal to 1 - sum_{i=1}^k (p_i^2)

    :param counts: 1d array of counts where ith element is the number of counts on the ith class(label).
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :param n: The sum of counts.
    :return: the Gini impurity of the node, as well as its estimated variance if ret_var
    """
    if n is None:
        n = np.sum(counts, dtype=np.int64)
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


def get_entropy(
    counts: np.ndarray, ret_var=False, pop_size: int = None, n: int = None,
) -> Union[Tuple[float, float], float]:
    """
    Compute the entropy impurity for a given node, where the node is represented by the number of counts of each class
    label. The entropy impurity is equal to - sum{i=1}^k (p_i * log_2 p_i)

    :param counts: 1d array of counts where ith element is the number of counts on the ith class(label)
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :param n: The sum of counts.
    :return: the entropy impurity of the node, as well as its estimated variance if ret_var
    """
    if n is None:
        n = np.sum(counts, dtype=np.int64)
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
    args: np.ndarray, ret_var: bool = False, pop_size: int = None,
) -> Union[Tuple[float, float], float]:
    """
    Compute the MSE for a given node, where the node is represented by the pile of all target values. Also Compute the
    confidence bound of our estimation by using Hoeffding's inequality for bounded values

    :param args: args = (number of samples, mean of samples, variance of samples)
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :return: the mse(variance) of the node, as well as its estimated variance if ret_var
    """
    n = args[0]
    second_moment = args[2]
    fourth_moment = (
        3 * second_moment ** 2
    )  # see https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments

    if n <= 1:
        if ret_var:
            return 0, 0
        return 0

    if pop_size == n:
        if ret_var:
            return second_moment, 0
        return second_moment
    estimated_mse = (
        second_moment * n / (n - 1)
    )  # 2nd central moment is mse with mean as a predicted value and use Bessel's correction

    if pop_size is not None and pop_size > 3:
        assert pop_size >= n, "Sample size is greater than population size"
        # Todo: Add a formula when pop_size is equal to 3.
        N = pop_size
        c1 = (
            N
            * (N - n)
            * (N * n - N - n - 1)
            / (n * (n - 1) * (N - 1) * (N - 2) * (N - 3))
        )
        c3 = -(
            N
            * (N - n)
            * ((N ** 2) * n - 3 * n - 3 * (N ** 2) + 6 * N - 3)
            / (n * (n - 1) * ((N - 1) ** 2) * (N - 2) * (N - 3))
        )
        # Use the formula of the variance of sample variance when sampled with replacement, see
        # https://www.asasrms.org/Proceedings/y2008/Files/300992.pdf#page=2
        V_mse = c1 * fourth_moment + c3 * (second_moment ** 2)

        # Derive myself with reference to
        # https://stats.stackexchange.com/questions/5158/explanation-of-finite-population-correction-factor
        estimated_mse = estimated_mse * (pop_size - 1) / pop_size
    else:
        if n == 2:
            # This variance comes from the variance of sample variance, see
            # https://math.stackexchange.com/questions/72975/variance-of-sample-variance
            # Use sample variance as an estimation of population variance.
            V_mse = (fourth_moment + estimated_mse ** 2) / 2
        else:
            # This variance comes from the variance of sample variance, see
            # https://en.wikipedia.org/wiki/Variance#Distribution_of_the_sample_variance
            # Use sample variance as an estimation of population variance.
            V_mse = (fourth_moment - (second_moment ** 2) * (n - 3) / (n - 1)) / n
    if ret_var:
        return estimated_mse, V_mse
    return estimated_mse


def get_mse_with_chi(
    args: np.ndarray, ret_var: bool = False, pop_size: int = None,
) -> Union[Tuple[float, float], float]:
    """
    Compute the MSE for a given node, where the node is represented by the pile of all target values. Also Compute the
    confidence bound of our estimation by using Hoeffding's inequality for bounded values. Assume normal distribution
    of labels!!

    :param args: args = (number of samples, mean of samples, variance of samples)
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :return: the mse(variance) of the node, as well as its estimated variance if ret_var
    """
    n = args[0]
    mean = args[1]
    if pop_size is None:
        pop_var = args[2] * n / (n - 1)
        sample_var = args[2]
    else:
        pop_var = args[2] * n / (n - 1) * (pop_size - 1) / pop_size
        sample_var = args[2]


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
    histogram's bin edges. Impurity is measured either by Gini index or entropy

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
        n = int(
            np.sum(h.left[0, :], dtype=np.int64) + np.sum(h.right[0, :], dtype=np.int64)
        )
    else:
        n = int(h.left_pile[0][0]) + int(h.right_pile[0][0])
    for i in range(b):
        b_idx = bin_edge_idcs[i]

        # Impurity is weighted by population of each node during a split
        if is_classification:
            left_sum = int(np.sum(h.left[b_idx, :], dtype=np.int64))
            right_sum = int(np.sum(h.right[b_idx, :], dtype=np.int64))
        else:
            left_sum = int(h.left_pile[b_idx][0])
            right_sum = int(h.right_pile[b_idx][0])

        # Population of left and right node is approximated by left_weight and right_weight
        left_size = None if pop_size is None else int(left_sum * pop_size / n)
        right_size = None if pop_size is None else int(right_sum * pop_size / n)
        left_weight = left_sum / n
        right_weight = right_sum / n
        if is_classification:
            IL, V_IL = get_impurity(
                h.left[b_idx, :], ret_var=True, pop_size=left_size, n=left_sum
            )
            IR, V_IR = get_impurity(
                h.right[b_idx, :], ret_var=True, pop_size=right_size, n=right_sum
            )
        else:
            IL, V_IL = get_impurity(
                h.left_pile[b_idx], ret_var=True, pop_size=left_size
            )
            IR, V_IR = get_impurity(
                h.right_pile[b_idx], ret_var=True, pop_size=right_size
            )

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
            h.curr_pile, ret_var=True, pop_size=pop_size
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
