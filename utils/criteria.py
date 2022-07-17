from typing import Tuple, List, Callable, Union
import scipy
import numpy as np

from data_structures.histogram import Histogram
from utils.constants import GINI, ENTROPY, MSE, KURTOSIS


def get_gini(
        counts_vec: np.ndarray,
        ret_var: bool = False,
        pop_size: np.ndarray = None,
        n: np.ndarray = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compute the Gini impurity for a given node, where the node is represented by the number of counts of each class
    label. The Gini impurity is equal to 1 - sum_{i=1}^k (p_i^2)

    :param counts_vec: 2d array of counts where (i, j)th element is the number of counts on the jth class(label)
    of ith bin.
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :param n: 1d array of the sum of counts for each bin.
    :return: the Gini impurity of the node, as well as its estimated variance if ret_var
    """
    if len(counts_vec.shape) == 1:
        counts_vec = np.expand_dims(counts_vec, 0)
    if n is None:
        n = np.sum(counts_vec, axis=1, dtype=np.int64)
    p = counts_vec / np.expand_dims(n, axis=1)  # Expand dimension to broadcast
    p = np.nan_to_num(p, nan=0, posinf=0, neginf=0)  # Deal with the case when n = 0
    G = 1 - np.sum(p * p, axis=1)
    if ret_var:
        V_p = p * (1 - p) / np.expand_dims(n, axis=1)
        if pop_size is not None:
            # Use FPC for variance calculation, see
            # https://stats.stackexchange.com/questions/376417/sampling-from-finite-population-with-replacement
            V_p *= np.expand_dims((pop_size - n) / (pop_size - 1), axis=1)  # Expand dimension to broadcast

        # This variance comes from propagation of error formula, see
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
        dG_dp = -2 * p[:, :-1] + 2 * np.expand_dims(
            p[:, -1], axis=1
        )  # Note: len(dG_dp) is len(p) - 1 since p[-1] is dependent variable on p[:-1]
        V_G = np.sum(dG_dp ** 2 * V_p[:, :-1], axis=1)
        return G, V_G
    return G


def get_entropy(
        counts_vec: np.ndarray,
        ret_var=False,
        pop_size: int = None,
        n: int = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compute the entropy impurity for a given node, where the node is represented by the number of counts of each class
    label. The entropy impurity is equal to - sum{i=1}^k (p_i * log_2 p_i)

    :param counts_vec: 2d array of counts where (i, j)th element is the number of counts on the jth class(label)
    of ith bin.
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :param n: The sum of counts.
    :return: the entropy impurity of the node, as well as its estimated variance if ret_var
    """
    if len(counts_vec.shape) == 1:
        counts_vec = np.expand_dims(counts_vec, 0)
    if n is None:
        n = np.sum(counts_vec, axis=1, dtype=np.int64)
    p = counts_vec / np.expand_dims(n, axis=1)  # Expand dimension to broadcast
    p = np.nan_to_num(p, nan=0, posinf=0, neginf=0)  # Deal with the case when n = 0
    log_p = np.nan_to_num(np.log(p), nan=0, posinf=0, neginf=0)
    E = np.sum(-log_p * p, axis=1)  # Note: when p -> 0, (-log(p) * p ) -> 0
    if ret_var:
        V_p = p * (1 - p) / np.expand_dims(n, axis=1)
        if pop_size is not None:
            # Use FPC for variance calculation, see
            # https://stats.stackexchange.com/questions/376417/sampling-from-finite-population-with-replacement
            V_p *= np.expand_dims((pop_size - n) / (pop_size - 1), axis=1)  # Expand dimension to broadcast

        # This variance comes from propagation of error formula, see
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification
        dE_dp = (
                -log_p[:, -1] + np.expand_dims(log_p[-1], axis=1)
        )  # Note: len(dE_dp) is len(p) - 1 since p[-1] is dependent variable on p[:-1]
        V_E = np.sum(dE_dp ** 2 * V_p[:, :-1], axis=1)
        return E, V_E
    return E


def get_mse(
        args: np.ndarray,
        ret_var: bool = False,
        pop_size: np.ndarray = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compute the MSE for a given node, where the node is represented by the pile of all target values. Also Compute the
    confidence bound of our estimation by using Hoeffding's inequality for bounded values

    :param args: args[i] = (number of samples, mean of samples, variance of samples) of ith bin
    :param ret_var: Whether to return the variance of the estimate
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :return: the mse(variance) of the node, as well as its estimated variance if ret_var
    """
    n = args[0]
    second_moment = args[2]
    if pop_size is None:
        estimated_mse = (
                second_moment * n / (n - 1)
        )  # 2nd central moment is mse with mean as a predicted value and use Bessel's correction
    else:
        assert np.all(pop_size >= n), "population size should not be less than sample size"
        estimated_mse = second_moment * (n * (pop_size - 1)) / ((n - 1) * pop_size)  # Derive myself with reference to
        # https://stats.stackexchange.com/questions/5158/explanation-of-finite-population-correction-factor
    estimated_mse = np.nan_to_num(estimated_mse, nan=0, posinf=0, neginf=0)  # Deal with case when n = 1
    if ret_var:
        estimated_fourth_moment = (
                KURTOSIS * estimated_mse ** 2
        )  # see https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments
        if pop_size is not None:
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
            V_mse = c1 * estimated_fourth_moment + c3 * (estimated_mse ** 2)
        else:
            # This variance comes from the variance of sample variance, see
            # https://en.wikipedia.org/wiki/Variance#Distribution_of_the_sample_variance
            # Use sample variance as an estimation of population variance.
            V_mse = (estimated_fourth_moment - (estimated_mse ** 2) * (n - 3) / (n - 1)) / n
        V_mse = np.nan_to_num(V_mse, nan=0, posinf=0, neginf=0)  # Deal with the case when n <= 3
        # or pop_size = n
        return estimated_mse, V_mse
    return estimated_mse


def get_impurity_fn(impurity_measure: str) -> Callable:
    if impurity_measure == GINI:
        get_impurity: Callable = get_gini
    elif impurity_measure == ENTROPY:
        get_impurity: Callable = get_entropy
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
        is_vectorization: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Given a histogram of counts for each bin, compute the impurity reductions if we were to split a node on any of the
    histogram's bin edges. Impurity is measured either by Gini index or entropy

    :param is_classification: Whether the problem is a classification problem(True) or a regression problem(False)
    :param histogram: Histogram class object
    :param bin_edge_idcs: Bin edge indices that we consider
    :param ret_vars: Whether to return variance
    :param impurity_measure: A type of impurity measure
    :param pop_size: The size of population size to do FPC(Finite Population Correction). If None, don't do FPC.
    :param is_vectorization: Whether to use vectorization
    :returns: Impurity reduction when splitting node by bins in _bin_edge_idcs
    """
    if impurity_measure == "":
        impurity_measure = GINI if is_classification else MSE
    get_impurity = get_impurity_fn(impurity_measure)
    if is_vectorization:
        h = histogram
        if is_classification:
            left = h.left[bin_edge_idcs]
            right = h.right[bin_edge_idcs]
            left_sum = np.sum(left, axis=1, dtype=np.int64)
            right_sum = np.sum(right, axis=1, dtype=np.int64)
        else:
            left = h.left_pile[bin_edge_idcs]
            right = h.right_pile[bin_edge_idcs]
            left_sum = left[:, 0]
            right_sum = right[:, 0]
            n = left_sum + right_sum
        n = left_sum + right_sum

        # Population of left and right node is approximated by left_weight and right_weight
        if pop_size is None:
            left_size = None
            right_size = None
        else:
            left_size = (left_sum * pop_size / n).astype(np.int64)
            right_size = (right_sum * pop_size / n).astype(np.int64)
        left_weight = left_sum / n
        right_weight = right_sum / n

        if is_classification:
            left_impurity, left_var = get_impurity(
                counts_vec=left, ret_var=True, pop_size=left_size, n=left_sum
            )
            right_impurity, right_var = get_impurity(
                counts_vec=right, ret_var=True, pop_size=right_size, n=right_sum
            )
            curr_impurity, curr_var = get_impurity(
                counts_vec=left[0] + right[0], ret_var=True, pop_size=pop_size
            )
        else:
            left_impurity, left_var = get_impurity(
                args=left, ret_var=True, pop_size=left_size
            )
            right_impurity, right_var = get_impurity(
                args=right, ret_var=True, pop_size=right_size
            )
            curr_impurity, curr_var = get_impurity(
                args=h.curr_pile, ret_var=True, pop_size=pop_size
            )
        left_impurity *= left_weight
        left_var *= left_weight ** 2
        right_impurity *= right_weight
        right_var *= right_weight ** 2
        return (
            (left_impurity + right_impurity - curr_impurity),
            (curr_var + left_var + right_var),
        )

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
    return impurity_reductions
