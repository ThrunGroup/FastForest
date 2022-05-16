# Create a new file to avoid circular imports
def welford_variance_calc(
    n1: int, mean1: float, var1: float, n2: int, mean2: float, var2: float
) -> float:
    """
    Calculate the variance of A U B given |A|, |B|, mean(A), mean(B), var(A), and var(B). See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    if var1 == 0:
        return var2
    elif var2 == 0:
        return var1

    n = n1 + n2
    delta = mean1 - mean2
    M1 = n1 * var1
    M2 = n2 * var2
    M = M1 + M2 + (delta ** 2) * (n1 * n2) / n
    return M / n
