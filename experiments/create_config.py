"""
Convenience code to automatically generate a list of experiments to run.
Default output file is  auto_exp_config.py.
"""

import itertools
import numpy as np
from typing import List


def q(str_: str) -> str:
    """
    Helper function that prepends and appends double quotes to a string.

    :param str_: string to put double quotes around
    """
    return '"' + str_ + '"'


def list_to_str(input_: List[int]) -> str:
    """
    Convenience function for printing an array of numbers to the same number of decimal places. Return a string
    representation of the given input array for easy printing.
    :param input_: input array to convert to a string
    :return: string representation of the array
    """
    return ", ".join(map(str, map(lambda num: np.round(num, decimals=5), input_)))


def dict_to_str(params: dict) -> str:
    params_str = "{"
    for p in params:
        params_str += str(p) + ": " + params[p] + ","
    params_str += "}"
    return params_str


def exp_string(
    classification: bool,
    algo: str,
    metric_name: str,
    metric_value: float,
    params: dict,
) -> str:
    """
    :param classification:
    :param metric_name:
    :param metric_value:
    :param algo:
    :param params:
    :return: string to be written to exp file
    """
    task = q("C") if classification else q("R")  # Classification or regression
    param_str = dict_to_str(params)
    return (
        "\t["
        + ", ".join(map(str, [task, q(metric_name), metric_value, q(algo), param_str]))
        + "],\n"
    )


def main() -> None:
    """
    Write the experiments you want to run to a file that will be ingested by run_exps.py.

    Modify the arrays below to modify the generated experiments.

    :return: None
    """
    tasks = [True]  # Classification or regression
    algos = ["FastForest"]
    metric_names = ["Budget"]
    metric_values = [10000]
    params = [{}]
    exps = itertools.product(tasks, algos, metric_names, metric_values, params)
    with open("auto_exp_config.py", "w+") as fout:
        fout.write("experiments = [\n")
        for e in exps:
            exp = exp_string(*e)
            fout.write(exp)
        fout.write("]")


if __name__ == "__main__":
    main()
