import ast
import os
import sys
from typing import List, Dict
from scaling_exps import make_scaling_plot

from utils.constants import FLIGHT, AIR, APS, BLOG, SKLEARN_REGRESSION, MNIST_STR, HOUSING, COVTYPE, KDD

pm = " \u00B1 "  # plus minus
ndigits = 3  # number of digits for rounding



def s(value: float, ndgits: int = ndigits):
    return str(round(value, ndigits))


def ordinal_num(num: int):
    """
    Change an integer to ordinal number string
    """
    assert num > 0, "number should be greater than 0"
    if num == 1:
        return "1st"
    elif num == 2:
        return "2nd"
    elif num == 3:
        return "3rd"
    else:
        return f"{num}th"


def print_table(headers: List, data: List):
    format_row = "{:<40}" * (len(headers) + 1)
    print(format_row.format("", *headers))
    for i, row in enumerate(data):
        print(format_row.format(f"{ordinal_num(int(i / 2 + 1))} row", *row))


def write_runtime_data(table_data: List, log_dict: Dict, filename: str):
    """
    A helper function for producing table 1 and 2 (runtime experiments).
    """
    ours_data = [
        filename[: filename.find("_")] + " + MABSplit",
        s(log_dict["our_avg_train_time"])
        + pm
        + s(log_dict["our_std_train_time"]),
        s(log_dict["our_avg_num_queries"])
        + pm
        + s(log_dict["our_std_num_queries"]),
        s(log_dict["our_avg_test"])
        + pm
        + s(log_dict["our_std_test"]),
    ]
    theirs_data = [
        filename[: filename.find("_")],
        s(log_dict["their_avg_train_time"])
        + pm
        + s(log_dict["their_std_train_time"]),
        s(log_dict["their_avg_num_queries"])
        + pm
        + s(log_dict["their_std_num_queries"]),
        s(log_dict["their_avg_test"])
        + pm
        + s(log_dict["their_std_test"]),
    ]
    table_data.append(theirs_data)
    table_data.append(ours_data)


def write_budget_data(table_data: List, log_dict: Dict, filename: str):
    """
    A helper function for producing table 3 and 4 (budget experiments).
    """
    ours_data = [
        filename[: filename.find("_")] + " + MABSplit",
        s(log_dict["our_avg_num_trees"])
        + pm
        + s(log_dict["our_std_num_trees"]),
        s(log_dict["our_avg_test"])
        + pm
        + s(log_dict["our_std_test"]),
    ]
    theirs_data = [
        filename[: filename.find("_")],
        s(log_dict["their_avg_num_trees"])
        + pm
        + s(log_dict["their_std_num_trees"]),
        s(log_dict["their_avg_test"])
        + pm
        + s(log_dict["their_std_test"]),
    ]
    table_data.append(theirs_data)
    table_data.append(ours_data)


def produce_table1():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    runtime_logs_dir = os.path.join(this_dir, "runtime_exps", "logs")
    header = ["Model", "Time(s)", "# insertions", "Accuracy"]
    classification_models = ["HRFC", "HRPC", "ERFC"]
    for dataset in [COVTYPE, MNIST_STR, APS, FLIGHT]:
        filename_list = [dataset + "_" + c_m + "_dict" for c_m in classification_models]
        table1_data = []
        for filename in filename_list:
            with open(os.path.join(runtime_logs_dir, filename), "r") as fin:
                log_dict = ast.literal_eval(fin.read())
                write_runtime_data(table1_data, log_dict, filename)
        print("=" * 30)
        print("Table 1 Classification: " + dataset)
        print_table(header, table1_data)


def produce_table2():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    runtime_logs_dir = os.path.join(this_dir, "runtime_exps", "logs")
    header = ["Model", "Time(s)", "# insertions", "MSE"]
    regression_models = ["HRFR", "HRPR", "ERFR"]
    for dataset in [SKLEARN_REGRESSION, AIR, BLOG]:
        filename_list = [dataset + "_" + r_m + "_dict" for r_m in regression_models]
        table2_data = []
        for filename in filename_list:
            with open(os.path.join(runtime_logs_dir, filename), "r") as fin:
                log_dict = ast.literal_eval(fin.read())
                write_runtime_data(table2_data, log_dict, filename)
        print("=" * 30)
        print("Table 2 Regression: " + dataset)
        print_table(header, table2_data)


def produce_table3():
    dir = "budget_exps"
    filename_list = ["HRFC_dict", "ERFC_dict", "HRPC_dict"]
    header = ["Model", "# trees", "Accuracy"]
    table3_data = []
    for filename in filename_list:
        with open(os.path.join(dir, filename), "r") as fin:
            log_dict = ast.literal_eval(fin.read())
            write_budget_data(table3_data, log_dict, filename)
    print("=" * 30)
    print("Table 3 Classification: MNIST (budget = 10M)")
    print_table(header, table3_data)


def produce_table4():
    dir = "budget_exps"
    filename_list = ["HRFR_dict", "ERFR_dict", "HRPR_dict"]
    header = ["Model", "# trees", "MSE"]
    table4_data = []
    for filename in filename_list:
        with open(os.path.join(dir, filename), "r") as fin:
            log_dict = ast.literal_eval(fin.read())
            write_budget_data(table4_data, log_dict, filename)
    print("=" * 30)
    print("Table 4 Classification: Random Linear (budget = Q * 24M)")
    print_table(header, table4_data)


def produce_table5():
    dir = os.path.join(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"
        ),
        "stat_test_stability_log",
    )
    filename_list = [
        "HRFC+MID_dict",
        "HRFR+MID_dict",
        "HRFC+Perm_dict",
        "HRFR+Perm_dict",
    ]
    header = ["Importance Model", "Dataset", "Stability"]
    table5_data = []
    for filename in filename_list:
        with open(os.path.join(dir, filename), "r+") as fin:
            log_dict = ast.literal_eval(fin.read())
            if "RFR" in filename:
                dataset = "Random Regression"
            else:
                dataset = "Random Classification"
            avg_stab_ours = (float(log_dict["lb_mab"]) + float(log_dict["ub_mab"])) / 2
            avg_stab_theirs = (
                                      float(log_dict["lb_exact"]) + float(log_dict["ub_exact"])
                              ) / 2
            std_stab_ours = (
                    abs(float(log_dict["lb_mab"]) - float(log_dict["ub_mab"])) / 2
            )
            std_stab_theirs = (
                    abs(float(log_dict["lb_exact"]) - float(log_dict["ub_exact"])) / 2
            )
            ours_data = [
                filename[: filename.find("_")] + " + MABSplit",
                dataset,
                s(avg_stab_ours)
                + pm
                + s(std_stab_ours),
            ]
            theirs_data = [
                filename[: filename.find("_")],
                dataset,
                s(avg_stab_theirs)
                + pm
                + s(std_stab_theirs),
            ]
            table5_data.append(theirs_data)
            table5_data.append(ours_data)
    print("=" * 30)
    print("Table 5 Stability Model (Budget: Q * 100000)")
    print_table(header, table5_data)


def produce_table6():
    dir = "sklearn_exps"
    filename_list = ["RFC_dict", "ERFC_dict", "RFR_dict", "ERFR_dict"]
    header = ["Model", "Task and Dataset", "Performance Metric", "Test Performance"]
    table6_data = []
    for filename in filename_list:
        with open(os.path.join(dir, filename), "r+") as fin:
            log_dict = ast.literal_eval(fin.read())
            if "RFR" in filename:
                dataset = "Regression: California Housing"
                metric = "MSE"
            else:
                dataset = "Classification: 20 Newsgroups"
                metric = "Accuracy"
            ours_data = [
                filename[: filename.find("_")] + "(Ours)",
                dataset,
                metric,
                s(log_dict["our_avg_train"])
                + pm
                + s(log_dict["our_std_train"]),
            ]
            theirs_data = [
                filename[: filename.find("_")] + "(Sklearn)",
                dataset,
                metric,
                s(log_dict["their_avg_train"])
                + pm
                + s(log_dict["their_std_train"]),
            ]
            table6_data.append(theirs_data)
            table6_data.append(ours_data)
    print("=" * 30)
    print("Table 6 Compare our model vs sklearn")
    print_table(header, table6_data)


def produce_figure1():
    make_scaling_plot.main(filename=os.path.join("scaling_exps", "HRFC_size_to_time_dict"))
    make_scaling_plot.main(
        filename=os.path.join("scaling_exps", "HRFR_size_to_time_dict_regression")
    )


if __name__ == "__main__":
    produce_table1()
    produce_table2()
    produce_table3()
    produce_table4()
    produce_table5()
    produce_table6()
    produce_figure1()
