# import pandas as pd
# import os
# import json
#
# dir = os.path.join("experiments", "tables")
# os.makedirs(dir, exist_ok=True)
#
# # Produce Table3
# filename = os.path.join(dir, "table3.csv")
# if os.path.exists(filename):
#     prev_table3 = pd.read_csv(filename)
#
# source_dir = os.path.join("experiments", "budget_experiments")
# with open(os.path.join(source_dir, "budget_HRFC_dict"), "r+") as f:
#     row1_dict = json.loads(f.read())
#
# # Produce Table4

import ast
import os
import sys
from typing import List


def print_table(headers: List, data: List):
    format_row = "{:<40}" * (len(headers) + 1)
    print(format_row.format("", *headers))
    for i, row in enumerate(data):
        print(format_row.format(f"{int(i / 2 + 1)}th row", *row))


# Produce Table1
dir = "runtime_exps"
filename_list = ["HRFC_dict", "ERFC_dict", "HRPC_dict"]
header = ["Model", "Time(s)", "# insertions", "Accuracy"]
table1_data = []
pm = "\u00B1"  # plus minus
for filename in filename_list:
    with open(os.path.join(dir, filename), "r") as fin:
        log_dict = ast.literal_eval(fin.read())
        ours_data = [
            filename[:4] + " + MABSplit",
            str(round(log_dict["our_avg_train_time"], 3))
            + pm
            + str(round(log_dict["our_std_train_time"], 3)),
            str(round(log_dict["our_avg_num_queries"], 3))
            + pm
            + str(round(log_dict["our_std_num_queries"], 3)),
            str(round(log_dict["our_avg_test"], 3))
            + pm
            + str(round(log_dict["our_std_test"], 3)),
        ]
        theirs_data = [
            filename[:4],
            str(round(log_dict["their_avg_train_time"], 3))
            + pm
            + str(round(log_dict["their_std_train_time"], 3)),
            str(round(log_dict["their_avg_num_queries"], 3))
            + pm
            + str(round(log_dict["their_std_num_queries"], 3)),
            str(round(log_dict["their_avg_test"], 3))
            + pm
            + str(round(log_dict["their_std_test"], 3)),
        ]
        table1_data.append(theirs_data)
        table1_data.append(ours_data)
print("=" * 30)
print("Table1 Classification: MNIST")
print_table(header, table1_data)

# Produce Table2
dir = "runtime_exps"
filename_list = ["HRFR_dict", "ERFR_dict", "HRPR_dict"]
header = ["Model", "Time(s)", "# insertions", "MSE"]
table2_data = []
for filename in filename_list:
    with open(os.path.join(dir, filename), "r") as fin:
        log_dict = ast.literal_eval(fin.read())
        ours_data = [
            filename[:4] + " + MABSplit",
            str(round(log_dict["our_avg_train_time"], 3))
            + pm
            + str(round(log_dict["our_std_train_time"], 3)),
            str(round(log_dict["our_avg_num_queries"], 3))
            + pm
            + str(round(log_dict["our_std_num_queries"], 3)),
            str(round(log_dict["our_avg_test"], 3))
            + pm
            + str(round(log_dict["our_std_test"], 3)),
        ]
        theirs_data = [
            filename[:4],
            str(round(log_dict["their_avg_train_time"], 3))
            + pm
            + str(round(log_dict["their_std_train_time"], 3)),
            str(round(log_dict["their_avg_num_queries"], 3))
            + pm
            + str(round(log_dict["their_std_num_queries"], 3)),
            str(round(log_dict["their_avg_test"], 3))
            + pm
            + str(round(log_dict["their_std_test"], 3)),
        ]
        table2_data.append(theirs_data)
        table2_data.append(ours_data)
print("=" * 30)
print("Table2 Regression: Random Linear Model")
print_table(header, table2_data)

# Produce Table3
dir = "budget_exps"
filename_list = ["HRFC_dict", "ERFC_dict", "HRPC_dict"]
header = ["Model", "# trees", "Accuracy"]
table3_data = []
for filename in filename_list:
    with open(os.path.join(dir, filename), "r") as fin:
        log_dict = ast.literal_eval(fin.read())
        ours_data = [
            filename[:4] + " + MABSplit",
            str(round(log_dict["our_avg_num_trees"], 3))
            + pm
            + str(round(log_dict["our_std_num_trees"], 3)),
            str(round(log_dict["our_avg_test"], 3))
            + pm
            + str(round(log_dict["our_std_test"], 3)),
        ]
        theirs_data = [
            filename[:4],
            str(round(log_dict["their_avg_num_trees"], 3))
            + pm
            + str(round(log_dict["their_std_num_trees"], 3)),
            str(round(log_dict["their_avg_test"], 3))
            + pm
            + str(round(log_dict["their_std_test"], 3)),
        ]
        table3_data.append(theirs_data)
        table3_data.append(ours_data)
print("=" * 30)
print("Table3 Classification: MNIST (budget = 10M)")
print_table(header, table3_data)

# Produce Table4
dir = "budget_exps"
filename_list = ["HRFR_dict", "ERFR_dict", "HRPR_dict"]
header = ["Model", "# trees", "Accuracy"]
table4_data = []
for filename in filename_list:
    with open(os.path.join(dir, filename), "r") as fin:
        log_dict = ast.literal_eval(fin.read())
        ours_data = [
            filename[:4] + " + MABSplit",
            str(round(log_dict["our_avg_num_trees"], 3))
            + pm
            + str(round(log_dict["our_std_num_trees"], 3)),
            str(round(log_dict["our_avg_test"], 3))
            + pm
            + str(round(log_dict["our_std_test"], 3)),
        ]
        theirs_data = [
            filename[:4],
            str(round(log_dict["their_avg_num_trees"], 3))
            + pm
            + str(round(log_dict["their_std_num_trees"], 3)),
            str(round(log_dict["their_avg_test"], 3))
            + pm
            + str(round(log_dict["their_std_test"], 3)),
        ]
        table4_data.append(theirs_data)
        table4_data.append(ours_data)
print("=" * 30)
print("Table4 Classification: Random Linear (budget = 24M)")
print_table(header, table4_data)

# Produce Table 5
dir = os.path.join(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"),
    "stat_test_stability_log",
)
filename_list = ["HRFC+MID_dict", "HRFR+MID_dict", "HRFC+Perm_dict", "HRFR+Perm_dict"]
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
        avg_stab_theirs = (float(log_dict["lb_exact"]) + float(log_dict["ub_exact"])) / 2
        std_stab_ours = (float(log_dict["lb_mab"]) - float(log_dict["ub_mab"])) / 2
        std_stab_theirs = (float(log_dict["lb_exact"]) - float(log_dict["ub_exact"])) / 2
        ours_data = [
            filename[:filename.find("_")] + " + MABSplit",
            dataset,
            str(round(avg_stab_ours, 3))
            + pm
            + str(round(std_stab_ours, 3)),
        ]
        theirs_data = [
            filename[:filename.find("_")] + " + MABSplit",
            dataset,
            str(round(avg_stab_theirs, 3))
            + pm
            + str(round(std_stab_theirs, 3)),
        ]
        table5_data.append(theirs_data)
        table5_data.append(ours_data)
print("=" * 30)
print("Table5 Stability Model (Budget: Q * 100000)")
print_table(header, table5_data)

# Produce Table 6
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
            filename[:filename.find("_")] + "(Ours)",
            dataset,
            metric,
            str(round(log_dict["our_avg_train"], 3))
            + pm
            + str(round(log_dict["our_std_train"], 3)),
        ]
        theirs_data = [
            filename[:filename.find("_")] + "(Sklearn)",
            dataset,
            metric,
            str(round(log_dict["their_avg_train"], 3))
            + pm
            + str(round(log_dict["their_std_train"], 3)),
        ]
        table6_data.append(theirs_data)
        table6_data.append(ours_data)
print("=" * 30)
print("Table6 Compare our model vs sklearn")
print_table(header, table6_data)

# Figure 1
from scaling_exps import make_scaling_plot

make_scaling_plot.main(filename=os.path.join("scaling_exps", "size_to_time_dict"))
make_scaling_plot.main(filename=os.path.join("scaling_exps", "size_to_time_dict_regression"))
