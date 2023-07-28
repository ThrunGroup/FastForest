import pickle
import random
import argparse
import sys
import time

import typing
from typing import Sequence, Tuple

import data_structures
from data_structures import forest_classifier
from data_structures.forest_classifier import ForestClassifier as FastForestClassifier

import exp_utils
from exp_utils import get_datasets, compute_RF_accuracy, get_log_file_filepath

TIME_TO_MS = 1000

# Experiment with the FastForest codebase using parameters and return 
# a list of string-based results for pr
def exp_and_results(all_data: Tuple[Sequence[float], Sequence[int], Sequence[float], Sequence[float]], 
                    n_estimators: int, 
                    max_depth : int, 
                    seed : int, 
                    log_split: bool, 
                    precision: int = 3) -> Tuple[Sequence[str], Sequence[float]]:
    random.seed(seed)
    X_train, Y_train, X_test, Y_test = all_data

    print("TRAINING FOREST")
    train_start = time.time()
    FFC = FastForestClassifier(data = X_train, labels = Y_train, n_estimators = n_estimators, max_depth = max_depth, use_logarithmic_split = log_split)
    FFC.fit()
    train_end   = time.time()
    train_time  = round((train_end - train_start) * TIME_TO_MS, precision)
    print("TRAINING COMPLETE")

    print("EVALUATING FOREST")
    train_eval_start = time.time()
    train_accuracy   = round(compute_RF_accuracy(FFC, X_train, Y_train) * 100, precision)
    train_eval_end   = time.time()
    test_eval_start  = time.time()
    test_accuracy    = round(compute_RF_accuracy(FFC, X_test, Y_test) * 100, precision)
    test_eval_end    = time.time()
    train_eval_time  = round((train_eval_end - train_eval_start) * TIME_TO_MS, precision)
    test_eval_time   = round((test_eval_end - test_eval_start) * TIME_TO_MS, precision)
    print("EVALUATION COMPLETE")

    train_time_res = "Time to train forest: " + str(train_time) + "ms"
    train_eval_time_res = "Time to eval forest on train set: " + str(train_eval_time) + "ms"
    test_eval_time_res  = "Time to eval forest on test set: " + str(test_eval_time) + "ms"
    train_accuracy_res  = "Accuracy on train set: " + str(train_accuracy) + "%"
    test_accuracy_res   = "Accuracy on test set: " + str(test_accuracy) + "%"
    str_results = [train_time_res, train_eval_time_res, test_eval_time_res, train_accuracy_res, test_accuracy_res]
    pure_results = {"train_time" : train_time, 
                    "train_eval_time" : train_eval_time, "test_eval_time" : test_eval_time, 
                    "train_accuracy" : train_accuracy, "test_accuracy" : test_accuracy}
    return (str_results, pure_results)

def aggregate_results(nls_pure_results : dict, ls_pure_results: dict) -> Tuple[Sequence[str], Sequence[float]]:
    train_speedup = 100 * (nls_pure_results["train_time"] - ls_pure_results["train_time"])/nls_pure_results["train_time"]
    train_eval_speedup = 100 * (nls_pure_results["train_eval_time"] - ls_pure_results["train_eval_time"])/nls_pure_results["train_eval_time"]
    test_eval_speedup = 100 * (nls_pure_results["test_eval_time"] - ls_pure_results["test_eval_time"])/nls_pure_results["test_eval_time"]
    train_acc_gain = ls_pure_results["train_accuracy"] - nls_pure_results["train_accuracy"]
    test_acc_gain  = ls_pure_results["test_accuracy"]  - nls_pure_results["test_accuracy"]

    train_speedup_res = "Logarithmic Splitting trains forest " + str(train_speedup) + "% faster"
    train_eval_speedup_res = "Logarithmic Splitting evals on train set " + str(train_eval_speedup) + "% faster"
    test_eval_speedup_res  = "Logarithmic Splitting evals on test set " + str(test_eval_speedup) + "% faster"
    train_acc_gain_res = "Logarithmic Splitting gains " + str(train_acc_gain) + "% on training accuracy"
    test_acc_gain_res = "Logarithmic Splitting gains " + str(test_acc_gain) + "% on training accuracy"

    str_results = [train_speedup_res, train_eval_speedup_res, test_eval_speedup_res, train_acc_gain_res, test_acc_gain_res]
    pure_results = {"train_speedup" : train_speedup, 
                    "train_eval_speedup" : train_eval_speedup, "test_eval_speedup" : test_eval_speedup, 
                    "train_accuracy_gain" : train_acc_gain, "test_accuracy_gain" : test_acc_gain}

    return str_results, pure_results

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="the general name of dataset, example: loan for loan_X_train.txt, loan_Y_train.txt, etc.",
    )

    parser.add_argument(
        "--tree_count",
        type=int,
        required=True,
        default=100,
        help="number of trees in random forest",
    )

    parser.add_argument(
        "--max_layers_per_tree",
        type=int,
        required=True,
        default=10,
        help="max number of layers per decision tree",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required = False,
        default=42,
        help="seed",
    )

    args = parser.parse_args()

    all_data = get_datasets(args.dataset_name)
    
    print("STARTING NO LOG SPLITTING EXPERIMENT")
    nls_str_results, nls_pure_results = exp_and_results(all_data, args.tree_count, args.max_layers_per_tree, args.seed, False)
    print("STARTING LOG SPLITTING EXPERIMENT")
    ls_str_results,  ls_pure_results  = exp_and_results(all_data, args.tree_count, args.max_layers_per_tree, args.seed, True)
    agg_str_results, agg_pure_results = aggregate_results(nls_pure_results, ls_pure_results)

    nls_str_results = ["NO LOGARITHMIC SPLITTING RESULTS:"] + nls_str_results
    ls_str_results  = ["WITH LOGARITHMIC SPLITTING RESULTS:"] + ls_str_results
    agg_str_results = ["AGGREGATE RESULTS WITH AND WITHOUT LOGARITHMIC SPLITTING:"] + agg_str_results
    full_results    = nls_str_results + ["\n"] + ls_str_results + ["\n"] + agg_str_results

    full_print = "\n".join(full_results)
    print(full_print)

    results_filepath = get_log_file_filepath(args.dataset_name, args.tree_count, args.max_layers_per_tree, args.seed)
    text_file = open(results_filepath, "w")
    _ = text_file.write(full_print)
    text_file.close()

if __name__ == "__main__":
    main()
