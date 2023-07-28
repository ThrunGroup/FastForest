import numpy as np
import typing
from typing import Sequence, Tuple

# Generate datasets X,Y,train,test given the name of dataset
def get_datasets(name : str) -> Tuple[Sequence[Sequence[float]], Sequence[float], Sequence[Sequence[float]], Sequence[float]]:
    prefix = "data/"
    suffixes = ["_X_train.txt", "_Y_train.txt", "_X_test.txt", "_Y_test.txt"]
    filepaths = [prefix + name + suffix for suffix in suffixes]
    X_train = np.loadtxt(filepaths[0])
    Y_train = np.loadtxt(filepaths[1])
    X_test  = np.loadtxt(filepaths[2])
    Y_test  = np.loadtxt(filepaths[3])
    return (X_train, Y_train, X_test, Y_test)

# Get characteristics of dataset (classes, features) from the dataset
def get_dataset_characteristics(
    X_train : Sequence[Sequence[float]], 
    Y_train : Sequence[int], 
    X_test : Sequence[Sequence[float]], 
    Y_test : Sequence[int]
    ) -> Tuple[int, list]:
    classes = len(set([int(i) for i in Y_train] + [int(i) for i in Y_test]))
    feature_count = len(X_train[0])
    features = [i for i in range(feature_count)]
    return classes, features

# Compute accuracy on given data points via X, Y using predictions under cutoff
def compute_RF_accuracy(RF, X, Y) -> float:
    correct = 0
    for idx, datapoint in enumerate(X):
        prediction, vals = RF.predict(datapoint)
        correct += (Y[idx] == prediction)
    return correct/len(Y)

# Get the filepath to save the current model's results to
def get_log_file_filepath(name: str, n_estimators: int, max_depth: int, seed: int) -> str:
    name_path = "log_results/" + str(name) + "_data_exp"
    n_estimators_path = "n_estimators_" + str(n_estimators)
    max_depth_path = "max_depth_" + str(max_depth)
    seed_path = "seed_" + str(seed)
    filepath_components = [name_path, n_estimators_path, max_depth_path, seed_path]
    return "_".join(filepath_components) + ".txt"