import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import pandas as pd

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text,
)
from sklearn.datasets import load_diabetes, make_classification, make_regression

from experiments.heart.fit_heart import append_dict_as_row
from data_structures.tree_classifier import TreeClassifier
from data_structures.tree_regressor import TreeRegressor
from data_structures.forest_classifier import ForestClassifier
from data_structures.forest_regressor import ForestRegressor
from utils.constants import SQRT, EXACT, MAB, LINEAR, IDENTITY, RANDOM

"""
args
0: model name
1: data_size
2: n_features
3: informative_ratio
4: seed
5: max_depth 
6. max_leaf_nodes
7: with_replacement
8: epsilon
9: use_dynamic_epsilon
10: use_logarithmic split point 
11. is_classifier
12. n_estimators
"""
params_list = [
    "model_name",
    "data_size",
    "n_features",
    "informative_ratio",
    "seed",
    "max_depth",
    "max_leaf_nodes",
    "with_replacement",
    "epsilon",
    "use_dynamic_epsilon",
    "use_logarithmic split point",
    "is_classifier",
    "n_estimators",
]


def compare_models(args):
    (
        model,
        data_size,
        n_features,
        informative_ratio,
        seed,
        max_depth,
        max_leaf_nodes,
        with_replacement,
        epsilon,
        use_dynamic_epsilon,
        use_logarithmic_split_point,
        is_classifier,
        n_estimators,
    ) = args
    n_informative = int(n_features * informative_ratio)
    np.random.seed(seed)
    if is_classifier:
        data, labels = make_classification(
            data_size,
            n_features=n_features,
            n_informative=n_informative,
            random_state=seed,
        )
    else:
        data, labels = make_regression(
            data_size,
            n_features=n_features,
            n_informative=n_informative,
            random_state=seed,
        )
    if model == "TreeClassifier":
        exact_model = TreeClassifier
        our_model = TreeClassifier
    elif model == "TreeRegressor":
        exact_model = TreeRegressor
        our_model = TreeRegressor
    elif model == "ForestClassifier":
        exact_model = ForestClassifier
        our_model = ForestClassifier
    elif model == "ForestRegressor":
        exact_model = ForestRegressor
        our_model = ForestRegressor
    if n_estimators is None:
        if is_classifier:
            exact_model = exact_model(
                data=data,
                labels=labels,
                max_depth=max_depth,
                bin_type=LINEAR,
                solver=EXACT,
                with_replacement=with_replacement,
                min_impurity_decrease=0,
                max_leaf_nodes=max_leaf_nodes,
                classes={0: 0, 1: 1},
            )
            our_model = our_model(
                data=data,
                labels=labels,
                max_depth=max_depth,
                bin_type=LINEAR,
                solver=MAB,
                with_replacement=with_replacement,
                min_impurity_decrease=0,
                max_leaf_nodes=max_leaf_nodes,
                classes={0: 0, 1: 1},
            )
        else:
            exact_model = exact_model(
                data=data,
                labels=labels,
                max_depth=max_depth,
                bin_type=LINEAR,
                solver=EXACT,
                with_replacement=with_replacement,
                min_impurity_decrease=0,
                max_leaf_nodes=max_leaf_nodes,
            )
            our_model = our_model(
                data=data,
                labels=labels,
                max_depth=max_depth,
                bin_type=LINEAR,
                solver=MAB,
                with_replacement=with_replacement,
                min_impurity_decrease=0,
                max_leaf_nodes=max_leaf_nodes,
            )
    else:
        exact_model.n_estimators = n_estimators
        our_model.n_estimators = n_estimators
    print("-" * 30)
    print("Fitting ", model, "\n")
    start = time.time()
    exact_model.fit(data, labels)
    exact_time = time.time() - start
    print(f"exact model classification time: {exact_time}")

    start = time.time()
    our_model.fit()
    our_time = time.time() - start
    print(f"our classification model time: {our_time}")
    if is_classifier:
        score_exact = np.sum((exact_model.predict_batch(data)[0] == labels)) / len(data)
        score_our = np.sum((our_model.predict_batch(data)[0] == labels)) / len(data)
    else:
        score_exact = np.sum(np.square(exact_model.predict_batch(data) - labels)) / len(
            data
        )
        score_our = np.sum(np.square(our_model.predict_batch(data) - labels)) / len(
            data
        )

    print(f"score of exact model: {score_our}/ score of our models: {score_exact}")
    print(
        f"num_queries of exact model: {exact_model.num_queries}/ num_queries of our model: {our_model.num_queries}"
    )
    print(
        f"depth of exact model: {exact_model.depth}/ depth of our_model: {our_model.depth}"
    )

    # Log the results
    log_dict = {
        "time_diff": str(-(exact_time - our_time) / exact_time * 100) + "%",
        "score_diff": str(-(score_exact - score_our) / score_exact * 100) + "%",
        "num_queries_diff": str(
            -(exact_model.num_queries - our_model.num_queries) / exact_model.num_queries * 100
        )
        + "%",
    }
    experiment_info = dict(zip(params_list, args))
    log_dict.update(experiment_info)
    dir_name = "exact_vs_ours"
    log_filename = os.path.join(dir_name, "exact_vs_ours.csv")
    if not os.path.exists(log_filename):
        os.makedirs(dir_name, exist_ok=True)
        df = pd.DataFrame(columns=log_dict.keys())
        df.to_csv(log_filename, index=False)
    append_dict_as_row(log_filename, log_dict, log_dict.keys())


"""
args
0: model name
1: data_size
2: n_features
3: informative_ratio
4: seed
5: max_depth 
6. max_leaf_nodes
7: with_replacement
8: epsilon
9: use_dynamic_epsilon
10: use_logarithmic split point 
11. is_classifier
12. n_estimators
"""

if __name__ == "__main__":
    args1 = [
        "TreeClassifier",
        20000,
        20,
        1 / 4,
        0,
        100,
        100,
        False,
        0.05,
        False,
        True,
        True,
        None,
    ]
    params_to_idx = dict(zip(params_list, range(13)))
    args1[params_to_idx["model_name"]] = "TreeRegressor"
    args1[params_to_idx["data_size"]] = 1000000
    args1[params_to_idx["n_features"]] = 50
    args1[params_to_idx["informative_ratio"]] = 0.06
    args1[params_to_idx["seed"]] = 1
    args1[params_to_idx["max_depth"]] = 2
    args1[params_to_idx["max_leaf_nodes"]] = 100
    args1[params_to_idx["with_replacement"]] = False
    args1[params_to_idx["epsilon"]] = 0.01
    args1[params_to_idx["use_dynamic_epsilon"]] = False
    args1[params_to_idx["use_logarithmic split point"]] = True
    args1[params_to_idx["is_classifier"]] = False
    args1[params_to_idx["n_estimators"]] = None  # set to be None if don't use forest.
    compare_models(args1)

# np.random.seed(0)
# data, labels = make_classification(
#     200000, n_informative=5, n_features=20, random_state=1
# )
# # Uncomment this to do compare with sklearn decision tree. Set CONF_MULTIPLIER < 0.2 gives a faster result.
# # exact_model = DecisionTreeClassifier(max_depth=2)
# exact_model = TreeClassifier(
#     data=data,
#     labels=labels,
#     max_depth=11,
#     bin_type=LINEAR,
#     solver=EXACT,
#     with_replacement=False,
#     classes={0: 0, 1: 1},
#     min_impurity_decrease=0,
#     max_leaf_nodes=222,
# )
# our_model = TreeClassifier(
#     data=data,
#     labels=labels,
#     max_depth=11,
#     bin_type=LINEAR,
#     solver=MAB,
#     with_replacement=False,
#     classes={0: 0, 1: 1},
#     min_impurity_decrease=-0,
#     max_leaf_nodes=222,
# )
# start = time.time()
# exact_model.fit(data, labels)
# print(f"exact model classification time: {time.time() - start}")
# start = time.time()
# our_model.fit()
# print(f"our classification model time: {time.time() - start}")
# acc_exact = np.sum((exact_model.predict_batch(data)[0] == labels)) / len(data)
# acc_our = np.sum((our_model.predict_batch(data)[0] == labels)) / len(data)
#
# print(f"accuracy of exact model: {acc_exact}/ accuracy of our models: {acc_our}")
# print(
#     f"num_queries of exact model: {exact_model.num_queries}/ num_queries of our model: {our_model.num_queries}"
# )
# print(exact_model.depth, our_model.depth)
#
#
# print("---" * 30)
# print("Regression Task")
# data, labels = make_regression(3000000, n_features=35, n_informative=4, random_state=0)
# sklearn = DecisionTreeRegressor(max_leaf_nodes=600)
# sklearn.fit(data, labels)
# print("sklearn acc: ", np.sum(np.square(sklearn.predict(data) - labels)) / len(data))
# exact_model = TreeRegressor(
#     data=data,
#     labels=labels,
#     max_depth=3,
#     bin_type=LINEAR,
#     solver=EXACT,
#     with_replacement=False,
#     min_impurity_decrease=0,
#     max_leaf_nodes=100,
# )
# our_model = TreeRegressor(
#     data=data,
#     labels=labels,
#     max_depth=3,
#     bin_type=LINEAR,
#     solver=MAB,
#     with_replacement=False,
#     min_impurity_decrease=0,
#     max_leaf_nodes=100,
# )
# erf_model = TreeRegressor(
#     data=data,
#     labels=labels,
#     max_depth=100,
#     bin_type=RANDOM,
#     solver=MAB,
#     with_replacement=False,
#     min_impurity_decrease=-1,
#     max_leaf_nodes=300,
# )
# start = time.time()
# exact_model.fit()
# print(f"exact model regression time: {time.time() - start}")
# start = time.time()
# our_model.fit()
# print(f"our model time: {time.time() - start}")
# mse_exact = np.sum(np.square(exact_model.predict_batch(data) - labels)) / len(data)
# mse_our = np.sum(np.square(our_model.predict_batch(data) - labels)) / len(data)
#
# print(f"mse of exact_model: {mse_exact} / mse of our model: {mse_our}")
# print(
#     f"num_queries of exact model: {exact_model.num_queries}/ num_queries of our model: {our_model.num_queries}"
# )
# print(exact_model.depth, our_model.depth)

# for i in range(1, 10):
#     sklearn = DecisionTreeRegressor(max_depth=i)
#     sklearn.fit(data, labels)
#     print(np.sum(np.square(sklearn.predict(data)-labels))/ len(labels))


# np.random.seed(0)
# data, labels = make_classification(10000, n_informative=5, n_features=20, random_state=0)
# # Uncomment this to do compare with sklearn decision tree. Set CONF_MULTIPLIER < 0.2 gives a faster result.
# # exact_model = DecisionTreeClassifier(max_depth=2)
# exact_model = ForestClassifier(
#     n_estimators=10,
#     data=data,
#     labels=labels,
#     max_depth=5,
#     bin_type=LINEAR,
#     solver=MAB,
#     with_replacement=False,
#     classes={0: 0, 1: 1},
#     min_impurity_decrease=0,
# )
# our_model = ForestClassifier(
#     n_estimators=10,
#     data=data,
#     labels=labels,
#     max_depth=5,
#     bin_type=LINEAR,
#     solver=MAB,
#     with_replacement=False,
#     classes={0: 0, 1: 1},
#     min_impurity_decrease=0,
#     is_precomputed_minmax=True,
# )
# start = time.time()
# exact_model.fit(data, labels)
# print(f"exact model classification time: {time.time() - start}")
# start = time.time()
# our_model.fit()
# print(f"our classification model time: {time.time() - start}")
# acc_exact = np.sum((exact_model.predict_batch(data)[0] == labels)) / len(data)
# acc_our = np.sum((our_model.predict_batch(data)[0] == labels)) / len(data)
#
# print(f"accuracy of exact model: {acc_exact}/ accuracy of our models: {acc_our}")
# print(f"num_queries of exact model: {exact_model.num_queries}/ num_queries of our model: {our_model.num_queries}")
# print(np.sum(exact_model.predict_batch(data)[0] == our_model.predict_batch(data)[0]) / len(data))
#
# print("---" * 30)
# print("Regression Task")
# data, labels = make_regression(10000, random_state=0)
# exact_model = ForestRegressor(
#     n_estimators=10,
#     data=data,
#     labels=labels,
#     max_depth=4,
#     bin_type=LINEAR,
#     solver=MAB,
#     with_replacement=False,
#     min_impurity_decrease=0
# )
# our_model = ForestRegressor(
#     n_estimators=10,
#     data=data,
#     labels=labels,
#     max_depth=4,
#     bin_type=LINEAR,
#     solver=MAB,
#     with_replacement=False,
#     min_impurity_decrease=0,
#     is_precomputed_minmax=True,
# )
# start = time.time()
# exact_model.fit()
# print(f"exact model regression time: {time.time() - start}")
# start = time.time()
# our_model.fit()
# print(f"our model time: {time.time() - start}")
# mse_exact = np.sum(np.square(exact_model.predict_batch(data) - labels)) / len(data)
# mse_our = np.sum(np.square(our_model.predict_batch(data) - labels)) / len(data)
#
# print(f"mse of exact_model: {mse_exact} / mse of our model: {mse_our}")
# print(f"num_queries of exact model: {exact_model.num_queries}/ num_queries of our model: {our_model.num_queries}")
# print(np.sum(exact_model.predict_batch(data) == our_model.predict_batch(data)) / len(data))
