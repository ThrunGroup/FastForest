import matplotlib.pyplot as plt
import numpy as np
import random
import time

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes, make_classification, make_regression

from data_structures.tree_classifier import TreeClassifier
from data_structures.tree_regressor import TreeRegressor
from data_structures.forest_classifier import ForestClassifier
from utils.constants import SQRT, EXACT, MAB, LINEAR, IDENTITY

np.random.seed(0)
data, labels = make_classification(
    100000, n_informative=5, n_features=200, random_state=0
)
exact_model = DecisionTreeClassifier(max_depth=2)
# exact_model = TreeClassifier(
#     data=data,
#     labels=labels,
#     max_depth=8,
#     bin_type=LINEAR,
#     solver=EXACT,
#     with_replacement=False,
#     classes={0: 0, 1: 1},
#     min_impurity_decrease=0,
# )
our_model = TreeClassifier(
    data=data,
    labels=labels,
    max_depth=2,
    bin_type=LINEAR,
    solver=MAB,
    with_replacement=False,
    classes={0: 0, 1: 1},
    min_impurity_decrease=0,
)
start = time.time()
exact_model.fit(data, labels)
print(f"exact model classification time: {time.time() - start}")
start = time.time()
our_model.fit()
print(f"our classification model time: {time.time() - start}")
acc_exact = np.sum((exact_model.predict(data) == labels)) / len(data)
acc_our = np.sum((our_model.predict_batch(data)[0] == labels)) / len(data)

print(f"accuracy of exact model: {acc_exact}/ accuracy of our models: {acc_our}")
print(f"num_queries: {our_model.num_queries}")
print(np.sum(exact_model.predict(data) == our_model.predict_batch(data)[0]) / len(data))

print("---" * 30)
print("Regression Task")
data, labels = make_regression(100000, random_state=0)
# Uncomment below to do compare with sklearn decision tree. Set CONF_MULTIPLIER < 0.2 gives a faster result.
# sklearn = DecisionTreeRegressor(max_depth=2)
exact_model = TreeRegressor(
    data=data,
    labels=labels,
    max_depth=2,
    bin_type=LINEAR,
    solver=EXACT,
    with_replacement=False,
    min_impurity_decrease=0,
)
our_model = TreeRegressor(
    data=data,
    labels=labels,
    max_depth=2,
    bin_type=LINEAR,
    solver=MAB,
    with_replacement=False,
    min_impurity_decrease=0,
)
start = time.time()
exact_model.fit()
print(f"exact model regression time: {time.time() - start}")
start = time.time()
our_model.fit()
print(f"our model time: {time.time() - start}")
mse_exact = np.sum(np.square(exact_model.predict_batch(data) - labels)) / len(data)
mse_our = np.sum(np.square(our_model.predict_batch(data) - labels)) / len(data)

print(f"mse of exact_model: {mse_exact} / mse of our model: {mse_our}")
print(f"num_queries: {our_model.num_queries}")
print(
    np.sum(exact_model.predict_batch(data) == our_model.predict_batch(data)) / len(data)
)
