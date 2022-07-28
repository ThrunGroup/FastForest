import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import os

from data_structures.wrappers.histogram_random_forest_classifier import HistogramRandomForestClassifier as HRFC
from utils.constants import MAB, EXACT

d_train = pd.read_csv("train-1m.csv")
d_test = pd.read_csv("test.csv")
d_train_test = d_train.append(d_test)

vars_categ = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime", "Distance"]


def get_dummies(d, col):
    dd = pd.get_dummies(d[col])
    dd.columns = [col + "_%s" % c for c in dd.columns]
    return (dd)


start = time.time()
X_train_test_categ = pd.concat([get_dummies(d_train_test, col) for col in vars_categ], axis=1)
print(time.time() - start)
X_train_test = pd.concat([X_train_test_categ, d_train_test[vars_num]], axis=1).to_numpy()
y_train_test = np.where(d_train_test["dep_delayed_15min"] == "Y", 1, 0)

X_train = X_train_test[0:d_train.shape[0]]
y_train = y_train_test[0:d_train.shape[0]]
X_test = X_train_test[d_train.shape[0]:]
y_test = y_train_test[d_train.shape[0]:]
print(X_train.shape)
model = HRFC(
    max_leaf_nodes=50,
    max_depth=100,
    solver=MAB,
    n_estimators=1,
    verbose=True,
)
model2 = RandomForestClassifier(
    max_leaf_nodes=50,
    n_estimators=1,
    max_features=None,
)
model3 = HRFC(
    max_leaf_nodes=50,
    max_depth=100,
    solver=EXACT,
    n_estimators=1,
    verbose=True,
)
print("Start fitting")
start = time.time()
model.fit(data=X_train, labels=y_train)
print(time.time() - start)
print("acc: ", np.mean(model.predict_batch(X_test)[0] == y_test))


print("Start fitting")
start = time.time()
model2.fit(X_train, y_train)
print(time.time() - start)
print("acc: ", np.mean(model2.predict(X_test) == y_test))

print("Start fitting")
start = time.time()
model3.fit(data=X_train, labels=y_train)
print(time.time() - start)
print("acc: ", np.mean(model3.predict_batch(X_test)[0] == y_test))
