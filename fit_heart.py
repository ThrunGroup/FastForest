import pandas as pd
import numpy as np
import os
import wandb
import random
import preprocess_heart
import argparse

from csv import DictWriter
from sklearn.model_selection import train_test_split
from forest import Forest

filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
df = pd.read_csv(filepath)
X = df.iloc[:, :-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)
ground_truth_forest(X, Y, 10, max_depth=20)
f = Forest(data=X, labels=Y, n_estimators=10, max_depth=20)
f.fit()
acc = np.sum(f.predict_batch(X)[0] == Y)
print("MAB solution Forest Train Accuracy:", acc / len(X))
