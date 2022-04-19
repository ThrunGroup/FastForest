import pandas as pd
import numpy as np
import os

from IPython.display import display
from test_ff import ground_truth_forest
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text,
)

filepath = os.path.join("dataset", "heart_2020_cleaned.csv")
df = pd.read_csv(filepath)


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


def preprocess_heart(df: pd.DataFrame) -> pd.DataFrame:
    """
    df is a dataframe from heart_2020_cleaned.csv
    """
    last_column = df.columns[-1]
    df = swap_columns(
        df, "Race", last_column
    )  # Race is the only categorical feature, so put it in the last column
    columns = df.columns

    # For categorical features that could be ordered, convert them to integers
    for i in range(
            len(df.columns) - 1
    ):  # Iterate over the columns except a label and a categorical feature column
        column = columns[i]
        if df[column][0] == float:
            continue
        column_series = df[column]
        cf = np.unique(column_series)  # cf is column feature
        column_to_int = dict(zip(cf, range(len(cf))))
        df[columns[i]] = df[columns[i]].map(lambda x: column_to_int[x])

    # For "race" feature, encode it using OneHotEncoder
    one_hot_race = pd.get_dummies(df["Race"])
    df = df.drop("Race", axis=1)
    df = df.join(one_hot_race)

    # old_columns: label, feature1, feature2, ... --> new_columns: feature1, feature2, ...., label
    old_columns = df.columns
    new_columns = list(old_columns[1:]) + [old_columns[0]]
    df = df[new_columns]
    return df


df = preprocess_heart(df)

if not os.path.exists("dataset"):
    os.makedirs("dataset")
filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
df.to_csv(filepath)


