import pandas as pd
import numpy as np
import os


def swap_columns(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    :param df: dataframe whose columns we want to swap
    :param col1: name of column1 of df
    :param col2: name of column2 of df
    :return: dataframe we get after swapping col1 and col2 of df
    """
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


def preprocess_heart(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: df is a dataframe from heart_2020_cleaned.csv
    :return: returns a preprocessed data of df
    """
    last_column = df.columns[-1]
    df = swap_columns(
        df, "Race", last_column
    )  # Race is the only categorical feature that can not be ordered, so put it in the last column
    columns = df.columns

    # For categorical features that can be ordered, convert them to integers
    for i in range(len(df.columns) - 1):
        column = columns[i]
        if df[column][0] == float:  # Pass the numerical features
            continue
        column_series = df[column]
        cf = np.unique(column_series)  # cf is column feature
        column_to_int = dict(zip(cf, range(len(cf))))  # label encoding
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


def main() -> None:
    """
    Preprocess heart disease data and store it as csv file. Download heart_2020_cleansed.csv from
    https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
    """
    filepath = os.path.join("dataset", "heart_2020_cleaned.csv")
    assert os.path.exists(
        filepath
    ), "Download data from the link in the comment of line 55 and save it as heart_2020_cleaned.csv in dataset file"
    df = pd.read_csv(filepath)
    df = preprocess_heart(df)
    os.makedirs("dataset", exist_ok=True)
    filepath = os.path.join("dataset", "new_heart_2020_cleaned.csv")
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
