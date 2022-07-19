import numpy as np
from typing import Callable, Union, List, Tuple, Iterable
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from data_structures.forest_base import ForestBase
from data_structures.forest_regressor import ForestRegressor


def generate_data(
    bayes_model: Callable,
    num_data: int,
    noise: float,
    rng: np.random.Generator = np.random.default_rng(0),
):
    X = rng.normal(size=(num_data, 50))
    Y = bayes_model(X) + rng.normal(scale=np.sqrt(noise), size=num_data)
    return X, Y


def toy_bayes_model(
    x: np.ndarray, coef: List[float] = [5.0, 5.0, -1.0, 3.0, 10.0],
) -> np.ndarray:
    """
    A random model that maps 2d-vector X to 1d-vector Y.
    """
    return (
        coef[0] * np.sin(x[:, 0]) ** 2
        + coef[1] * np.square(x[:, 1])
        + coef[2] * np.sqrt(np.exp(x[:, 2]))
        + coef[3]
    ) / coef[4]


def bias_variance_analysis(
    model: Union[ForestBase, BaseEstimator],
    is_sklearn: bool = False,
    num_data: int = 100000,
    random_state: int = 0,
) -> Tuple:
    """
    Decompose the generalization error of supervised model (especially, random forest) into noise, bias,
    and variance by running a few times of experiments.
    :param model: A supervised model
    :param is_sklearn: Whether the model is implemented in sklearn package
    :param num_data: Number of toy data used for acquiring an error by the model
    :param random_state: Random state(seed) of this function for the purpose or reproducibility
    :return: A tuple of noise, bias squared, variance and generalization error of the model
    """
    rng = np.random.default_rng(random_state)
    rng2 = np.random.default_rng(random_state + 1)
    X, Y = generate_data(
        bayes_model=toy_bayes_model, num_data=num_data, noise=0.5, rng=rng
    )
    X_test, Y_test = generate_data(
        bayes_model=toy_bayes_model, num_data=num_data, noise=0.5, rng=rng2
    )
    train_size = 1000
    num_trials = 10
    model_predict_array = np.empty((num_trials, num_data))
    for i in range(num_trials):
        print(f"--{i}th training--")
        sample_indices = rng.choice(num_data, train_size)
        X_train = X[sample_indices]
        Y_train = Y[sample_indices]
        model.fit(X_train, Y_train)
        if is_sklearn:
            model_predict_array[i, :] = model.predict(X_test)
        else:
            model_predict_array[i, :] = model.predict_batch(X_test)
            print("num_queries:", model.num_queries)
            model.reset()
    bayes_predict = toy_bayes_model(X_test)
    noise = (bayes_predict - Y_test) ** 2
    mean_model_predict_array = model_predict_array.mean(axis=0)
    bias_square = (bayes_predict - mean_model_predict_array) ** 2
    variance = ((mean_model_predict_array - model_predict_array) ** 2).mean(axis=0)
    error = ((model_predict_array - Y_test) ** 2).mean(axis=0)
    return noise, bias_square, variance, error


if __name__ == "__main__":
    model1 = RandomForestRegressor(
        n_estimators=10, max_leaf_nodes=32, max_features="sqrt"
    )
    model2 = ForestRegressor(
        n_estimators=20,
        max_leaf_nodes=32,
        feature_subsampling="SQRT",
        batch_size=30,
        epsilon=0,
    )
    model3 = ForestRegressor(
        n_estimators=10, max_leaf_nodes=32, feature_subsampling="SQRT", solver="EXACT"
    )
    models = (
        ("sklearn regressor", model1),
        ("ff regressor", model2),
        ("ff regressor exact", model3),
    )
    names = []
    noises = []
    biases = []
    variances = []
    errors = []
    for i, (name, model) in enumerate(models):
        print("-" * 30)
        print(f"Training {name}")
        is_sklearn = "sklearn" in name
        noise, bias, variance, error= bias_variance_analysis(
            model, is_sklearn=is_sklearn, random_state=1512
        )
        names.append(name)
        noises.append(np.mean(noise))
        biases.append(np.mean(bias))
        variances.append(np.mean(variance))
        errors.append(np.mean(error))
    marker = "*"
    plt.figure()
    plt.plot(names, noises, marker=marker, label="Noise")
    plt.plot(names, biases, marker=marker, label="Bias^2")
    plt.plot(names, variances, marker=marker, label="Var")
    plt.plot(names, errors, marker=marker, label="Error")

    def annotate_yvals(xvals: Iterable, yvals: Iterable):
        """
        A helper function for annotating y values in the figure (matplotlib)
        """
        for x, y in zip(xvals, yvals):
            rounded_y = str(round(y, 3))
            plt.annotate(rounded_y, (x, y), xytext=(0, 5), textcoords="offset points")

    for yvals in [noises, biases, variances, errors]:
        annotate_yvals(names, yvals)
    plt.legend()
    plt.show()
