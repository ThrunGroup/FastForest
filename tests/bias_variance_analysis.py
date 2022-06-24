import numpy as np
from typing import Generator
from sklearn.ensemble import RandomForestRegressor
import sklearn
import matplotlib.pyplot as plt

from data_structures.forest_base import ForestBase
from data_structures.forest_regressor import ForestRegressor


def generate_data(bayes_model: callable, num_data: int, noise: float, rng: Generator = np.random.default_rng(0)):
    X = rng.normal(size=(num_data, 50))
    Y = bayes_model(X) + rng.normal(scale=np.sqrt(noise), size=num_data)
    return X, Y


def bayes_model1(X: np.ndarray) -> np.ndarray:
    return (np.sin(X[:, 0]) ** 2 * 5 + np.square(X[:, 1]) * 5 - np.sqrt(np.exp(X[:, 2])) + 3)/10


def bias_variance_analysis(model, is_sklearn: bool=False, num_data=100000, random_state: int = 0):
    rng = np.random.default_rng(random_state)
    rng2 = np.random.default_rng(random_state+1)
    X, Y = generate_data(bayes_model=bayes_model1, num_data=num_data, noise=0.5, rng=rng)
    X_test, Y_test = generate_data(bayes_model=bayes_model1, num_data=num_data, noise=0.5, rng=rng2)
    train_size = 1000
    num_trials = 10
    model_predict_array = np.empty((num_trials, num_data))
    for i in range(num_trials):
        print(f'--{i}th training--')
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
    bayes_predict = bayes_model1(X_test)
    noise = (bayes_predict - Y_test) ** 2
    mean_model_predict_array = model_predict_array.mean(axis=0)
    bias_square = (bayes_predict - mean_model_predict_array) ** 2
    variance = ((mean_model_predict_array - model_predict_array) ** 2).mean(axis=0)
    error = ((model_predict_array - Y_test) ** 2).mean(axis=0)

    return noise, bias_square, variance, error, X_test, Y_test

if __name__ == "__main__":
    model1 = RandomForestRegressor(n_estimators=10, max_leaf_nodes=32, max_features="sqrt")
    model2 = ForestRegressor(n_estimators=20, max_leaf_nodes=32, feature_subsampling="SQRT", batch_size=30, epsilon=0)
    model3 = ForestRegressor(n_estimators=10, max_leaf_nodes=32, feature_subsampling="SQRT", solver="EXACT")
    models = (("sklearn regressor", model1), ("ff regressor", model2), ("ff regressor exact", model3))


    names = []
    noise_list = []
    bias_list = []
    variance_list = []
    error_list = []
    for i, (name, model) in enumerate(models):
        print("-" * 30)
        print(f"Training {name}")
        is_sklearn = "sklearn" in name
        noise, bias, variance, error, X_test, Y_test = (bias_variance_analysis(model, is_sklearn=is_sklearn, random_state=1512))
        names.append(name)
        noise_list.append(np.mean(noise))
        bias_list.append(np.mean(bias))
        variance_list.append(np.mean(variance))
        error_list.append(np.mean(error))
    plt.figure()
    plt.scatter(names, noise_list, label="noise")
    plt.scatter(names, bias_list, label="bias^2")
    plt.scatter(names, variance_list, label="var")
    plt.scatter(names, error_list, label="error")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(names, noise_list, label="noise")
    plt.plot(names, bias_list, label="bias^2")
    plt.plot(names, variance_list, label="var")
    plt.plot(names, error_list, label="error")
    plt.legend()
    plt.show()

    print(names, "\n", noise_list, "\n", bias_list, "\n", variance_list, "\n", error_list)

