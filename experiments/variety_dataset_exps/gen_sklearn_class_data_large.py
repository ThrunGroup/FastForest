import numpy as np
from sklearn import datasets
import random
random.seed(42)

SEED = 42
NAME = "sklearn_gen_class_large"

def gen_data_and_save() -> None:
    X, Y = datasets.make_classification(n_samples = 10000, n_features = 15, n_classes = 4, n_informative = 10, random_state = SEED)
    X = np.array(X)
    Y = np.array(Y)

    N = len(X)
    idxs = np.arange(N)
    np.random.shuffle(idxs)

    X = X[idxs]
    Y = Y[idxs]

    train_prop = 0.8
    train_amt = int(round(N * train_prop))

    X_train, X_test = X[:train_amt], X[train_amt:]
    Y_train, Y_test = Y[:train_amt], Y[train_amt:]

    def balance_train(X_train, Y_train):
        f = len(X_train[0])
        d = {}
        for y in Y_train:
            d[y] = d.get(y,0) + 1
        ma = max(d.values())
        X_fin = np.copy(X_train)
        Y_fin = np.copy(Y_train)
        for c, v in d.items():
            need = ma - v
            idxs = [i for i,x in enumerate(Y_train) if x == c]
            add_idxs = random.choices(idxs, k = need)
            X_other = np.reshape(X_train[add_idxs], (-1,f))
            Y_other = Y_train[add_idxs]
            X_fin   = np.concatenate((X_fin, X_other), axis = 0)
            Y_fin   = np.concatenate((Y_fin, Y_other), axis = 0)
        return X_fin, Y_fin

    X_train, Y_train = balance_train(X_train, Y_train)

    np.savetxt("data/" + NAME + "_X_train.txt", X_train)
    np.savetxt("data/" + NAME + "_Y_train.txt", Y_train)
    np.savetxt("data/" + NAME + "_X_test.txt", X_test)
    np.savetxt("data/" + NAME + "_Y_test.txt", Y_test)

if __name__ == "__main__":
    gen_data_and_save()