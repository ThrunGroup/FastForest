import os
import numpy as np
import pandas as pd
import random
SEED = 42
DATA_FOLDER = "data/"
TRAIN_PROP  = 0.8
np.random.seed(SEED)

def main():
    df = pd.read_csv(DATA_FOLDER + "splice.data", header = None)
    np_data = df.to_numpy()

    res_data = []
    allowed = {'N' : 0, 'EI' : 1, 'IE' : 2, 'A' : 0, 'G' : 1, 'T' : 2, 'C' : 3, 'D' : 4, 'N' : 5, 'S' : 6, 'R' : 7}
    for r, row in enumerate(np_data):
        add_row = []
        clas = np_data[r][0]
        dat  = np_data[r][2].strip()
        for char in dat:
            assert char in allowed
            add_row.append(allowed[char])
        assert clas in allowed
        add_row.append(allowed[clas])
        res_data.append(add_row)


    fin_data = np.array(res_data)
    assert len(fin_data.shape) == 2

    X, Y = fin_data[:,:-1], fin_data[:,-1]
    assert len(X.shape) == 2 and len(Y.shape) == 1
    assert X.shape[0] == Y.shape[0]

    idxs = np.arange(len(X))
    np.random.shuffle(idxs)

    X, Y = X[idxs], Y[idxs]

    train_amt = int(TRAIN_PROP * len(X))

    X_train_data, Y_train_data = X[:train_amt], Y[:train_amt]
    X_test_data,  Y_test_data  = X[train_amt:], Y[train_amt:]

    counts = {}
    for y in Y_train_data:
        counts[y] = counts.get(y,0) + 1
    max_count = max(counts.values())

    for y,c in counts.items():
        idxs = list((Y_train_data == y).nonzero()[0])
        if max_count - c == 0:
            continue
        sampled = random.choices(idxs, k = max_count - c)
        X_train_data = np.concatenate((X_train_data, np.reshape(X_train_data[sampled], (len(sampled),-1))), axis = 0)
        Y_train_data = np.append(Y_train_data, Y_train_data[sampled])
    #X_train_data = np.reshape(X_train_data, (-1, X.shape[1]))

    for y,c in counts.items():
        new = np.sum(Y_train_data == y)
        print("Class " + str(y) + " went from " + str(c) + " to " + str(new))


    np.savetxt(DATA_FOLDER + "splice_X_train.txt", X_train_data)
    np.savetxt(DATA_FOLDER + "splice_Y_train.txt", Y_train_data)
    np.savetxt(DATA_FOLDER + "splice_X_test.txt", X_test_data)
    np.savetxt(DATA_FOLDER + "splice_Y_test.txt", Y_test_data)

if __name__ == "__main__":
    main()
