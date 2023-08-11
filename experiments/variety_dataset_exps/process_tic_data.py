import numpy as np
import pandas as pd
import random
SEED = 42
DATA_FOLDER = "data/"
TRAIN_PROP  = 0.8
np.random.seed(SEED)
random.seed(SEED)

def main():
    df = pd.read_csv(DATA_FOLDER + "tic-tac-toe.data", header = None)
    np_data = df.to_numpy()

    res_data = []
    allowed = {'x' : 0, 'o' : 1, 'b' : 2, 'positive' : 1, 'negative' : 0}
    for r, row in enumerate(np_data):
        if 0 in row:
            continue
        add_row = []
        for c, col in enumerate(row):
            char = np_data[r][c]
            assert char in allowed
            add_row.append(allowed[char])
        res_data.append(add_row)
        
    fin_data = np.array(res_data)
    assert len(fin_data.shape) == 2 and fin_data.shape[0] == 958 and fin_data.shape[1] == 10

    X, Y = fin_data[:,:-1], fin_data[:,-1]
    assert len(X.shape) == 2 and len(Y.shape) == 1
    assert X.shape[0] == Y.shape[0] and X.shape[1] == 9

    idxs = np.arange(len(X))
    np.random.shuffle(idxs)

    X, Y = X[idxs], Y[idxs]

    train_amt = int(TRAIN_PROP * len(X))

    X_train_data, Y_train_data = X[:train_amt], Y[:train_amt]
    X_test_data,  Y_test_data  = X[train_amt:], Y[train_amt:]

    train_pos_count = np.sum(Y_train_data == 1)
    train_neg_count = len(Y_train_data) - train_pos_count
    amt_add         = abs(train_pos_count - train_neg_count)
    class_equal_to  = int(train_pos_count < train_neg_count)
    idxs = list((Y_train_data == class_equal_to).nonzero()[0])
        
    print("positive class count before balancing train set: " + str(np.sum(Y_train_data == 1)))
    print("negative class count before balancing train set: " + str(np.sum(Y_train_data == 0)))
        
    X_train_data = np.reshape(X_train_data, (-1))
    for _ in range(amt_add):
        idx = random.sample(idxs, 1)
        X_train_data = np.append(X_train_data, X[idx])
        Y_train_data = np.append(Y_train_data, Y[idx])
    X_train_data = np.reshape(X_train_data, (-1,X.shape[1]))

    print("positive class count after balancing train set: " + str(np.sum(Y_train_data == 1)))
    print("negative class count after balancing train: set " + str(np.sum(Y_train_data == 0)))

    assert(X_train_data.shape[1] == X.shape[1] and X_train_data.shape[0] == Y_train_data.shape[0])
    assert(len(X_train_data.shape) == 2 and len(Y_train_data.shape) == 1)

    np.savetxt("data/loan_X_train.txt", X_train_data)
    np.savetxt("data/loan_Y_train.txt", Y_train_data)
    np.savetxt("data/loan_X_test.txt",  X_test_data)
    np.savetxt("data/loan_Y_test.txt",  Y_test_data)

    np.savetxt(DATA_FOLDER + "tic_X_train.txt", X_train_data)
    np.savetxt(DATA_FOLDER + "tic_Y_train.txt", Y_train_data)
    np.savetxt(DATA_FOLDER + "tic_X_test.txt", X_test_data)
    np.savetxt(DATA_FOLDER + "tic_Y_test.txt", Y_test_data)

if __name__ == "__main__":
    main()
