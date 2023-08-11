import pandas as pd
import numpy as np
import random
random.seed(42)

def main():
    # Obtain and pre-process data into numerical format from loan acceptance
    df = pd.read_csv("data/unprocessed_loan_dataset.csv")
    df = df.dropna()
    df = df.replace("Male", int(1))
    df = df.replace("Female", int(0))
    df = df.replace("Graduate", int(1))
    df = df.replace("Not Graduate", int(0))
    df = df.replace("Yes", int(1))
    df = df.replace("No", int(0))
    df = df.replace("Y", int(1))
    df = df.replace("N", int(0))
    df = df.replace("Rural", int(1))
    df = df.replace("Urban", int(0))
    df = df.replace("Semiurban", int(2))
    df = df.replace("3+", int(3))
    columns = df.columns
    data = df[columns[1:]].to_numpy().astype(float)
    X_data, Y_data = np.reshape(data[:,:-1], (len(data), -1)), data[:,-1]
    assert X_data.shape[0] == len(data) and X_data.shape[1] == len(data[0]) - 1 and len(Y_data.shape) == 1 and Y_data.shape[0] == len(data)

    train_prop = 0.8
    train_amt = int(train_prop * len(X_data))
    test_amt  = len(X_data) - train_amt
    X_train_data, Y_train_data, X_test_data, Y_test_data = X_data[:train_amt], Y_data[:train_amt], X_data[train_amt:], Y_data[train_amt:]

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
        X_train_data = np.append(X_train_data, X_data[idx])
        Y_train_data = np.append(Y_train_data, Y_data[idx])
    X_train_data = np.reshape(X_train_data, (-1,X_data.shape[1]))

    print("positive class count after balancing train set: " + str(np.sum(Y_train_data == 1)))
    print("negative class count after balancing train: set " + str(np.sum(Y_train_data == 0)))

    assert(X_train_data.shape[1] == X_data.shape[1] and X_train_data.shape[0] == Y_train_data.shape[0])
    assert(len(X_train_data.shape) == 2 and len(Y_train_data.shape) == 1)

    np.savetxt("data/loan_X_train.txt", X_train_data)
    np.savetxt("data/loan_Y_train.txt", Y_train_data)
    np.savetxt("data/loan_X_test.txt",  X_test_data)
    np.savetxt("data/loan_Y_test.txt",  Y_test_data)

if __name__ == "__main__":
    main()
