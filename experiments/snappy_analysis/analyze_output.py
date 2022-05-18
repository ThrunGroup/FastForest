import numpy as np


def main():
    with open("../results_filtered.txt") as fin:
        results = np.zeros((11, 10))
        size_idx = 0
        rc_idx = 0
        print(results)
        for line in fin:
            if "Round count:" in line:
                results[size_idx, rc_idx] = int(line.strip().split()[-1])
                rc_idx += 1
            elif "Results for" in line:
                size_idx += 1
                rc_idx = 0

    if rc_idx != 9:
        print("Warning: last row has incomplete results")

    print(results)
    print(np.mean(results, axis=1))
    print(np.std(results, axis=1))


if __name__ == "__main__":
    main()
