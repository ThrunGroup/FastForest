import numpy as np


def main():
    with open("../with_repl_results_filtered.csv") as fin:
        sizes = []
        NUM_SIZES = 30
        NUM_TRIALS = 20
        results = np.zeros((NUM_SIZES, NUM_TRIALS))
        size_idx = 0
        rc_idx = 0
        for line in fin:
            if "Round count:" in line:
                results[size_idx, rc_idx] = int(line.strip().split()[-1])
                rc_idx += 1
            elif "Results for" in line:
                size_idx += 1
                rc_idx = 0
                sizes.append(int(line.strip().split()[-2]))

    if results[-1, -1] == 0:
        print("Warning: last row has incomplete results")

    print(results)
    print()
    print(list(sizes))
    print()
    print(list(np.mean(results, axis=1)))
    print()
    print(list(np.std(results, axis=1)))


if __name__ == "__main__":
    main()
