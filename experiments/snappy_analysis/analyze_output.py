import numpy as np


def main():
    with open("../results_filtered.csv") as fin:
        sizes = []
        results = np.zeros((12, 20))
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
                sizes.append(int(line.strip().split()[-2]))

    if rc_idx != 9:
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
