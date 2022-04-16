import cProfile as profile
import pstats
import numpy as np
from test_ff import test_tree_iris

if __name__ == "__main__":
    prof = profile.Profile()
    prof.enable()
    test_tree_iris() # Put the test we want to profile here
    prof.disable()
    stats = pstats.Stats(prof).strip_dirs().sort_stats("tottime")
    stats.print_stats(20)