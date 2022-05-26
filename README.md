# Dear Reviewer:

We thank you for your time in reviewing our submission. 
We understand that you are performing a service to the community.
In order to ensure that review of this code is as easy as possible, we have:
- Included a one-line script to recreate all results (`repro_results.sh`)
- Heavily commented and documented the code
- Included an overview of the codebase and the files.

Thank you for reviewing our submission!

# Description of Files

The files are organized as follows:

- `data_structures` contains all of the data structures used in our experiments, e.g., 
forests, trees, nodes, and histograms.
  - the `wrappers` subdirectory contains convenience classes to instantiate models of various types,
  e.g., a `RandomForestClassifier` is a forest classifier with `bootstrap=True` 
  (indicating to draw a bootstrap sample of the `n` datapoints for each tree), `feature_subsampling=SQRT`
  (indicating to consider only `sqrt(F)` features of the original `F` features at each node split), etc.
- the `experiments` subdirectory contains all the code for our core experiments
  - `experiments/runtime_exps` contains the script (`compare_runtimes.py`) to reproduce the results of Tables 1 and 2, as well as the results of running that script (the files ending in `_profile` or `_dict`)
  - `experiments/budget_exps` contains the script (`compare_budgets.py`) to reproduce the results of Tables 3 and 4, as well as the results of running that script (the files ending in `_dict`)
  - `experiments/sklearn_exps` contains the script (`compare_baseline_implementations.py`) to reproduce the results of Table 6 in Appendix 4
  - `experiments/scaling_exps` contains the scripts (`investigate_scaling.py` and `make_scaling_plot.py`) to reproduce Appendix Figure 1 in Appendix 2 
- the `tests` subdirectory tests that we wrote to verify the correctness of our implementations
  - `tests/feature_importance_tests.py` is also used to regenerate the results in Table 5
    -  You can reproduce the results for just Table 5 by running `tests/feature_importance_tests.py`. The results will be stored in the first 4 lines of `tests/stat_test_stability_log/reproduce_stability.csv` file. 
- the `utils` directory contains helper code for training forest-based models
  - `utils/solvers.py` includes the core implementation of MABSplit in the `solve_mab()` function
  
# Reproduce the tables
- To reproduce the results in all the tables, and to reproduce the figure in Appendix 2, please run `repro_script.sh`. This may take many hours.
