# Dear Reviewer

We thank you for your time in reviewing our submission. 
We understand that you are performing a service to the community.
In order to ensure that review of this code is as easy as possible, we have:
- Included a one-line script to recreate all results (`recreate_all_results.sh`)
- Heavily commented and documented the code
- Included an overview of the codebase and each file below.

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
    - Include description of each profile
  - the `tests` subdirectory tests that we wrote to verify the correctness of our implementations
  - the `utils` directory contains helper code for training forest-based models