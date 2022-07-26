#!/bin/bash
cd experiments/runtime_exps
python compare_runtimes.py
cd ../budget_exps
python compare_budgets.py
cd ../scaling_exps
python investigate_scaling.py
python make_scaling_plot.py
cd ../../tests
python feature_importance_tests.py
cd ../experiments/sklearn_exps
python compare_baseline_implementations.py
cd ..
python produce_tables.py