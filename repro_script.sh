cd experiments/budget_exps
python compare_budgets.py
cd ../runtime_exps
python compare_runtimes.py
cd ../scaling_exps
python investigate_scaling.py
python make_scaling_plots.py
cd ../..
python tests/feature_importance_tests.py