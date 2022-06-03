cd experiments/budget_exps
python3 -m compare_budgets.py
cd ../runtime_exps
python3 -m compare_runtimes.py
cd ../scaling_exps
python3 -m investigate_scaling.py
python3 -m make_scaling_plots.py
cd ..
cd ..
python3 -m tests/feature_importance_tests.py