# Parameters
BATCH_SIZE = 1000
CONF_MULTIPLIER = 1
TOLERANCE = -1e-1
BUFFER = 1000000
DEFAULT_MIN_IMPURITY_DECREASE = 5e-3
DEFAULT_ALPHA_N = 0.6
DEFAULT_ALPHA_F = 0.8
SAMPLE_SIZE = 60000
VECTORIZE = False

# Datasets
IRIS = "IRIS"
DIGITS = "DIGITS"
HEART = "HEART"
AIR = "AIR"
FLIGHT = "FLIGHT"
APS = "APS"
BLOG = "BLOG"
MNIST_STR = "MNIST"
SKLEARN = "SKLEARN"

# Algorithms
FASTFOREST = "FASTFOREST"
SKLEARN = "SKLEARN"

# Solvers
MAB = "MAB"
EXACT = "EXACT"

# Criteria
GINI = "GINI"
ENTROPY = "ENTROPY"
MSE = "MSE"

# Splitters
BEST = "BEST"
DEPTH = "DEPTH"

# Bin types
LINEAR = "LINEAR"
DISCRETE = "DISCRETE"
IDENTITY = "IDENTITY"
RANDOM = "RANDOM"
DEFAULT_NUM_BINS = 11

# Feature subsampling
SQRT = "SQRT"

# For Boosting
DEFAULT_CLASSIFIER_LOSS = "CELoss"
DEFAULT_REGRESSOR_LOSS = "MSELoss"
DEFAULT_GRAD_SMOOTHING_VAL = 1e-5

# For randomness
MAX_SEED = 2 ** 31

# For MSE estimation
KURTOSIS = 3  # normal distribution has kurtosis = 3

# Models for speedup comparison
CLASSIFICATION_MODELS = ["ERFC", "HRFC", "HRPC"]
REGRESSION_MODELS = ["ERFR", "GBERFR", "HRFR", "GBHRFR", "HRPR", "GBHRPR"]

# Feature Importance
JACCARD = "JACCARD"
SPEARMAN = "SPEARMAN"
KUNCHEVA = "KUNCHEVA"
MIN_IMPORTANCE = 0.0

# Budget
FOREST_UNIT_BUDGET_DIGIT = 1000000
FOREST_UNIT_BUDGET_DIABETES = 200000
FOREST_UNIT_BUDGET_REGRESSION = 1000000
FOREST_UNIT_BUDGET = 3000000

# Args for Table reproduce
t5_l1_args = {
    "seed": 123132,
    "data_size": 10000,
    "num_features": 60,
    "num_informative": 5,
    "num_trials": 30,
    "num_forests": 5,
    "max_depth": 6,
    "max_leaf_nodes": 40,
    "num_trees_per_feature": 10,
    "feature_subsampling": "SQRT",
    "best_k_feature": 6,
    "epsilon": 0.03,
    "budget": 350000,
    "importance_score": "impurity",
    "is_classification": True,
    "conf_multiplier": 1.96,
    "data_name": None,
}
t5_l2_args = {
    "seed": 133,
    "data_size": 10000,
    "num_features": 100,
    "num_informative": 5,
    "num_trials": 30,
    "num_forests": 5,
    "max_depth": 6,
    "max_leaf_nodes": 40,
    "num_trees_per_feature": 10,
    "feature_subsampling": "SQRT",
    "best_k_feature": 6,
    "epsilon": 0.03,
    "budget": 500000,
    "importance_score": "impurity",
    "is_classification": False,
    "conf_multiplier": 1.96,
    "data_name": None,
}
t5_l3_args = {
    "seed": 0,
    "data_size": None,
    "num_features": None,
    "num_informative": None,
    "num_trials": 30,
    "num_forests": 5,
    "max_depth": 3,
    "max_leaf_nodes": 24,
    "num_trees_per_feature": 20,
    "feature_subsampling": None,
    "best_k_feature": 10,
    "epsilon": 0.00,
    "budget": FOREST_UNIT_BUDGET_DIGIT,
    "importance_score": "permutation",
    "is_classification": True,
    "conf_multiplier": 1.96,
    "data_name": "digits",
}
t5_l4_args = {
    "seed": 21234,
    "data_size": 10000,
    "num_features": 100,
    "num_informative": 5,
    "num_trials": 30,
    "num_forests": 5,
    "max_depth": 6,
    "max_leaf_nodes": 40,
    "num_trees_per_feature": 10,
    "feature_subsampling": "SQRT",
    "best_k_feature": 6,
    "epsilon": 0.03,
    "budget": 500000,
    "importance_score": "impurity",
    "is_classification": False,
    "conf_multiplier": 1.96,
    "data_name": None,
}
