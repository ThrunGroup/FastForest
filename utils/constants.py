# Parameters
BATCH_SIZE = 100
CONF_MULTIPLIER = 1
TOLERANCE = -1e-1
BUFFER = 10000
DEFAULT_MIN_IMPURITY_DECREASE = 5e-3

# Datasets
IRIS = "IRIS"
DIGITS = "DIGITS"
HEART = "HEART"

# Algorithms
FASTFOREST = "FASTFOREST"
SKLEARN = "SKLEARN"

# Solvers
MAB = "MAB"
EXACT = "EXACT"

# Criteria
GINI = "GINI"
ENTROPY = "ENTROPY"
VARIANCE = "VARIANCE"
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

# Budget
FOREST_UNIT_BUDGET = 5000000