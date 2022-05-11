# Parameters
BATCH_SIZE = 1000
CONF_MULTIPLIER = 1
TOLERANCE = -1e-1
BUFFER = 1000

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
DEFAULT_LEARNING_RATE = 1e-1

# For randomness
MAX_SEED = 2 ** 32

