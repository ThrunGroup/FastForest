class Forest:
    def __init__(self):
        # Same parameters as sklearn.ensembleRandomForestClassifier. We won't need all of them.
        # See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        self.n_estimators = 100
        self.criterion = "gini"
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0.0
        self.max_features = "auto"
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.bootstrap = True
        self.oob_score = False
        self.n_jobs = None
        self.random_state = None
        self.verbose = 0
        self.warm_start = False
        self.class_weight = None
        self.ccp_alpha = 0.0
        self.max_samples = None
