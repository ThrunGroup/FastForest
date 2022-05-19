import sklearn.datasets
import numpy as np
import math
from scipy.stats import rankdata
from typing import Union, List

from data_structures.forest_classifier import ForestClassifier
from data_structures.forest_regressor import ForestRegressor
from data_structures.tree_classifier import TreeClassifier
from data_structures.tree_regressor import TreeRegressor
from utils.utils import class_to_idx
from utils.constants import MAB, EXACT, MIN_IMPORTANCE, JACCARD, SPEARMAN, KUNCHEVA, MAX_SEED
from utils.utils import set_seed


class PermutationImportance:
    """
    Class for determining feature importance based on the permutation model in scikit-learn.
    Use this to find an array of size K X F were K is the number of forets and F is the dimension of features.
    """
    def __init__(
        self,
        seed: int,
        data: np.ndarray,
        labels: np.ndarray,
        max_depth: int = 5,
        num_forests: int = 5,
        num_trees_per_forest: int = 10,
        is_classification: bool = True,
        solver: str = MAB,
        stability_metric: str = JACCARD,
        budget_per_forest: int = None,
        feature_subsampling: str = None,
        max_leaf_nodes: int = None,
        epsilon: float = 0,
    ):
        assert num_forests > 1, "we need at least two forests"
        set_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.data = data
        self.labels = labels
        self.max_depth = max_depth
        self.budget_per_forest = budget_per_forest
        self.num_forests = num_forests
        self.num_trees_per_forest = num_trees_per_forest
        self.is_classification = is_classification
        self.solver = solver
        self.stability_metric = stability_metric
        self.forests: List[Union[ForestRegressor, ForestClassifier]] = []
        self.is_train = False
        self.feature_subsampling = feature_subsampling
        self.max_leaf_nodes = max_leaf_nodes
        self.epsilon = epsilon

    def train_forest(self, seed: int) -> None:
        """
        Trains a forest and adds it to the list of forests
        """
        if self.is_classification:
            forest = ForestClassifier(
                random_state=seed,
                data=self.data,
                labels=self.labels,
                max_depth=self.max_depth,
                n_estimators=self.num_trees_per_forest,
                solver=self.solver,
                budget=self.budget_per_forest,
                feature_subsampling=self.feature_subsampling,
                max_leaf_nodes=self.max_leaf_nodes,
                epsilon=self.epsilon,
                bootstrap=True,
                oob_score=True,
            )
        else:
            forest = ForestRegressor(
                random_state=seed,
                data=self.data,
                labels=self.labels,
                max_depth=self.max_depth,
                n_estimators=self.num_trees_per_forest,
                solver=self.solver,
                budget=self.budget_per_forest,
                feature_subsampling=self.feature_subsampling,
                max_leaf_nodes=self.max_leaf_nodes,
                epsilon=self.epsilon,
                bootstrap=True,
                oob_score=True,
            )
        forest.fit()
        print("number of trees: ", len(forest.trees))
        print("number of queries: ", forest.num_queries)
        self.forests.append(forest)

    def train_forests(self) -> None:
        """
        Trains all the forests by calling self.train_forest num_forests times.
        """
        for i in range(self.num_forests):
            print("training forest: ", i+1)
            seed = self.rng.integers(0, MAX_SEED)
            self.train_forest(seed)

    def compute_importance_vec(self, forest_idx: int) -> np.ndarray:
        """
        Compute the importance vector for the forest corresponding to forest_idx.
        Sorts the indices of the feature by relative importance. Assumes that all forests have been trained.

        :return: sorted array of indices
        """
        assert self.is_train, "Forest isn't trained"
        importance_vec = []
        forest = self.forests[forest_idx]
        num_trees = len(forest.trees)   # number of trees depends on the budget
        data_copy = np.ndarray.copy(self.data)

        model_score = np.sum(forest.predict_batch(data_copy)[0] == self.labels)
        for feature_idx in range(len(data_copy[0])):
            self.rng.shuffle(data_copy[:, feature_idx])  # shuffles in-place
            model_score = forest.get_oob_score(self.data)
            permutated_model_score = forest.get_oob_score(data_copy)
            importance_vec.append(np.abs(model_score - permutated_model_score))
        return np.asarray(importance_vec)

    def get_importance_array(self) -> np.ndarray:
        """
        Trains all of its forests and computes the importance vector for each of the trained forests.
        This is the main function that an object of this class will call.

        :return: an array of importance vectors.
        """
        self.is_train = True
        self.train_forests()
        num_forests = len(self.forests)

        result = np.array([])
        for forest_idx in range(num_forests):
            imp_vec = self.compute_importance_vec(forest_idx)
            if len(result) == 0:
                result = imp_vec    # initialize with the first importance vector
            else:
                result = np.vstack((result, imp_vec))
        return result

    def get_pairwise_stability(self, v1_ranks: np.ndarray, v2_ranks: np.ndarray) -> float:
        """
        Computes pairwise stability for two importance vectors.
        NOTE: stability is computed with the "ranks" of the features.
        """
        pairwise_stability = 0.0
        length = len(v1_ranks)
        if self.stability_metric == JACCARD:
            for i in range(1, length):
                sub_v1, sub_v2 = v1_ranks[:i+1], v2_ranks[:i+1]     # compare subsections of array
                pairwise_stability += len(np.intersect1d(sub_v1, sub_v2)) / len(np.union1d(sub_v1, sub_v2))
            return pairwise_stability / length

        elif self.stability_metric == SPEARMAN:
            for i in range(1, length):
                pairwise_stability += math.pow((v1_ranks[i] - v2_ranks[i]), 2)
            denom = length * (math.pow(length, 2) - 1)
            return 1.0 - 6.0 * pairwise_stability / denom

        elif self.stability_metric == KUNCHEVA:
            for i in range(1, length):
                sub_v1, sub_v2 = v1_ranks[:i+1], v2_ranks[:i+1]   # compare subsections of array
                numer = len(np.intersect1d(sub_v1, sub_v2)) - math.pow(i, 2) / length
                denom = i - math.pow(i, 2) * length
                pairwise_stability += numer/denom
            return pairwise_stability / length

        else:
            raise NotImplementedError("Use Jaccard similarity.")

    def get_stability_pairwise(self, imp_data: np.ndarray) -> float:
        """
        Find the stability of the importance array using pairwise stability metrics.
        """
        stability = 0
        length = len(imp_data)
        ranking_maps = rankdata(imp_data, method="dense", axis=1)   # rank 1 is the most important feature!

        for i in range(length - 1):
            for j in range(i + 1, length):
                stability += self.get_pairwise_stability(
                    v1_ranks=ranking_maps[i],
                    v2_ranks=ranking_maps[j],
                )
        return 2.0 * stability / (length * (length - 1))

    @staticmethod
    def get_stability_freq(imp_data: np.ndarray, best_k_features: int) -> float:
        """
        Find the stability of the importance array using frequency of feature subsets
        """
        N = len(imp_data)
        F = len(imp_data[0])
        assert F > best_k_features, "Feature subset size should be less than feature dimension"

        # preprocess data
        best_idcs = np.argsort(-imp_data)[:, :best_k_features]
        for i in range(N):
            top_k_imps = imp_data[i][best_idcs[i]]
            # zero-out the top k importances that aren't below threshold (default 0.0)
            best_idcs[i] = np.where(top_k_imps >= MIN_IMPORTANCE, best_idcs[i], -1)

        c_var = 0
        for i in range(F):
            freq = np.sum(np.where(best_idcs == i, 1, 0)) / N
            c_var += N * freq * (1 - freq) / (N - 1)

        # d_bar is the avg number of features selected over N feature sets (default k).
        d_bar = np.sum(np.where(best_idcs >= 0, 1, 0)) / N
        numer = c_var/F
        denom = d_bar * (1 - d_bar/F) / F
        # print(numer, denom)
        return 1 - numer/denom

    def run_baseline(self, best_k_features: int = None) -> float:
        imp_matrix = self.get_importance_array()
        print("this is importance array: \n", imp_matrix)
        print("this is importance array indices: \n", np.argsort(-imp_matrix))
        # print("\n\n")
        if best_k_features is None:
            return self.get_stability_pairwise(imp_matrix)
        else:
            return self.get_stability_freq(imp_matrix, best_k_features)



