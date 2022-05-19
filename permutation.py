import sklearn.datasets
import numpy as np
import math
from scipy.stats import rankdata

from data_structures.forest_classifier import ForestClassifier
from data_structures.forest_regressor import ForestRegressor
from data_structures.tree_classifier import TreeClassifier
from data_structures.tree_regressor import TreeRegressor
from utils.utils import class_to_idx
from utils.constants import MAB, EXACT, JACCARD, SPEARMAN, KUNCHEVA


class PermutationImportance:
    """
    Class for determining feature importance based on the permutation model in scikit-learn.
    Use this to find an array of size K X F were K is the number of forets and F is the dimension of features.
    """
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        max_depth: int = 5,
        num_forests: int = 5,
        num_trees_per_forest: int = 10,
        is_classification: bool = True,
        solver: str = MAB,
        stability_metric: str = JACCARD,
        budget_per_forest: int = None,
    ):
        assert num_forests > 1, "we need at least two forests"
        self.data = data
        self.labels = labels
        self.max_depth = max_depth
        self.budget_per_forest = budget_per_forest
        self.num_forests = num_forests
        self.num_trees_per_forest = num_trees_per_forest
        self.is_classification = is_classification
        self.solver = solver
        self.stability_metric = stability_metric
        self.forests = []

    def train_forest(self) -> None:
        """
        Trains a forest and adds it to the list of forests
        """
        if self.is_classification:
            forest = ForestClassifier(
                data=self.data,
                labels=self.labels,
                max_depth=self.max_depth,
                n_estimators=self.num_trees_per_forest,
                solver=self.solver,
                budget=self.budget_per_forest,
            )
        else:
            forest = ForestRegressor(
                data=self.data,
                labels=self.labels,
                max_depth=self.max_depth,
                n_estimators=self.num_trees_per_forest,
                solver=self.solver,
                budget=self.budget_per_forest,
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
            self.train_forest()

    def compute_importance_vec(self, forest_idx: int) -> np.ndarray:
        """
        Compute the importance vector for the forest corresponding to forest_idx.
        Sorts the indices of the feature by relative importance. Assumes that all forests have been trained.

        :return: sorted array of indices
        """
        importance_vec = []
        forest = self.forests[forest_idx]
        num_trees = len(forest.trees)   # number of trees depends on the budget

        model_score = np.sum(forest.predict_batch(self.data)[0] == self.labels)
        for feature_idx in range(len(self.data[0])):
            c_score = 0
            for tree_idx in range(num_trees):
                np.random.shuffle(self.data[:, feature_idx])  # shuffles in-place
                tree = forest.trees[tree_idx]
                c_score += np.sum(tree.predict_batch(self.data)[0] == self.labels)
            avg_c_score = c_score / num_trees
            importance_vec.append(model_score - avg_c_score)
            # importance_vec.append((model_score - avg_c_score)/len(self.data))
        return np.asarray(importance_vec)

    def get_importance_array(self) -> np.ndarray:
        """
        Trains all of its forests and computes the importance vector for each of the trained forests.
        This is the main function that an object of this class will call.

        :return: an array of importance vectors.
        """
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

    def get_stability(self, imp_data: np.ndarray) -> float:
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

    def run_baseline(self) -> float:
        imp_matrix = self.get_importance_array()
        return self.get_stability(imp_matrix)



