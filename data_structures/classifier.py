import numpy as np
from abc import ABC
from typing import Tuple


class Classifier(ABC):
    def predict_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version of predict(). Allows for passing many datapoints at once

        :param data: Data to classify
        :return: Pair of numpy arrays where the first contains the labels and the second contains the class probs
        """
        pred_labels = np.empty(len(data))
        # why is pred_probs 2-dimensional? self.predict returns a float for pred_probs.
        pred_probs = np.empty((len(data), len(self.classes)))
        for d_idx, datapoint in enumerate(data):
            pred_labels[d_idx], pred_probs[d_idx] = self.predict(datapoint)
        return pred_labels, pred_probs
