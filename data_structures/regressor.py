import numpy as np
from abc import ABC
from typing import Tuple


class Regressor(ABC):
    def predict_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version of predict(). Allows for passing many datapoints at once
        :param data: Data to regress
        :return: Array of predicted regression values
        """
        predictions = np.empty(len(data))
        for d_idx, datapoint in enumerate(data):
            predictions[d_idx] = self.predict(datapoint)
        return predictions