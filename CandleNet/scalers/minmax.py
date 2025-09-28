import numpy as np


class MinMaxScaler:

    @staticmethod
    def fit_transform(
        X: np.ndarray, feature_range: tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Scale features to a given range, typically [0, 1].
        This method scales the data to fit within the specified range by removing the minimum and scaling according to the range (max - min).

        Parameters:
        X (np.ndarray): The input data to be scaled. Shape (n_samples, n_features).
        feature_range (tuple[float, float]): Desired range of transformed data.

        Returns:
        np.ndarray: The scaled data.

        Note:
        The feature_range parameter specifies the desired range of transformed data.
        """
        X = np.asarray(X, dtype=np.float64)
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)

        a, b = feature_range
        scale = np.zeros_like(data_min)
        scale[data_max != data_min] = (b - a) / (
            data_max[data_max != data_min] - data_min[data_max != data_min]
        )
        min_ = a - data_min * scale

        X_trans = X * scale + min_

        if original_shape == X_trans.shape:
            return X_trans
        if len(original_shape) == 1:
            return X_trans.squeeze()
        return X_trans.reshape(original_shape)
