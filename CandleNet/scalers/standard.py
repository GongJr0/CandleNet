import numpy as np


class StandardScaler:

    @staticmethod
    def fit_transform(X: np.ndarray, ddof: float = 1) -> np.ndarray:
        """
        Standardize features by removing the mean and scaling to unit variance.
        This method scales the data to have a mean of 0 and a standard deviation of 1.

        Parameters:
        X (np.ndarray): The input data to be scaled. Shape (n_samples, n_features).

        Returns:
        np.ndarray: The scaled data.
        """

        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, ddof=ddof, axis=0)  # sample standard deviation
        std = np.where(std == 0, 1e-6, std)

        return (X - mean) / std
