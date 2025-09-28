import numpy as np


class RobustScaler:

    @staticmethod
    def fit_transform(X: np.ndarray) -> np.ndarray:
        """
        Scale features using statistics that are robust to outliers.
        This method removes the median and scales the data according to the interquartile range (IQR).

        Parameters:
        X (np.ndarray): The input data to be scaled. Accepts (n_samples,) or (n_samples, n_features).

        Returns:
        np.ndarray: The scaled data with shape (n_samples, n_features), matching scikit-learn conventions.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Compute per-feature robust center and scale
        med = np.nanmedian(X, axis=0)
        # Use 25th and 75th percentiles per feature
        # (avoid specifying method to keep broad NumPy compatibility)
        q25, q75 = np.nanpercentile(X, [25.0, 75.0], axis=0)
        iqr = q75 - q25
        # Avoid division by zero for constant features (match sklearn behavior)
        iqr = np.where(iqr == 0.0, 1.0, iqr)

        X_tr = (X - med) / iqr
        return X_tr
