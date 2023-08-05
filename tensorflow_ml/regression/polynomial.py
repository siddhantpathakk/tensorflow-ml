import numpy as np
import tensorflow as tf
from linear import LinearRegression

class PolynomialRegression(LinearRegression):
    def __init__(self):
        super().__init__()
        self.degree = 1

    def set_params(self, params: dict):
        super().set_params(params)
        if 'degree' in params:
            self.degree = params['degree']

    def get_params(self) -> dict:
        params = super().get_params()
        params['degree'] = self.degree
        return params

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        # Polynomial features
        X_poly = self._get_polynomial_features(X, self.degree)

        # Call the fit method from the LinearRegression class
        super().fit(X_poly, y, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Polynomial features
        X_poly = self._get_polynomial_features(X, self.degree)

        # Call the predict method from the LinearRegression class
        return super().predict(X_poly)

    def _get_polynomial_features(self, X: np.ndarray, degree: int) -> np.ndarray:
        num_samples = X.shape[0]
        num_features = X.shape[1]
        polynomial_features = []

        for d in range(1, degree + 1):
            degree_features = X ** d
            polynomial_features.append(degree_features)

        return np.hstack([np.ones((num_samples, 1)), *polynomial_features])
