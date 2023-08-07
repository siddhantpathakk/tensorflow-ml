import numpy as np
import tensorflow as tf
from .linear import LinearRegression

class PolynomialRegression(LinearRegression):
    """
    Defines a PolynomialRegression class that
    extends the LinearRegression class and adds
    functionality for polynomial regression.
    
    Polynomial regression is a form of linear regression in which the relationship between the independent
    variable x and the dependent variable y is modeled as an nth degree polynomial. Polynomial regression fits
    a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |
    x), and has been used to describe nonlinear phenomena such as the growth rate of tissues, the distribution
    of carbon isotopes in lake sediments, and the progression of disease epidemics.

    """
    def __init__(self):
        super().__init__()
        self.degree = 1

    def set_params(self, params: dict):
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.reg_strength = params['reg_strength']
        self.tolerance = params['tolerance']
        self.patience = params['patience']
        if 'degree' in params:
            self.degree = params['degree']        
        if self.verbose:
            print("PolynomialRegression(learning_rate={}, num_epochs={}, batch_size={}, reg_strength={}, tolerance={}, patience={}, degree={})".format(self.learning_rate, self.num_epochs, self.batch_size, self.reg_strength, self.tolerance, self.patience,self.degree))


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
