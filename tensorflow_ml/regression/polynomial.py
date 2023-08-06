import numpy as np
import tensorflow as tf
from .linear import LinearRegression

class PolynomialRegression(LinearRegression):
    """
    Defines a PolynomialRegression class that
    extends the LinearRegression class and adds
    functionality for polynomial regression.
    """
    def __init__(self):
        super().__init__()
        self.degree = 1

    def set_params(self, params: dict):
        """
        The function `set_params` sets the parameters for a Polynomial Regression model.
        
        :param params: The `params` dictionary contains the following parameters:
        :type params: dict
        """
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
        """
        The function `get_params` returns a dictionary of parameters, including the degree parameter.
        :return: The `get_params` method is returning a dictionary containing the parameters of the object.
        It first calls the `get_params` method of the superclass (assuming there is one) and then adds an
        additional parameter called 'degree' to the dictionary before returning it.
        """
        params = super().get_params()
        params['degree'] = self.degree
        return params

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        The `fit` function fits a polynomial regression model to the input data using the `fit` method from
        the `LinearRegression` class.
        
        :param X: The input features for training the model. It is a numpy array of shape (n_samples,
        n_features), where n_samples is the number of samples and n_features is the number of input features
        :type X: np.ndarray
        :param y: The parameter `y` is the target variable or the dependent variable. It represents the
        values that we are trying to predict or model. In a supervised learning problem, `y` is typically a
        one-dimensional array or a column vector containing the target values for each sample in the input
        data `X`
        :type y: np.ndarray
        :param X_val: The validation set input features. It is an optional parameter and is used for
        evaluating the model's performance during training
        :type X_val: np.ndarray
        :param y_val: The parameter `y_val` is an optional array-like object that represents the target
        variable for the validation set. It is used when you want to evaluate the performance of the model
        on a separate validation set during the training process. If provided, it should have the same
        length as `X_val`
        :type y_val: np.ndarray
        """
        # Polynomial features
        X_poly = self._get_polynomial_features(X, self.degree)

        # Call the fit method from the LinearRegression class
        super().fit(X_poly, y, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        The `predict` function takes an input array `X`, applies polynomial features to it, and then calls
        the `predict` method from the `LinearRegression` class to make predictions.
        
        :param X: An input array of shape (n_samples, n_features) containing the features for which
        predictions are to be made
        :type X: np.ndarray
        :return: The predict method is returning a numpy array.
        """
        # Polynomial features
        X_poly = self._get_polynomial_features(X, self.degree)

        # Call the predict method from the LinearRegression class
        return super().predict(X_poly)

    def _get_polynomial_features(self, X: np.ndarray, degree: int) -> np.ndarray:
        """
        The function `_get_polynomial_features` takes an input array `X` and a degree `degree`, and returns
        an array of polynomial features up to the specified degree.
        
        :param X: An input array of shape (num_samples, num_features) where num_samples is the number of
        samples and num_features is the number of features
        :type X: np.ndarray
        :param degree: The "degree" parameter represents the maximum degree of the polynomial features to be
        generated. For example, if degree = 3, then the function will generate polynomial features up to
        degree 3
        :type degree: int
        :return: a numpy array that contains the polynomial features of the input array X. The polynomial
        features are computed up to the specified degree. The returned array has a shape of (num_samples,
        num_features * degree + 1), where num_samples is the number of samples in X and num_features is the
        number of features in X. The first column of the returned array contains a column of
        """
        num_samples = X.shape[0]
        num_features = X.shape[1]
        polynomial_features = []

        for d in range(1, degree + 1):
            degree_features = X ** d
            polynomial_features.append(degree_features)

        return np.hstack([np.ones((num_samples, 1)), *polynomial_features])
