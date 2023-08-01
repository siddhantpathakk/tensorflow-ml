import tensorflow as tf
import numpy as np
import os

from utils import loss, grad, param_list

class LinearRegression:
    def __init__(self):
        self.W = tf.Variable(np.random.randn()) # initial, random, value for predicted weight (m)
        self.B = tf.Variable(np.random.randn()) # initial, random, value for predicted bias (c)
        self.params = {(key, 0) for key in param_list}
        self.x, self.y = None, None
    
    def set_params(self, parameters : dict):
        """Set the training parameters of the model.

        Args:
            parameters (dict): Dictionary of parameters to be set with keys as parameter names and values as parameter values.

        Raises:
            KeyError: Parameter `key` is not supported.
        """
        for key, value in parameters.items():
            if key in self.params.keys():
                self.params[key] = value
            else:
                raise KeyError(f"Parameter {key} is not supported.")
    
    def get_params(self) -> dict:
        """Retrieve the training parameters of the model.
        
        Returns:
            dict: Dictionary of parameters with keys as parameter names and values as parameter values.
        """
        return self.params
    
    def fit(self, x : np.ndarray, y : np.ndarray, verbose : bool=False):
        """Fit the model to the data.
        
        Args:
            x (np.ndarray): X values of the data.
            y (np.ndarray): y values of the data.
            verbose (bool, optional): Whether to print the loss at each/display-step. Defaults to False.
        """
        self.x, self.y = x, y
        for step in range(self.params['training_steps']): # iterate for each training step
            # direction (sign)  and value of the gradient of our loss w.r.t weight and bias
            deltaW, deltaB = grad(x, y, self.W, self.B) 
            
            change_W = deltaW *  self.params["learning_rate"] # adjustment amount for weight
            change_B = deltaB *  self.params["learning_rate"] # adjustment amount for bias
            
            self.W.assign_sub(change_W) # subract change_W from W
            self.B.assign_sub(change_B) # subract change_B from B
            
            if verbose:
                if step==0 or step % self.params["display_step"] == 0:
                    # print(deltaW.numpy(), deltaB.numpy()) # uncomment if you want to see the gradients
                    print("Loss at step {:02d}: {:.6f}".format(step, loss(x, y, self.W, self.B)))
        
    def predict(self, X : np.ndarray) -> np.ndarray:
        """Predict the values for the given data.

        Args:
            X (np.ndarray): Data

        Returns:
            np.ndarray: Predicted values.
        """
        predicted_value = []
        for to_predict in X:
            predicted_value.append(to_predict * self.W.numpy() + self.B.numpy())
        assert len(predicted_value) == len(X)
        return predicted_value
    
    def evaluate(self, metrics:str = 'mse') -> np.float32:
        """Evaluate the model on the given data.

        Args:
            metrics (str, optional): Evaluation metric to be used. Defaults to 'mse'. Currently only 'mse' is supported.
        Returns:
            float: Evaluation metric value.
        """
        y_true = np.array(self.y)
        y_pred = np.array(self.predict(self.x))
        
        actual_values = tf.constant(y_true, dtype=tf.float32)
        predicted_values = tf.constant(y_pred, dtype=tf.float32)
        return tf.keras.losses.mean_squared_error(actual_values, predicted_values).numpy()

    def get_coeffs(self) -> tuple:
        """Get the coefficients of the model.

        Returns:
            tuple: Tuple of coefficients (weight, bias).
        """
        return self.W.numpy(), self.B.numpy()