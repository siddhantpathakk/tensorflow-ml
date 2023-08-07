import numpy as np
import tensorflow as tf

class LinearRegression:
    """
    Defines a Linear Regression model class in Python with methods for setting parameters, 
    fitting the model to data, predicting values, evaluating the model, and retrieving the coefficients.
    
    Linear regression is a supervised learning algorithm that is used to predict the value of a
    continuous variable (y) based on one or more predictor variables (x). The goal of linear regression
    is to find the best-fitting straight line through the points. The best-fitting line is called a
    regression line. The regression line is defined by the equation y = mx + b, where m is the slope
    of the line and b is the y-intercept.
    
    The linear regression model can be represented as y = X * w + b, where X is the input data, y is
    the target variable, w is the model coefficients, and b is the bias term.
    """
    def __init__(self):
        self.x, self.y = None, None
        self.learning_rate, self.num_epochs, self.coefficients = None, None, None
        self.batch_size, self.reg_strength, self.tolerance = None, None, None
        self.patience = None
        self.verbose = True
        
    def set_params(self, params : dict):
        """
        The function `set_params` sets the parameters for a linear regression model.
        
        Parameters
        ----------
            params : dict
                A dictionary containing the values of the learning rate, number of epochs, batch size,
                regularization strength, tolerance, and patience. 
        """

        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.reg_strength = params['reg_strength']
        self.tolerance = params['tolerance']
        self.patience = params['patience']
        
        if self.verbose:
            print(f"LinearRegression(learning_rate={self.learning_rate}, num_epochs={self.num_epochs}, batch_size={self.batch_size}, reg_strength={self.reg_strength}, tolerance={self.tolerance}, patience={self.patience})")
    
    def get_params(self) -> dict:
        """
        The function `get_params` returns a dictionary containing the values of various parameters.
        
        Returns
        -------
            params : dict
                A dictionary containing the values of the learning rate, number of epochs, batch size,
                regularization strength, and tolerance.
        """

        return {"learning_rate": self.learning_rate, "num_epochs": self.num_epochs, 
                "batch_size": self.batch_size, "reg_strength": self.reg_strength, 
                "tolerance": self.tolerance}
         
    def fit(self, 
            X : np.ndarray, y : np.ndarray, 
            X_val : np.ndarray = None,
            y_val : np.ndarray = None):
        """
        The `fit` function trains a regression model using mini-batch gradient descent with early
        stopping based on validation loss.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter `X` is a numpy array that represents the input features for training the
                model. It has shape `(num_samples, num_features)`, where `num_samples` is the number of
                training samples and `num_features` is the number of features for each sample
                
            y : np.ndarray
                The parameter `y` represents the target variable or the dependent variable in the
                supervised learning problem. It is a numpy array that contains the true values of the
                target variable for the corresponding samples in the input data `X`. The shape of `y` is
                `(num_samples,)`, where `num_samples` is the number of training samples
                
            X_val : np.ndarray
                X_val is the validation set features. It is an optional parameter that allows you to
                evaluate the model's performance on a separate validation set during training. It should
                be a numpy array of shape (num_samples, num_features), where num_samples is the number of
                samples in the validation set and num_features
                
            y_val : np.ndarray
                `y_val` is the validation set labels. It is an array containing the true values of the
                target variable for the validation set
        """
        num_samples, num_features = X.shape

        X = np.hstack((np.ones((num_samples, 1)), X))  # Add a bias term (1) to the scaled features

        # Convert numpy arrays to TensorFlow tensors
        X = tf.constant(X, dtype=tf.float32)
        y = tf.constant(y.reshape(-1, 1), dtype=tf.float32)

        # Initialize the coefficients (weights) randomly
        self.coefficients = tf.Variable(tf.random.normal((num_features + 1, 1), stddev=0.01))

        # Learning Rate Scheduling with momentum-based optimizer (Adam)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Training loop with Mini-batch Gradient Descent and early stopping
        num_batches = int(np.ceil(num_samples / self.batch_size))
        best_val_loss = float('inf')
        patience_count = 0
        for epoch in range(self.num_epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for batch in range(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_X = tf.gather(X, batch_indices)
                batch_y = tf.gather(y, batch_indices)

                with tf.GradientTape() as tape:
                    predictions = tf.matmul(batch_X, self.coefficients)
                    loss = tf.reduce_mean(tf.square(predictions - batch_y))

                gradients = tape.gradient(loss, self.coefficients)
                optimizer.apply_gradients(zip([gradients], [self.coefficients]))

            # Calculate validation loss and perform early stopping with patience
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                if val_loss + self.tolerance < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break  # Stop training if validation loss doesn't improve for 'patience' epochs
            
            if self.verbose:
                if X_val is None or y_val is None:
                    print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {loss:.5f}")
                else:
                    print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {loss:.5f} - Val Loss: {val_loss:.5f}")

    def score(self, X : np.ndarray, y : np.ndarray) -> float:
        """
        The function calculates the R-squared score between the predicted and true values of a regression
        model.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X is an ndarray (numpy array) that represents the input data for which we
                want to calculate the score. It could be a matrix or a vector depending on the specific
                problem. For example, if we are trying to predict the price of a house based on its size
                and number of bedrooms, then X will be a matrix with shape (num_samples, num_features),
                where num_samples is the number of samples or observations and num_features is the number
                of features or variables
            y : np.ndarray
                The parameter `y` represents the true labels or target values of the dataset. It is an
                array-like object containing the actual values that we are trying to predict for the
                corresponding samples in the input data `X`. For example, if we are trying to predict the
                price of a house based on its size and number of bedrooms, then y will be a vector of
                shape (num_samples,), where num_samples is the number of samples or observations in the
                dataset.
        Returns
        -------
            r2 : float
                The R-squared score, which is a measure of how well the predicted values (y_pred) match
                the true values (y_true) in the given dataset.
        """

        y_pred = self.predict(X)
        y_true = y
        
        y_true_mean = np.mean(y_true)
        ssr = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - y_true_mean) ** 2)

        r2 = 1 - (ssr / sst)
        return r2
        
    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        The `predict` function takes an input array `X`, adds a bias term to it, performs matrix
        multiplication with the model coefficients, and returns the predictions as a flattened numpy array.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X is an input array of shape (num_samples, num_features), where num_samples 
                is the number of samples and num_features is the number of features for each sample
        Returns
        -------
            predictions : np.ndarray
                A numpy array of predictions.
        """

        if self.coefficients is None:
            raise ValueError("Model has not been trained. Please call 'fit' first.")

        num_samples = X.shape[0]
        X = np.hstack((np.ones((num_samples, 1)), X))  # Add a bias term (1) to the input features
        X = tf.constant(X, dtype=tf.float32)

        predictions = tf.matmul(X, self.coefficients)
        return predictions.numpy().flatten()
    
    def evaluate(self, X, y):
        """
        The evaluate function calculates the mean squared error between the predicted values and the actual
        values.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter `X` represents the input data or features. It is a matrix or array-like object
                with shape (n_samples, n_features), where n_samples is the number of samples or observations
                and n_features is the number of features or variables.
            y : np.ndarray
                The parameter `y` represents the true values of the target variable. It is an array-like
                object containing the actual values that we are trying to predict for the corresponding
                samples in the input data `X`.
        Returns
        -------
            mse : float
                The mean squared error (mse) between the predicted and true values.
        """

        y_pred = self.predict(X)
        mse = np.mean(np.square(y - y_pred))
        return mse
    
    def get_coeff(self) -> np.ndarray:
        """
        The function `get_coeff` returns the coefficients of a trained model as a flattened numpy array.
        
        Returns
        -------
            coefficients : np.ndarray
                The coefficients of the model as a flattened numpy array.
        """

        if self.coefficients is None:
            raise ValueError("Model has not been trained. Please call 'fit' first.")
        return self.coefficients.numpy().flatten()
