import numpy as np
import tensorflow as tf
from .linear import LinearRegression

class RidgeRegression(LinearRegression):
    """
    An implementation of Ridge Regression
    with Mini-batch Gradient Descent and early stopping.
    
    :param params: The `params` dictionary contains the
    following parameters:
    :type params: dict
    """
    
    def set_params(self, params: dict):
        """
        The function `set_params` sets the parameters for a Ridge Regression model and prints them if the
        `verbose` flag is set to True.
        
        :param params: The `params` dictionary contains the following parameters:
        :type params: dict
        """
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.reg_strength = params['reg_strength']
        self.tolerance = params['tolerance']
        self.patience = params['patience']
        if self.verbose:
            print(f"RidgeRegression(learning_rate={self.learning_rate}, num_epochs={self.num_epochs}, batch_size={self.batch_size}, reg_strength={self.reg_strength}, tolerance={self.tolerance}, patience={self.patience})")
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        The `fit` function trains a linear regression model using Mini-batch Gradient Descent with early
        stopping and L2 regularization.
        
        :param X: The input features for training the model. It should be a numpy array of shape
        (num_samples, num_features), where num_samples is the number of training samples and num_features is
        the number of features for each sample
        :type X: np.ndarray
        :param y: The parameter `y` represents the target variable or the dependent variable in the
        supervised learning problem. It is a numpy array that contains the true values of the target
        variable for the corresponding samples in the input data `X`. The shape of `y` is `(num_samples,)`,
        where `num_samples
        :type y: np.ndarray
        :param X_val: X_val is the validation set of features. It is an ndarray containing the features of
        the validation set
        :type X_val: np.ndarray
        :param y_val: `y_val` is the validation target variable. It is an array containing the true values
        of the target variable for the validation set
        :type y_val: np.ndarray
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

                    # L2 regularization (ridge penalty)
                    if self.reg_strength > 0.0:
                        l2_loss = self.reg_strength * tf.reduce_sum(tf.square(self.coefficients))
                        loss += l2_loss

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