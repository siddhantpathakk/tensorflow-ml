import numpy as np
import tensorflow as tf
from .linear import LinearRegression

class RidgeRegression(LinearRegression):
    """
    Defines a Ridge Regression model using Mini-batch Gradient Descent with early
    stopping based on validation loss.
    
    Ridge Regression is a linear regression model with L2 regularization. It is used to prevent overfitting
    and perform feature selection. The L2 regularization term is the sum of the squares of the coefficients.
    It is used to shrink the coefficients of the model to zero, thereby reducing the number of features used
    in the model. This is useful when the dataset has a large number of features and only a few of them are
    important for the model.
    
    """
    
    def set_params(self, params: dict):
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.reg_strength = params['reg_strength']
        self.tolerance = params['tolerance']
        self.patience = params['patience']
        if self.verbose:
            print(f"RidgeRegression(learning_rate={self.learning_rate}, num_epochs={self.num_epochs}, batch_size={self.batch_size}, reg_strength={self.reg_strength}, tolerance={self.tolerance}, patience={self.patience})")
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
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