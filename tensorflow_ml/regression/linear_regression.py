import numpy as np
import tensorflow as tf

class LinearRegression:
    """
    Linear Regression model using Tensorflow. This model uses Mini-batch Gradient Descent with early stopping for training.
    This model supports L2 regularization, momentum-based optimization and learning rate scheduling.
    
    Parameters
    -------
        `learning_rate` (float): 
            Learning rate for the model.
        
        `num_epochs (int)`: 
            Number of epochs for training the model.
        
        `batch_size (int)`: 
            Batch size for training the model.
        
        `reg_strength (float)`: 
            Regularization strength for the model.
        
        `tolerance (float)`: 
            Tolerance for early stopping.
        
        `patience (int)`: 
            Patience for early stopping.
        
        `verbose (bool = True)`: 
            Print log status

    """
    def __init__(self):
        self.x, self.y = None, None
        self.learning_rate, self.num_epochs, self.coefficients = None, None, None
        self.batch_size, self.reg_strength, self.tolerance = None, None, None
        self.patience = None
        self.verbose = True
        
    def set_params(self, params : dict):
        """Set the training parameters of the model.
        Args:
            params (dict): Dictionary of parameters to be set with keys as parameter names and values as parameter values.
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
        """Retrieve the training parameters of the model.

        Returns:
            dict: Dictionary of parameters with keys as parameter names and values as parameter values.
        """
        return {"learning_rate": self.learning_rate, "num_epochs": self.num_epochs, 
                "batch_size": self.batch_size, "reg_strength": self.reg_strength, 
                "tolerance": self.tolerance}
         
    def fit(self, 
            X : np.ndarray, y : np.ndarray, 
            X_val : np.ndarray = None,
            y_val : np.ndarray = None):
        """Fit the model to the data.

        Args:
            X (np.ndarray): Data to be fitted.
            y (np.ndarray): Data to be fitted.
            X_val (np.ndarray, optional): Validation set. Defaults to None.
            y_val (np.ndarray, optional): Validation set. Defaults to None.
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
                    if self.reg_strength > 0.0:
                        loss += self.reg_strength * tf.reduce_sum(tf.square(self.coefficients))

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
        """Calculate the R^2 score of the model.
        Also known as the coefficient of determination of the prediction

        Args:
            X (np.ndarray): Data to be scored.
            y (np.ndarray): Data to be scored.

        Returns:
            float: R^2 score of the model
        """
        y_pred = self.predict(X)
        y_true = y
        
        y_true_mean = np.mean(y_true)
        ssr = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - y_true_mean) ** 2)

        r2 = 1 - (ssr / sst)
        return r2
        
    def predict(self, X : np.ndarray) -> np.ndarray:
        """Predict the values for the given data.

        Args:
            X (np.ndarray): Data

        Returns:
            np.ndarray: Predicted values.
        
        Raises:
            ValueError: Model has not been trained. Please call 'fit' first.
        """
        if self.coefficients is None:
            raise ValueError("Model has not been trained. Please call 'fit' first.")

        num_samples = X.shape[0]
        X = np.hstack((np.ones((num_samples, 1)), X))  # Add a bias term (1) to the input features
        X = tf.constant(X, dtype=tf.float32)

        predictions = tf.matmul(X, self.coefficients)
        return predictions.numpy().flatten()
    
    def evaluate(self, X, y):
        """Evaluate the model on the given data on MSE.
        Args:
            X (np.ndarray): Data
            y (np.ndarray): Data    
        Returns:
            float: Mean squared error of the model
        """
        y_pred = self.predict(X)
        mse = np.mean(np.square(y - y_pred))
        return mse
    
    def get_coeff(self) -> np.ndarray:
        """Get the coefficients of the model.

        Raises:
            ValueError: Model has not been trained. Please call 'fit' first.

        Returns:
            np.ndarray : Coefficients of the model.
        """
        if self.coefficients is None:
            raise ValueError("Model has not been trained. Please call 'fit' first.")
        return self.coefficients.numpy().flatten()
