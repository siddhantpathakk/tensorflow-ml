import numpy as np
import tensorflow as tf

class LogisticRegression:
    def __init__(self):
        self.learning_rate = None
        self.num_epochs = None
        self.batch_size = None
        self.reg_strength = None
        self.coefficients = None
        self.verbose = False
        self.regularization = None
        self.early_stopping_patience = None

    def set_params(self, params: dict):
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.reg_strength = params['reg_strength']
        self.batch_size = params['batch_size']
        self.early_stopping_patience = params['early_stopping_patience']
        self.regularization = params["regularization"]

        self.learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

    def get_params(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "reg_strength": self.reg_strength,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience
        }

    def _scale_features(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        scaled_X = (X - mean) / std
        return scaled_X

    def sigmoid(self, z):
        return 1 / (1 + tf.exp(-z))

    def softmax(self, z):
        return tf.nn.softmax(z)

    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        num_samples = X.shape[0]
        if random_state:
            np.random.seed(random_state)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_test_samples = int(test_size * num_samples)
        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test

    def _regularization_term(self, coefficients):
        if self.regularization == 'l1':
            return self.reg_strength * tf.reduce_sum(tf.abs(coefficients))
        elif self.regularization == 'l2':
            return self.reg_strength * tf.reduce_sum(tf.square(coefficients))
        else:
            return 0
    def fit(self, X, y, X_val=None, y_val=None):
        num_samples, num_features = X.shape

        X = np.hstack((np.ones((num_samples, 1)), X))  # Add a bias term (1) to the scaled features

        # Convert numpy arrays to TensorFlow tensors
        X = tf.constant(X, dtype=tf.float32)
        y = tf.constant(y.reshape(-1, 1), dtype=tf.float32)

        # Calculate class weights for data imbalance
        class_weights = self._calculate_class_weights(y)

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
                    logits = tf.matmul(batch_X, self.coefficients)
                    predictions = self.sigmoid(logits)
                    loss = -tf.reduce_mean(class_weights * batch_y * tf.math.log(predictions) + class_weights * (1 - batch_y) * tf.math.log(1 - predictions))
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

    def _calculate_class_weights(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = y.shape[0]
        class_weights = total_samples / (len(unique_classes) * class_counts)
        max_weight = np.max(class_weights)
        class_weights = max_weight / class_weights
        class_weights = tf.constant(class_weights, dtype=tf.float32)
        return class_weights
        
    def predict_proba(self, X):
        X_scaled = self._scale_features(X)
        X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
        X_scaled = tf.constant(X_scaled, dtype=tf.float32)

        logits = tf.matmul(X_scaled, self.coefficients)  # Use matrix multiplication directly
        probabilities = self.sigmoid(logits)
        return probabilities.numpy()

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        y_pred_proba = self.predict_proba(X)

        # Clip probabilities to avoid log(0) and log(1) issues
        y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)

        y_binary = np.array([np.array([1.0 if i == label else 0.0 for i in range(len(np.unique(y)))]) for label in y])

        # Use TensorFlow's binary cross-entropy loss function
        loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_proba).numpy()
        loss = np.mean(loss)

        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = np.mean(y_pred == y)

        return accuracy, loss

    def evaluate(self, X, y):
        return self.score(X, y)