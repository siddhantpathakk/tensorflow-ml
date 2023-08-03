import numpy as np
import tensorflow as tf


class LogisticRegression:
    def __init__(self):
        self.learning_rate = None
        self.num_epochs = None
        self.batch_size = None
        self.reg_strength = None
        self.coefficients = None
        self.verbose = True
        self.regularization = None

    def set_params(self, params : dict):
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.reg_strength = params['reg_strength']
        self.batch_size = params['batch_size']
        self.early_stopping_patience = params['early_stopping_patience']
        self.regularization = params["regularization"]
        
        self.learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate,
                                                                                      decay_steps=1000,
                                                                                      decay_rate=0.96,
                                                                                      staircase=True)
        
    def get_params(self) -> dict:

        return {"learning_rate": self.learning_rate, 
                "num_epochs": self.num_epochs, 
                "reg_strength": self.reg_strength,
                "batch_size": self.batch_size,
                "early_stopping_patience": self.early_stopping_patience}
         
       
    def _scale_features(self, X):
        # Mean normalization (subtract mean and divide by standard deviation)
        # mean = np.mean(X, axis=0)
        # std = np.std(X, axis=0)
        # scaled_X = (X - mean) / std
        # return scaled_X
        return X
        
    def sigmoid(self, z):
        return 1 / (1 + tf.exp(-z))
    
    def softmax(self, z):
        return tf.nn.softmax(z)

    def train_val_split(self, X, y, test_size=0.2, random_state=None):
        num_samples = X.shape[0]
        if random_state:
            np.random.seed(random_state)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_val_samples = int(test_size * num_samples)
        val_indices = indices[:num_val_samples]
        train_indices = indices[num_val_samples:]
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        return X_train, X_val, y_train, y_val

    def _regularization_term(self, coefficients):
        if self.regularization == 'l1':
            return self.reg_strength * tf.reduce_sum(tf.abs(coefficients))
        elif self.regularization == 'l2':
            return self.reg_strength * tf.reduce_sum(tf.square(coefficients))
        else:
            return 0

    def fit(self, X, y):
        X_scaled = self._scale_features(X)
        X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

        X_train, X_val, y_train, y_val = self.train_val_split(X_scaled, y, test_size=0.2, random_state=42)

        X_train, X_val = X_train.astype(np.float32), X_val.astype(np.float32)

        num_classes = len(np.unique(y))
        num_features = X_scaled.shape[1]

        self.coefficients = []
        for class_label in range(num_classes):
            y_binary = (y_train == class_label).astype(np.float32)

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_binary)).shuffle(len(X_train)).batch(self.batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, (y_val == class_label).astype(np.float32))).batch(self.batch_size)


            # Initialize the coefficients with Glorot uniform initialization
            coefficients_class = tf.Variable(tf.keras.initializers.GlorotUniform()((num_features, 1)))

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,clipvalue=1.0)

            best_val_loss = float('inf')
            early_stopping_count = 0

            for epoch in range(self.num_epochs):
                train_loss = 0
                for batch_X, batch_y in train_dataset:
                    with tf.GradientTape() as tape:
                        z = tf.matmul(batch_X, coefficients_class)
                        probabilities = self.sigmoid(z)
                        loss = -tf.reduce_mean(batch_y * tf.math.log(probabilities) + (1 - batch_y) * tf.math.log(1 - probabilities))
                        loss += self._regularization_term(coefficients_class)

                    gradients = tape.gradient(loss, coefficients_class)
                    optimizer.apply_gradients(zip([gradients], [coefficients_class]))

                    train_loss += loss

                train_loss /= len(train_dataset)

                val_loss = 0
                for batch_X, batch_y in val_dataset:
                    z = tf.matmul(batch_X, coefficients_class)
                    probabilities = self.sigmoid(z)
                    loss = -tf.reduce_mean(batch_y * tf.math.log(probabilities) + (1 - batch_y) * tf.math.log(1 - probabilities))
                    val_loss += loss

                val_loss /= len(val_dataset)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1

                if early_stopping_count >= self.early_stopping_patience:
                    print(f"Class {class_label}: Early stopping at epoch {epoch}")
                    break

                # print(f"Class {class_label}: Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            self.coefficients.append(coefficients_class)
            
    def predict_proba(self, X):
        X_scaled = self._scale_features(X)
        X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
        X_scaled = tf.constant(X_scaled, dtype=tf.float32)

        logits = [tf.matmul(X_scaled, coeff).numpy() for coeff in self.coefficients]
        probabilities = [self.softmax(logit) for logit in logits]
        return np.array(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        y_binary = np.array([np.array([1.0 if i == label else 0.0 for i in range(len(np.unique(y)))]) for label in y])
        y_pred_proba = self.predict_proba(X)
        loss = -np.mean(y_binary * np.log(y_pred_proba) + (1 - y_binary) * np.log(1 - y_pred_proba))

        return accuracy, loss

    def evaluate(self, X, y):
        y_pred = self.predict_proba(X)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss
