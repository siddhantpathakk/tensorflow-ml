import tensorflow as tf
import numpy as np
import pandas as pd

class BernoulliNaiveBayes:
    def __init__(self, learning_rate=0.01, momentum=0.9, batch_size=32, smoothing=1.0):
        self.class_priors = None
        self.feature_probabilities = None
        self.num_classes = None
        self.num_features = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.smoothing = smoothing

    def _calculate_class_priors(self, labels):
        _, class_counts = tf.unique(labels)
        return (class_counts + 1) / (tf.size(labels) + self.num_classes)  # Laplace smoothing

    def _calculate_feature_probabilities(self, features, labels):
        num_features = tf.shape(features)[1]
        num_classes = tf.size(tf.unique(labels).y)
        feature_probabilities_list = []

        for c in range(num_classes):
            class_features = tf.boolean_mask(features, labels == c)
            class_feature_sum = tf.reduce_sum(class_features, axis=0)
            class_feature_count = tf.size(class_features, out_type=tf.float32)

            feature_prob_update = (class_feature_sum + self.smoothing) / (class_feature_count + 2 * self.smoothing)
            feature_probabilities_list.append(feature_prob_update)

        feature_probabilities = tf.convert_to_tensor(feature_probabilities_list, dtype=tf.float32)
        return feature_probabilities

    def _bernoulli_likelihood(self, x, p):
        p = tf.clip_by_value(p, 1e-8, 1 - 1e-8)  # Avoid division by zero or one

        # Broadcast x to have the same shape as p
        x_broadcasted = tf.broadcast_to(tf.expand_dims(x, axis=1), tf.shape(p))

        likelihood = tf.where(x_broadcasted == 1, p, 1 - p)
        return tf.reduce_prod(likelihood, axis=-1)

    def fit(self, features, labels, epochs=1000, patience=10, validation_data=None):
        # Convert to binary (0 or 1) features
        features = tf.cast(features > 0, tf.float32)

        # Get the indices of non-zero variance features
        non_zero_var_features = tf.math.reduce_std(features, axis=0) > 0
        non_zero_var_feature_indices = tf.where(non_zero_var_features).numpy().flatten()

        # Get the non-zero variance feature names
        non_zero_var_feature_names = [str(i) for i in non_zero_var_feature_indices]

        # Filter the features using non-zero variance feature indices
        features = tf.gather(features, non_zero_var_feature_indices, axis=1)

        # Set the number of classes based on the unique labels
        self.num_classes = tf.size(tf.unique(labels).y)

        # Check if all classes are present in the training data
        unique_classes = tf.unique(labels).y
        missing_classes = [c for c in range(self.num_classes) if c not in unique_classes]
        if missing_classes:
            raise ValueError(f"Classes {missing_classes} are missing from the training data.")

        self.num_features = tf.shape(features)[1]
        self.class_priors = self._calculate_class_priors(labels)
        self.feature_probabilities = self._calculate_feature_probabilities(features, labels)

        # Create training dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=len(labels)).batch(self.batch_size)

        class_priors_tf = tf.Variable(self.class_priors, trainable=True)
        feature_probabilities_tf = tf.constant(self.feature_probabilities)

        # Define the optimizer with momentum
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)

        # Define the early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

        @tf.function
        def train_step(batch_features, batch_labels):
            batch_size = tf.shape(batch_features)[0]
            num_classes = tf.shape(feature_probabilities_tf)[0]

            with tf.GradientTape() as tape:
                # Compute the log-likelihoods for each sample and class
                feature_probabilities_tiled = tf.tile(tf.expand_dims(feature_probabilities_tf, 0), [batch_size, 1, 1])
                log_likelihoods = tf.math.log(self._bernoulli_likelihood(batch_features, feature_probabilities_tiled))

                # Compute the class priors for each sample and reshape for broadcasting
                class_priors_broadcasted = tf.tile(tf.expand_dims(class_priors_tf, 1), [1, num_classes])
                class_priors_broadcasted = tf.cast(class_priors_broadcasted, tf.float64)

                # Add the class priors to the log-likelihoods element-wise
                log_likelihoods = tf.cast(log_likelihoods, tf.float64)
                log_likelihoods += class_priors_broadcasted

                # Compute the loss
                loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=log_likelihoods), axis=1))

            # Compute gradients and update model parameters
            grads = tape.gradient(loss, [class_priors_tf, feature_probabilities_tf])
            optimizer.apply_gradients(zip(grads, [class_priors_tf, feature_probabilities_tf]))

            return loss

        # Validation dataset
        if validation_data is not None:
            val_features, val_labels = validation_data
            val_features = tf.cast(val_features > 0, tf.float32)
            val_features = tf.gather(val_features, non_zero_var_feature_indices, axis=1)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
            val_dataset = val_dataset.batch(self.batch_size)

        best_val_accuracy = 0.0
        patience_count = 0

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_features, batch_labels in dataset:
                batch_loss = train_step(batch_features, batch_labels)
                total_loss += batch_loss
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Validation
            if validation_data is not None:
                val_loss = 0.0
                val_batches = 0
                val_correct = 0
                for val_batch_features, val_batch_labels in val_dataset:
                    val_batch_features = tf.cast(val_batch_features > 0, tf.float32)
                    val_batch_features = val_batch_features[non_zero_var_feature_names]

                    val_batch_loss = train_step(val_batch_features, val_batch_labels)
                    val_loss += val_batch_loss
                    val_batches += 1

                    predictions = self.predict(val_batch_features)
                    val_correct += tf.reduce_sum(tf.cast(predictions == val_batch_labels, tf.float32))

                avg_val_loss = val_loss / val_batches
                val_accuracy = val_correct / tf.size(val_features, out_type=tf.float32)
                print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

                # Check for early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print("Early stopping: Validation accuracy did not improve.")
                        break

    def predict(self, features):
        features = tf.cast(features > 0, tf.float32)
        num_samples = tf.shape(features)[0]
        predictions = tf.zeros(num_samples, dtype=tf.int64)

        for i in range(num_samples):
            sample_probs = tf.math.log(self.class_priors)
            for j in range(self.num_features):
                feature_val = features[i, j]
                for c in range(self.num_classes):
                    prob = tf.math.log(self._bernoulli_likelihood(feature_val, self.feature_probabilities[c, j]))
                    sample_probs[c] += prob

            predictions[i] = tf.argmax(sample_probs)

        return predictions.numpy()

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        accuracy = tf.reduce_mean(tf.cast(predictions == labels, tf.float32))
        return accuracy