import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class BernoulliNaiveBayes:
    def __init__(self, smoothing=1.0):
        self.class_priors = None
        self.feature_probabilities = None
        self.smoothing = smoothing

    def _calculate_class_priors(self, labels):
        unique_labels, class_counts = np.unique(labels, return_counts=True)
        return class_counts / len(labels)

    def _calculate_feature_probabilities(self, features, labels):
        num_features = features.shape[1]
        num_classes = len(np.unique(labels))
        feature_probabilities = np.zeros((num_classes, num_features))

        for c in range(num_classes):
            class_mask = labels == c
            class_feature_count = np.sum(class_mask)
            class_features = features[class_mask]

            feature_probabilities[c] = (np.sum(class_features, axis=0) + self.smoothing) / (class_feature_count + 2 * self.smoothing)

        return feature_probabilities

    def fit(self, features, labels, epochs=100, learning_rate=0.01, verbose=True,smoothing_factor=0.9):
        # Convert to binary (0 or 1) features
        features = np.where(features > 0, 1, 0)

        # Set the number of classes based on the unique labels
        self.num_classes = len(np.unique(labels))

        # Check if all classes are present in the training data
        unique_classes = np.unique(labels)
        missing_classes = [c for c in range(self.num_classes) if c not in unique_classes]
        if missing_classes:
            raise ValueError(f"Classes {missing_classes} are missing from the training data.")

        self.num_features = features.shape[1]  # Initialize num_features
        self.class_priors = self._calculate_class_priors(labels)

        # Convert the feature probabilities to a TensorFlow variable
        self.feature_probabilities = tf.Variable(self._calculate_feature_probabilities(features, labels), dtype=tf.float32)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                expanded_probs = tf.expand_dims(self.feature_probabilities, axis=0)

                # Repeat expanded_probs along the first axis to match the number of samples in features
                repeated_probs = tf.repeat(expanded_probs, features.shape[0], axis=0)

                # Cast features to float32
                features_float32 = tf.cast(features, dtype=tf.float32)

                # Expand dimensions for broadcasting
                expanded_features = tf.expand_dims(features_float32, axis=1)

                # Calculate log likelihoods instead of probabilities
                log_likelihoods = expanded_features * tf.math.log(repeated_probs) + (1 - expanded_features) * tf.math.log(1 - repeated_probs)
                class_log_likelihoods = tf.reduce_sum(log_likelihoods, axis=2)  # Sum along the num_features axis
                class_priors_float32 = tf.cast(self.class_priors, dtype=tf.float32)
                class_posteriors = class_log_likelihoods + tf.math.log(class_priors_float32)
                class_posteriors -= tf.reduce_logsumexp(class_posteriors, axis=1, keepdims=True)

            gradients = tape.gradient(class_posteriors, self.feature_probabilities)

            # Apply momentum-like update with smoothing factor
            self.feature_probabilities.assign_add(learning_rate * gradients * (1 - smoothing_factor))
            self.feature_probabilities.assign(self.feature_probabilities * smoothing_factor)

            if verbose:
                accuracy = self.evaluate(features, labels)
                print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {accuracy:.4f}")

        self.feature_probabilities = tf.math.exp(self.feature_probabilities)  # Convert back to probabilities

    def predict(self, features):
        features = np.where(features > 0, 1, 0)
        features_tensor = tf.constant(features, dtype=tf.float32)

        num_samples = features.shape[0]
        predictions = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            sample_probs = tf.math.log(self.class_priors)
            for c in range(self.num_classes):
                class_feature_probs = self.feature_probabilities[c]
                likelihood = tfp.distributions.Bernoulli(probs=class_feature_probs)
                feature_likelihoods = likelihood.log_prob(features_tensor[i])
                sample_probs += tf.reduce_sum(tf.cast(feature_likelihoods, tf.float64))  # Explicitly cast to double

            predictions[i] = tf.argmax(sample_probs)

        return predictions

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        accuracy = np.mean(predictions == labels)
        return accuracy