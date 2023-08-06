import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class BernoulliNaiveBayes:
    """
    Defines a class for a probabilistic classifier that can be
    trained on features and labels, and used to make predictions and evaluate
    the accuracy of the predictions.
    
    :param smoothing: The `smoothing` parameter is a hyperparameter that
    controls the amount of smoothing applied to the feature probabilities
    during training. It is used to prevent zero probabilities and handle
    unseen features. A higher smoothing value will result in a smaller impact
    of the observed data on the feature probabilities, while a lower smoothing
    value
    """
    def __init__(self, smoothing=1.0):
        self.class_priors = None
        self.feature_probabilities = None
        self.smoothing = smoothing

    def _calculate_class_priors(self, labels):
        """
        The function calculates the class priors by counting the occurrences of each unique label and
        dividing by the total number of labels.
        
        :param labels: The `labels` parameter is a list or array containing the class labels for a set of
        data points. Each element in the `labels` list corresponds to the class label of a data point
        :return: the class priors, which are calculated by dividing the count of each class by the total
        number of labels.
        """
        unique_labels, class_counts = np.unique(labels, return_counts=True)
        return class_counts / len(labels)

    def _calculate_feature_probabilities(self, features, labels):
        """
        The function calculates the probabilities of each feature given each class in a classification
        problem.
        
        :param features: A numpy array containing the features of the dataset. Each row represents a data
        point and each column represents a feature
        :param labels: The `labels` parameter is a numpy array that contains the class labels for each data
        point. Each element in the array represents the class label for the corresponding data point
        :return: a numpy array of shape (num_classes, num_features) containing the calculated feature
        probabilities.
        """
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
        """
        The `fit` function trains a probabilistic classifier using the given features and labels, optimizing
        the feature probabilities using gradient descent with momentum-like update.
        
        :param features: The features are the input data used for training the model. It could be a numpy
        array or a pandas DataFrame with shape (num_samples, num_features)
        :param labels: The `labels` parameter is a numpy array or list containing the class labels for each
        sample in the training data
        :param epochs: The number of times the model will iterate over the entire training dataset during
        training. Each iteration is called an epoch. By default, it is set to 100, defaults to 100
        (optional)
        :param learning_rate: The learning rate determines the step size at each iteration of the
        optimization algorithm. It controls how much the model's parameters are updated based on the
        gradients of the loss function. A higher learning rate can lead to faster convergence, but it may
        also cause the optimization process to overshoot the optimal solution. On
        :param verbose: The `verbose` parameter is a boolean flag that determines whether or not to print
        the training accuracy at each epoch. If `verbose` is set to `True`, the training accuracy will be
        printed. If `verbose` is set to `False`, the training accuracy will not be printed, defaults to True
        (optional)
        :param smoothing_factor: The smoothing factor is a hyperparameter that controls the amount of
        momentum-like update applied to the feature probabilities during training. It is used to prevent
        drastic changes in the feature probabilities from one iteration to the next. A higher smoothing
        factor will result in a slower update, while a lower smoothing factor will allow for
        """
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
        """
        The `predict` function takes in a set of features, converts them to binary values, calculates the
        log probabilities for each class, and returns the predicted class for each sample.
        
        :param features: The `features` parameter is a numpy array containing the input features for which
        you want to make predictions
        :return: an array of predictions.
        """
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
        """
        The evaluate function calculates the accuracy of the predictions made by the model.
        
        :param features: The features parameter is a numpy array or a list of input data that you want to
        evaluate. It represents the independent variables or attributes of your dataset. Each row in the
        array or list corresponds to a single data point, and each column represents a different feature or
        attribute
        :param labels: The "labels" parameter refers to the true labels or target values of the data. These
        are the values that you are trying to predict or classify using the features
        :return: The accuracy of the predictions.
        """
        predictions = self.predict(features)
        accuracy = np.mean(predictions == labels)
        return accuracy