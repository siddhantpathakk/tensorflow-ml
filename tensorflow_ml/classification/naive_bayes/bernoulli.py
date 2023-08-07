import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class BernoulliNaiveBayes:
    """
    Defines a class for a probabilistic classifier that can be
    trained on features and labels, and used to make predictions and evaluate
    the accuracy of the predictions.
    
    Bernoulli Naive Bayes is a probabilistic classifier that assumes that the features are binary (0 or 1).
    It is based on Bayes' theorem, which states that the probability of a hypothesis (class) given the data
    (features) is equal to the probability of the data given the hypothesis multiplied by the probability of
    the hypothesis divided by the probability of the data. In other words, it is the posterior probability of
    a hypothesis given the data is equal to the likelihood of the data given the hypothesis multiplied by the
    prior probability of the hypothesis divided by the marginal likelihood of the data.
    
    """
    def __init__(self, smoothing=1.0):
        self.class_priors = None
        self.feature_probabilities = None
        self.smoothing = smoothing

    def _calculate_class_priors(self, labels):
        """
        The function calculates the class priors by counting the occurrences of each unique label and
        dividing by the total number of labels.
        
        Parameters
        ----------
            labels : np.ndarray
                The "labels" parameter is a np.ndarray that contains the class labels for each data point in the
                "features" array. Each element in the "labels" array corresponds to the class label of the
                corresponding data point in the "features" array
        
        Returns
        -------
            class_priors : np.ndarray
                The class priors are the probabilities of each class in the dataset. They are calculated by
                counting the occurrences of each class label and dividing by the total number of labels.
        
        """
        unique_labels, class_counts = np.unique(labels, return_counts=True)
        return class_counts / len(labels)

    def _calculate_feature_probabilities(self, features, labels):
        """
        The function calculates the probabilities of each feature given each class in a classification
        problem.
        Parameters
        ----------
            features : np.ndarray
                The `features` parameter is a np.ndarray that represents the input features for training the
                model. It has a shape of `(num_samples, num_features)`, where `num_samples` is the number of
                training samples and `num_features` is the number of features for each sample
                
            labels : np.ndarray
                The "labels" parameter is a np.ndarray that contains the class labels for each data point in the
                "features" array. Each element in the "labels" array corresponds to the class label of the
            corresponding data point in the "features" array
            
        Returns
        -------
            feature_probabilities : np.ndarray
                The feature probabilities are the probabilities of each feature given each class in a
                classification problem.
            
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
        
        Parameters
        ----------
            features : np.ndarray
                The `features` parameter is a np.ndarray that represents the input features for training the
                model. It has a shape of `(num_samples, num_features)`, where `num_samples` is the number of
                training samples and `num_features` is the number of features for each sample
                
            labels : np.ndarray
                The "labels" parameter is a np.ndarray that contains the class labels for each data point in the
                "features" array. Each element in the "labels" array corresponds to the class label of the
                corresponding data point in the "features" array
                
            epochs : int
                The `epochs` parameter is an integer that represents the number of epochs to train the model for.
                An epoch is one iteration over the entire training dataset.
                
            learning_rate : float
                The `learning_rate` parameter is a float that controls the size of the gradient descent step.
                
            verbose : bool
                The `verbose` parameter is a boolean that controls whether or not to print the training accuracy
                for each epoch.
                
            smoothing_factor : float
                The `smoothing_factor` parameter is a float that controls the amount of smoothing to apply to the
                feature probabilities. It is used to prevent the probabilities from becoming too extreme.
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
        
        Parameters
        ----------
            features : np.ndarray
                The `features` parameter is a np.ndarray that represents the input features for training the
                model. It has a shape of `(num_samples, num_features)`, where `num_samples` is the number of
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
        
        Parameters
        ----------
            features : np.ndarray
                The `features` parameter is a np.ndarray that represents the input features for training the
                model. It has a shape of `(num_samples, num_features)`, where `num_samples` is the number of
                training samples and `num_features` is the number of features for each sample
                
            labels : np.ndarray
                The "labels" parameter is a np.ndarray that contains the class labels for each data point in the
                "features" array. Each element in the "labels" array corresponds to the class label of the
                corresponding data point in the "features" array
            
        Returns
        -------
            accuracy : float
                The `accuracy` parameter is a float that represents the accuracy of the model on the given
                features and labels.
        """
        predictions = self.predict(features)
        accuracy = np.mean(predictions == labels)
        return accuracy