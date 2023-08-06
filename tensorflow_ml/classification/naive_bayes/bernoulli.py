import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class BernoulliNaiveBayes:
    """
    Defines a Naive Bayes classifier that can be used to fit,
    predict, and evaluate binary features.
    
    :param smoothing: The `smoothing` parameter is used to prevent zero
    probabilities when calculating the class priors and feature probabilities
    in the Naive Bayes classifier. It is added to the counts of class
    occurrences and feature occurrences to ensure that even if a class or
    feature has not been observed in the training data, it
    """
    def __init__(self, smoothing=1.0):
        self.class_priors = None
        self.feature_probabilities = None
        self.smoothing = smoothing

    def _calculate_class_priors(self, labels):
        """
        The function calculates the class priors for a given set of labels.
        
        :param labels: The "labels" parameter is a list or array containing the class labels for a set of
        data points. Each element in the list represents the class label for a single data point
        :return: the calculated class priors.
        """
        unique_labels, class_counts = np.unique(labels, return_counts=True)
        return (class_counts + self.smoothing) / (len(labels) + self.smoothing * len(unique_labels))

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

            # Update the shape of feature_probabilities
            feature_probabilities[c] = (np.sum(class_features, axis=0) + self.smoothing) / (class_feature_count + 2 * self.smoothing)

        return feature_probabilities

    def fit(self, features, labels):
        """
        The `fit` function converts features to binary, sets the number of classes, checks if all classes
        are present, calculates class priors, and calculates feature probabilities.
        
        :param features: The "features" parameter is a numpy array that represents the input features of the
        training data. Each row of the array corresponds to a single data point, and each column represents
        a specific feature
        :param labels: The "labels" parameter is a numpy array or list that contains the class labels for
        each data point in the training set. Each label represents the class or category to which the
        corresponding data point belongs
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

        # Convert the feature probabilities to a TensorFlow constant
        self.feature_probabilities = tf.constant(self._calculate_feature_probabilities(features, labels), dtype=tf.float32)

    def predict(self, features):
        """
        The `predict` function takes in a set of features, converts them to binary values, and uses a
        trained model to make predictions on the given features.
        
        :param features: The "features" parameter is a numpy array that represents the input features for
        which we want to make predictions. It is assumed that the features are already preprocessed and
        converted to binary values (0 or 1)
        :return: an array of predictions.
        """
        # Convert to binary (0 or 1) features
        features = np.where(features > 0, 1, 0)

        # Convert to a TensorFlow tensor
        features_tensor = tf.constant(features, dtype=tf.float32)

        num_samples = features.shape[0]
        predictions = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            sample_probs = tf.math.log(self.class_priors)
            for c in range(self.num_classes):
                class_feature_probs = self.feature_probabilities[c]
                # Use TensorFlow Probability Bernoulli distribution to compute the likelihood
                likelihood = tfp.distributions.Bernoulli(probs=class_feature_probs)
                feature_likelihoods = likelihood.log_prob(features_tensor[i])
                # Reduce feature_likelihoods along the feature axis (summing over features)
                feature_likelihoods_sum = tf.reduce_sum(feature_likelihoods, axis=-1)
                # Use broadcasting to add the feature_likelihoods_sum to sample_probs
                sample_probs += tf.cast(feature_likelihoods_sum, tf.float64)  # Explicitly cast to double

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