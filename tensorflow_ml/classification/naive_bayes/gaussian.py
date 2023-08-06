import tensorflow as tf
import numpy as np

class GaussianNaiveBayes:
    """
    Defines a Gaussian Naive Bayes classifier in Python using
    TensorFlow.
    
    :param learning_rate: The learning_rate parameter determines the step size
    at each iteration of the optimization algorithm. It controls how quickly
    the model learns from the training data. A higher learning rate may result
    in faster convergence, but it can also cause the model to overshoot the
    optimal solution. On the other hand, a lower learning
    :param momentum: The momentum parameter is a hyperparameter that controls
    the amount of momentum applied during gradient descent optimization. It
    affects how quickly the optimizer updates the model's parameters based on
    the gradients of the loss function. A higher momentum value means that the
    optimizer will take into account a larger portion of the previous gradients
    when updating
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.class_priors = None
        self.feature_params = None
        self.num_classes = None
        self.num_features = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)

    def _calculate_class_priors(self, labels):
        """
        The function calculates the class priors by counting the occurrences of each class label and
        dividing by the total number of labels.
        
        :param labels: The `labels` parameter is a list or array containing the class labels for a dataset.
        Each element in the `labels` list represents the class label for a corresponding data point in the
        dataset
        :return: the class priors, which are calculated by dividing the count of each class by the total
        number of labels.
        """
        _, class_counts = np.unique(labels, return_counts=True)
        return class_counts / len(labels)

    def _calculate_feature_params(self, features, labels):
        """
        The function `_calculate_feature_params` calculates the mean and variance of each feature for each
        class in a given dataset.
        
        :param features: The "features" parameter is a numpy array that represents the input features of a
        dataset. It has a shape of (n_samples, n_features), where n_samples is the number of samples in the
        dataset and n_features is the number of features for each sample
        :param labels: The "labels" parameter is a numpy array that contains the class labels for each data
        point in the "features" array. Each element in the "labels" array corresponds to the class label of
        the corresponding data point in the "features" array
        :return: a numpy array `feature_params` of shape `(num_classes, num_features, 2)`. Each element in
        the array represents the mean and variance of a feature for a particular class.
        """
        num_features = features.shape[1]
        num_classes = len(np.unique(labels))
        feature_params = np.zeros((num_classes, num_features, 2))  # (mean, variance)

        for c in range(num_classes):
            class_features = features[labels == c]
            feature_params[c, :, 0] = np.mean(class_features, axis=0)
            feature_params[c, :, 1] = np.var(class_features, axis=0)

        return feature_params

    def _gaussian_likelihood(self, x, mean, variance):
        """
        The function calculates the Gaussian likelihood of a given value.
        
        :param x: The input value for which we want to calculate the likelihood
        :param mean: The mean parameter represents the average value of the Gaussian distribution. It
        determines the center of the distribution
        :param variance: The variance parameter represents the variance of the Gaussian distribution. It
        determines the spread or dispersion of the distribution. A higher variance value indicates a wider
        distribution, while a lower variance value indicates a narrower distribution
        :return: the Gaussian likelihood of a given input value `x` with respect to a given mean and
        variance.
        """
        return tf.exp(-0.5 * tf.square((x - mean) / variance)) / (tf.sqrt(2 * np.pi * variance))

    def fit(self, features, labels, epochs=10):
        """
        The `fit` function trains a Gaussian Naive Bayes classifier using TensorFlow by calculating class
        priors and feature parameters, and optimizing them using gradient descent.
        
        :param features: The `features` parameter is a NumPy array that represents the input features for
        training the model. It has a shape of `(num_samples, num_features)`, where `num_samples` is the
        number of training samples and `num_features` is the number of features for each sample
        :param labels: The `labels` parameter is a NumPy array or TensorFlow tensor containing the target
        labels for the training data. Each label represents the class or category to which a corresponding
        data point belongs
        :param epochs: The parameter "epochs" represents the number of times the model will iterate over the
        entire dataset during training. Each iteration is called an epoch, defaults to 10 (optional)
        """
        self.num_classes = len(np.unique(labels))
        self.num_features = features.shape[1]
        self.class_priors = self._calculate_class_priors(labels)
        self.feature_params = self._calculate_feature_params(features, labels)

        # Convert NumPy arrays to TensorFlow variables
        self.class_priors = tf.Variable(self.class_priors, trainable=True)
        self.feature_params = tf.Variable(self.feature_params, trainable=True)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                log_likelihoods = []
                for i in range(self.num_features):
                    log_likelihoods.append(tf.math.log(self._gaussian_likelihood(features[:, i, None], self.feature_params[:, i, 0], self.feature_params[:, i, 1])))

                log_likelihoods = tf.reduce_sum(log_likelihoods, axis=0) + tf.math.log(self.class_priors)
                loss = -tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=log_likelihoods))

            grads = tape.gradient(loss, [self.class_priors, self.feature_params])
            self.optimizer.apply_gradients(zip(grads, [self.class_priors, self.feature_params]))

    def predict(self, features):
        """
        The `predict` function takes in a set of features and returns the predicted class labels using a
        Gaussian Naive Bayes classifier.
        
        :param features: The `features` parameter is a numpy array that represents the features of the
        samples you want to make predictions on. Each row of the array represents a sample, and each column
        represents a feature
        :return: an array of predictions.
        """
        num_samples = features.shape[0]
        predictions = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            sample_probs = np.log(self.class_priors)
            for j in range(self.num_features):
                feature_val = features[i, j]
                for c in range(self.num_classes):
                    prob = np.log(self._gaussian_likelihood(feature_val, self.feature_params[c, j, 0], self.feature_params[c, j, 1]))
                    sample_probs[c] += prob

            predictions[i] = np.argmax(sample_probs)

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