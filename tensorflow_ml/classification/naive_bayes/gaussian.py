import tensorflow as tf
import numpy as np

class GaussianNaiveBayes:
    """
    Defines a Gaussian Naive Bayes classifier in Python using
    TensorFlow.
    
    Gaussian Naive Bayes is a classification algorithm that assumes that the features follow a normal
    distribution. It is a variant of the Naive Bayes algorithm that is used for classification tasks. It is
    called "naive" because it assumes that the features are independent of each other. This assumption is
    called "naive" because it is rarely true in real-world applications. However, despite this assumption,
    Gaussian Naive Bayes performs surprisingly well in many cases.

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
        _, class_counts = np.unique(labels, return_counts=True)
        return class_counts / len(labels)

    def _calculate_feature_params(self, features, labels):
        """
        The function `_calculate_feature_params` calculates the mean and variance of each feature for each
        class in a given dataset.
        
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
            feature_params : np.ndarray
                The feature parameters are the mean and variance of each feature for each class in a given dataset.
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
        
        Parameters
        ----------
            x : np.ndarray
                The `x` parameter is a np.ndarray that represents the input features for training the model. It
                has a shape of `(num_samples, num_features)`, where `num_samples` is the number of training
                samples and `num_features` is the number of features for each sample
                
            mean : np.ndarray
                The `mean` parameter is a np.ndarray that represents the mean of the Gaussian distribution for
                each feature
                
            variance : np.ndarray
                The `variance` parameter is a np.ndarray that represents the variance of the Gaussian
                distribution for each feature
        
        Returns
        -------
            The Gaussian likelihood of a given value.
        """
        return tf.exp(-0.5 * tf.square((x - mean) / variance)) / (tf.sqrt(2 * np.pi * variance))

    def fit(self, features, labels, epochs=10):
        """
        The `fit` function trains a Gaussian Naive Bayes classifier using TensorFlow by calculating class
        priors and feature parameters, and optimizing them using gradient descent.
        
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
                An epoch is one iteration over the entire training dataset. For example, if the training dataset
                has 1000 samples and the batch size is 100, then it will take 10 iterations to complete 1 epoch.
                The default value is 10.
            
        """
        self.num_classes = len(np.unique(labels))
        self.num_features = features.shape[1]
        self.class_priors = self._calculate_class_priors(labels)
        self.feature_params = self._calculate_feature_params(features, labels)

        # Convert np.ndarrays to TensorFlow variables
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
        
        Parameters
        ----------
            features : np.ndarray
                The `features` parameter is a np.ndarray that represents the input features for training the
                model. It has a shape of `(num_samples, num_features)`, where `num_samples` is the number of
                training samples and `num_features` is the number of features for each sample
            
        Returns
        -------
            predictions : np.ndarray
                The `predictions` parameter is a np.ndarray that contains the predicted class labels for each
                data point in the `features` array. Each element in the `predictions` array corresponds to the
                predicted class label of the corresponding data point in the `features` array.
            
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
                The `accuracy` parameter is a float that represents the accuracy of the predictions made by the
                model.
            
        """
        predictions = self.predict(features)
        accuracy = np.mean(predictions == labels)
        return accuracy