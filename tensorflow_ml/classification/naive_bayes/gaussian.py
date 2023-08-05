import tensorflow as tf
import numpy as np

class GaussianNaiveBayes:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.class_priors = None
        self.feature_params = None
        self.num_classes = None
        self.num_features = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)

    def _calculate_class_priors(self, labels):
        _, class_counts = np.unique(labels, return_counts=True)
        return class_counts / len(labels)

    def _calculate_feature_params(self, features, labels):
        num_features = features.shape[1]
        num_classes = len(np.unique(labels))
        feature_params = np.zeros((num_classes, num_features, 2))  # (mean, variance)

        for c in range(num_classes):
            class_features = features[labels == c]
            feature_params[c, :, 0] = np.mean(class_features, axis=0)
            feature_params[c, :, 1] = np.var(class_features, axis=0)

        return feature_params

    def _gaussian_likelihood(self, x, mean, variance):
        return tf.exp(-0.5 * tf.square((x - mean) / variance)) / (tf.sqrt(2 * np.pi * variance))

    def fit(self, features, labels, epochs=10):
        self.num_classes = len(np.unique(labels))
        self.num_features = features.shape[1]
        self.class_priors = self._calculate_class_priors(labels)
        self.feature_params = self._calculate_feature_params(features, labels)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                class_priors_tf = tf.Variable(self.class_priors, trainable=True)
                feature_params_tf = tf.Variable(self.feature_params, trainable=True)

                log_likelihoods = []
                for i in range(self.num_features):
                    log_likelihoods.append(tf.math.log(self._gaussian_likelihood(features[:, i, None], feature_params_tf[:, i, 0], feature_params_tf[:, i, 1])))

                log_likelihoods = tf.reduce_sum(log_likelihoods, axis=0) + tf.math.log(class_priors_tf)
                loss = -tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=log_likelihoods))

            grads = tape.gradient(loss, [class_priors_tf, feature_params_tf])
            self.optimizer.apply_gradients(zip(grads, [self.class_priors, self.feature_params]))

    def predict(self, features):
        num_samples = features.shape[0]
        predictions = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            sample_probs = np.log(self.class_priors)
            for j in range(self.num_features):
                feature_val = features[i, j]
                for c in range(self.num_classes):
                    prob = np.log(self._bernoulli_likelihood(feature_val, self.feature_probabilities[c, j]))
                    sample_probs[c] += prob

            predictions[i] = np.argmax(sample_probs)

        return predictions

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        accuracy = np.mean(predictions == labels)
        return accuracy