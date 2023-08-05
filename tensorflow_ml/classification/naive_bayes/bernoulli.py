import tensorflow as tf
import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.class_priors = None
        self.feature_probabilities = None
        self.num_classes = None
        self.num_features = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)

    def _calculate_class_priors(self, labels):
        _, class_counts = np.unique(labels, return_counts=True)
        return class_counts / len(labels)

    def _calculate_feature_probabilities(self, features, labels):
        num_features = features.shape[1]
        num_classes = len(np.unique(labels))
        feature_probabilities = np.zeros((num_classes, num_features))

        for c in range(num_classes):
            class_features = features[labels == c]
            feature_probabilities[c] = (class_features.sum(axis=0) + 1) / (len(class_features) + 2)

        return feature_probabilities

    def _bernoulli_likelihood(self, x, p):
        return tf.where(x == 1, p, 1 - p)

    def fit(self, features, labels, epochs=10):
        self.num_classes = len(np.unique(labels))
        self.num_features = features.shape[1]
        self.class_priors = self._calculate_class_priors(labels)
        self.feature_probabilities = self._calculate_feature_probabilities(features, labels)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                class_priors_tf = tf.Variable(self.class_priors, trainable=True)
                feature_probabilities_tf = tf.Variable(self.feature_probabilities, trainable=True)

                log_likelihoods = []
                for i in range(self.num_features):
                    log_likelihoods.append(tf.math.log(self._bernoulli_likelihood(features[:, i, None], feature_probabilities_tf[:, i])))

                log_likelihoods = tf.reduce_sum(log_likelihoods, axis=0) + tf.math.log(class_priors_tf)
                loss = -tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=log_likelihoods))

            grads = tape.gradient(loss, [class_priors_tf, feature_probabilities_tf])
            self.optimizer.apply_gradients(zip(grads, [self.class_priors, self.feature_probabilities]))

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