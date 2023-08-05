import numpy as np
import tensorflow as tf

class KNNClassifier:
    def __init__(self, k=3, verbose=False):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.verbose = verbose

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train.astype(int)  # Ensure labels are integers

        if self.verbose:
            print("Training KNN classifier...")

    def predict(self, X_test):
        y_pred = []
        if self.verbose:
            print("Predicting...")
        for i, x in enumerate(X_test):
            distances = tf.reduce_sum(tf.square(self.X_train - x), axis=1)
            k_indices = tf.argsort(distances)[:self.k]
            k_nearest_labels = tf.gather(self.y_train, k_indices)
            
            # Flatten k_nearest_labels to 1D tensor
            k_nearest_labels = tf.reshape(k_nearest_labels, shape=[-1])
            
            with tf.device('/CPU:0'):  # Perform uniqueness and counting on CPU
                unique, _, counts = tf.unique_with_counts(k_nearest_labels)
            predicted_label = unique[tf.argmax(counts)]
            y_pred.append(predicted_label.numpy())
            if self.verbose and i % 100 == 0:
                print(f"Predicted {i}/{len(X_test)} samples")
        return np.array(y_pred)

    def evaluate(self, X_test, y_test):
        if self.verbose:
            print("Evaluating...")
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        if self.verbose:
            print("Evaluation complete.")
        return accuracy