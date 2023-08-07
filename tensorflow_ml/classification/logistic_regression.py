import numpy as np
import tensorflow as tf

class LogisticRegression:
    """
    Defines a class that implements a logistic regression model with various parameters
    and methods for training, testing, and evaluating the model.
    
    Logistic regression is a classification algorithm used to predict the probability of a categorical
    dependent variable. In logistic regression, the dependent variable is a binary variable that contains data
    coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model
    predicts P(Y=1) as a function of X.

    """
    def __init__(self):    
        self.learning_rate = None
        self.num_epochs = None
        self.batch_size = None
        self.reg_strength = None
        self.coefficients = None
        self.verbose = False
        self.regularization = None
        self.early_stopping_patience = None
        self.use_learning_rate_scheduling = True
        self.tolerance = 1e-4  # Set a tolerance value for early stopping

    def set_params(self, params: dict):
        """
        The function sets the parameters for a machine learning model, including learning rate, number of
        epochs, regularization strength, batch size, early stopping patience, regularization method, and
        tolerance.
        
        Parameters
        ----------
            params : dict
                A dictionary containing the values of the learning_rate, num_epochs, reg_strength, batch_size,
                early_stopping_patience, regularization, and tolerance attributes.
        """
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.reg_strength = params['reg_strength']
        self.batch_size = params['batch_size']
        self.early_stopping_patience = params['early_stopping_patience']
        self.regularization = params["regularization"]
        self.tolerance = float(params.get("tolerance", 1e-4))  # Convert tolerance to a float
        if self.use_learning_rate_scheduling:
            self.learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True
            )

    def get_params(self) -> dict:
        """
        The function `get_params` returns a dictionary containing the values of various parameters.
        
        Returns
        -------
            dict
                A dictionary containing the values of the learning_rate, num_epochs, reg_strength, batch_size,
        """
        return {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "reg_strength": self.reg_strength,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience
        }

    def _scale_features(self, X):
        """
        The function `_scale_features` returns the input `X` without any scaling applied to it.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X represents the input features or data that you want to scale
            
        Returns
        -------
            np.ndarray
                the input array `X` without any scaling or normalization applied to it.
        """
        # mean = np.mean(X, axis=0)
        # std = np.std(X, axis=0)
        # scaled_X = (X - mean) / std
        # return scaled_X
        return X

    def sigmoid(self, z):
        """
        The sigmoid function returns the value of 1 divided by 1 plus the exponential of the negative input
        value.
        
        Sigmoid function is a mathematical function that takes any real value and maps it to a value between
        0 and 1. It is a non-linear function used for binary classification tasks.
        
        Parameters
        ----------
            z : tf.Tensor
                The parameter "z" is a tensor representing the input to the sigmoid function. It can be a 1D or
                2D tensor
        Returns
        -------
            tf.Tensor
                The sigmoid function is being returned.
        """
        return 1 / (1 + tf.exp(-z))

    def softmax(self, z):
        """
        The softmax function takes in a vector of values and returns a vector of probabilities that sum up
        to 1.
        
        Softmax function is a mathematical function that takes a vector of real numbers and normalizes it into
        a probability distribution consisting of probabilities proportional to the exponentials of the input
        numbers. In other words, the softmax function converts a vector of numbers into a vector of
        probabilities that sum up to 1.
        
        Parameters
        ----------
            z : tf.Tensor
                The parameter "z" is a tensor representing the input to the softmax function. It can be a 1D or
                2D tensor
            
        Returns
        -------
            tf.Tensor
                The softmax function is being returned.
        """
        return tf.nn.softmax(z)

    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        """
        The function `train_test_split` splits the input data `X` and target variable `y` into training and
        testing sets based on the specified test size and random state.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X represents the input features or data that you want to split into training and
                testing sets
                
            y : np.ndarray
                The parameter `y` represents the target variable or the dependent variable in the supervised
                learning problem. It is a np.ndarray containing the true values of the target variable for the
                corresponding samples in the input data `X`     
                
            test_size : float, optional
                The parameter `test_size` represents the proportion of the dataset that should be allocated to
                the test set. The default value is 0.2, which means that 20% of the data will be used for testing
                
            random_state : int, optional
                The parameter `random_state` is used to set the random seed for reproducibility. By setting a
                specific value for `random_state`, you can ensure that the random initialization of the model's
                weights and any other random operations are consistent across different runs of the code.
            
        Returns
        -------
            np.ndarray
                The function returns four np.ndarrays: X_train, X_test, y_train, and y_test. X_train and y_train
        """
        num_samples = X.shape[0]
        if random_state:
            np.random.seed(random_state)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_test_samples = int(test_size * num_samples)
        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test

    def _regularization_term(self, coefficients):
        """
        The function calculates the regularization term based on the type of regularization and the given
        coefficients.
        
        Regularization is a technique used to reduce overfitting by adding a penalty term to the loss function.
        The regularization term is the sum of the squares of the coefficients. It is used to shrink the
        coefficients of the model to zero, thereby reducing the number of features used in the model. This is
        useful when the dataset has a large number of features and only a few of them are important for the
        model.
        
        Parameters
        ----------
            coefficients : tf.Variable
                The parameter `coefficients` represents the weights or coefficients of the model. It is a
                TensorFlow variable that contains the weights of the model
        
        Returns
        -------
            tf.Tensor
                The regularization term is being returned.
        """
        if self.regularization == 'l1':
            return self.reg_strength * tf.reduce_sum(tf.abs(coefficients))
        elif self.regularization == 'l2':
            return self.reg_strength * tf.reduce_sum(tf.square(coefficients))
        else:
            return 0

    def _create_learning_rate_scheduler(self):
        """
        The function `_create_learning_rate_scheduler` returns a learning rate scheduler that reduces the
        learning rate by 10% every 30 epochs.
        
        Returns
        -------
            function
                The function `learning_rate_scheduler` is being returned.
        """
        def learning_rate_scheduler(epoch, lr):
            """
            The function `learning_rate_scheduler` takes in the epoch number and the current learning rate
            and returns the new learning rate.
            
            Parameters
            ----------
                epoch : int
                    The parameter `epoch` represents the current epoch number
                    
                lr : float
                    The parameter `lr` represents the current learning rate
                
            Returns
            -------
                float
                    The new learning rate is being returned.
            """
            if epoch % 30 == 0:
                return lr * 0.1  # Reduce learning rate by 10% every 50 epochs
            return lr
        return learning_rate_scheduler

    def fit(self, X, y, X_val=None, y_val=None, random_seed=None):
        """
        The `fit` function trains a linear regression model using Mini-batch Gradient Descent with early
        stopping and learning rate scheduling.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X represents the input features or data that you want to use to train the model
                
            y : np.ndarray
                The parameter `y` represents the target variable or the dependent variable in the supervised
                learning problem. It is a np.ndarray containing the true values of the target variable for the
                corresponding samples in the input data `X`
                
            X_val : np.ndarray, optional
                The parameter `X_val` represents the input features or data that you want to use to validate the
                model. It is a np.ndarray containing the features of the validation set
                
            y_val : np.ndarray, optional
                The parameter `y_val` represents the target variable or the dependent variable in the supervised
                learning problem. It is a np.ndarray containing the true values of the target variable for the
                corresponding samples in the input data `X_val`
                
            random_seed : int, optional
                The parameter `random_seed` is used to set the random seed for reproducibility. By setting a
                specific value for `random_seed`, you can ensure that the random initialization of the model's
                weights and any other random operations are consistent across different runs of the code.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        num_samples, num_features = X.shape
        X = np.hstack((np.ones((num_samples, 1)), X))  # Add a bias term (1) to the scaled features

        # Convert np.ndarrays to TensorFlow tensors
        X = tf.constant(X, dtype=tf.float32)
        y = tf.constant(y.reshape(-1, 1), dtype=tf.float32)

        # Initialize the coefficients (weights) with Glorot uniform initialization
        glorot_initializer = tf.keras.initializers.GlorotUniform()
        self.coefficients = tf.Variable(glorot_initializer((num_features + 1, 1)))

        # Learning Rate Scheduling with momentum-based optimizer (Adam)
        if self.use_learning_rate_scheduling:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_scheduler,
                clipvalue=1.0
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipvalue=1.0
            )

        # Training loop with Mini-batch Gradient Descent and early stopping
        num_batches = int(np.ceil(num_samples / self.batch_size))
        best_val_loss = float('inf')
        patience_count = 0
        for epoch in range(self.num_epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for batch in range(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_X = tf.gather(X, batch_indices)
                batch_y = tf.gather(y, batch_indices)

                with tf.GradientTape() as tape:
                    predictions = tf.matmul(batch_X, self.coefficients)
                    loss = tf.reduce_mean(tf.square(predictions - batch_y))
                    if self.reg_strength > 0.0:
                        loss += self.reg_strength * tf.reduce_sum(tf.square(self.coefficients))

                gradients = tape.gradient(loss, self.coefficients)
                optimizer.apply_gradients(zip([gradients], [self.coefficients]))

            # Calculate validation loss and perform early stopping with patience
            if X_val is not None and y_val is not None:
                val_accuracy, val_loss = self.evaluate(X_val, y_val)
                if val_loss + self.tolerance < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.early_stopping_patience:
                        break  # Stop training if validation loss doesn't improve for 'patience' epochs

            if self.verbose:
                if X_val is None or y_val is None:
                    print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {loss:.5f}")
                else:
                    print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {loss:.5f} - Val Loss: {val_loss:.5f}")
                    
    def _calculate_class_weights(self, y):
        """
        The function calculates class weights based on the frequency of each class in the target variable.
        
        Parameters
        ----------
            y : np.ndarray
                The parameter `y` represents the target variable or the dependent variable in the supervised
                learning problem. It is a np.ndarray containing the true values of the target variable for the
                corresponding samples in the input data `X`
            
        Returns
        -------
            tf.Tensor
                The class weights are being returned.
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = y.shape[0]
        class_weights = total_samples / (len(unique_classes) * class_counts)
        max_weight = np.max(class_weights)
        class_weights = max_weight / class_weights
        class_weights = tf.constant(class_weights, dtype=tf.float32)
        return class_weights
        
    def predict_proba(self, X):
        """
        The `predict_proba` function takes in a set of features `X`, scales the features, adds a column of
        ones to the scaled features, performs matrix multiplication with the coefficients, applies the
        sigmoid function to the logits, and returns the probabilities.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X represents the input data for which you want to make predictions. It could be a
                single data point or a collection of data points. The shape of X should match the shape of the
                training data used to train the model
            
        Returns
        -------
            np.ndarray
                The `predict_proba` function returns a np.ndarray containing the predicted probabilities for
                each class.
        """
        X_scaled = self._scale_features(X)
        X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
        X_scaled = tf.constant(X_scaled, dtype=tf.float32)

        logits = tf.matmul(X_scaled, self.coefficients)  # Use matrix multiplication directly
        probabilities = self.sigmoid(logits)
        return probabilities.numpy()

    def predict(self, X):
        """
        The `predict` function takes in a set of input data `X` and returns the predicted class labels based
        on the highest probability from the `predict_proba` function.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X represents the input data for which you want to make predictions. It could be a
                single data point or a collection of data points. The shape of X should match the shape of the
                training data used to train the model
            
        Returns
        -------
            np.ndarray
                The `predict` function returns a np.ndarray containing the predicted class labels for each data
                point in `X`.
            
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """
        The function calculates the accuracy and loss of a binary classification model using TensorFlow's
        binary cross-entropy loss function.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X represents the input data or features that will be used to make predictions or
                classifications. It could be a matrix or an array-like object
                
            y : np.ndarray
                The parameter `y` represents the target variable or the dependent variable. It is the variable
                that we are trying to predict or model. In the context of machine learning, `y` typically
                represents the labels or classes of the data
            
        Returns
        -------
            tuple(float, float)
                The function returns a tuple containing the accuracy and loss of the model on the given dataset.
        """
        y_pred_proba = self.predict_proba(X)

        # Clip probabilities to avoid log(0) and log(1) issues
        y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)

        y_binary = np.array([np.array([1.0 if i == label else 0.0 for i in range(len(np.unique(y)))]) for label in y])

        # Use TensorFlow's binary cross-entropy loss function
        loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_proba).numpy()
        loss = np.mean(loss)

        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = np.mean(y_pred == y)

        return accuracy, loss

    def evaluate(self, X, y):
        """
        The evaluate function returns the score of a model on a given dataset.
        
        Parameters
        ----------
            X : np.ndarray
                The parameter X represents the input data or features that will be used to make predictions or
                classifications. It could be a matrix or an array-like object
                
            y : np.ndarray
                The parameter `y` represents the target variable or the dependent variable. It is the variable
                that we are trying to predict or model. In the context of machine learning, `y` typically
                represents the labels or classes of the data
            
        Returns
        -------
            tuple(float, float)
                The function returns a tuple containing the accuracy and loss of the model on the given dataset.
        """
        return self.score(X, y)