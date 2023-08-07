from typing import Any
import tensorflow as tf
import numpy as np
import tensorflow_decision_forests as tfdf

class DecisionTree:
    """
    Defines a class that implements a machine learning model using
    TensorFlow Decision Forests (TF-DF) for classification or regression tasks,
    including methods for loading datasets, training the model, making predictions,
    and evaluating the model's performance.
    
    TensorFlow Decision Forests (TF-DF) is a library for training and serving
    TensorFlow models for decision tasks. It is an open-source library that
    provides a collection of state-of-the-art algorithms for decision tasks,
    including classification, regression, ranking, and clustering.
    
    Decision trees (DT) are a type of supervised learning algorithm that can be used
    for both classification and regression tasks. They are a popular choice for
    many machine learning problems because they are easy to understand and
    interpret, and they can be used to solve a wide variety of problems.
    
    CART (Classification and Regression Trees) is a decision tree learning
    algorithm that uses the Gini impurity measure to determine the best split at
    each node. It is a popular choice for many machine learning problems because
    it is easy to understand and interpret, and it can be used to solve a wide
    variety of problems.
    
    Random forests (RFs) are an ensemble learning method that combines multiple
    decision trees to create a more powerful model. They are a popular choice
    for many machine learning problems because they are easy to understand and
    interpret, and they can be used to solve a wide variety of problems.
    
    Gradient Boosted Trees (GBTs) are a type of supervised learning algorithm that
    can be used for both classification and regression tasks. They are a popular
    choice for many machine learning problems because they are easy to understand
    and interpret, and they can be used to solve a wide variety of problems.
    """

    def __init__(self, model = "random forest", verbose = True, _task = "classification"):
        self.verbose = verbose
        self.train_ds = None
        self.test_ds = None
        self.val_ds = None
        self.tuner = tfdf.tuner.RandomSearch(num_trials=20)
        self.task = self._get_task(_task)
        self.model = self._get_model(model, self.task)
        self.class_weights = None
    
    def _setup_hyperparameters(self):
        """
        The function sets up hyperparameters for a machine learning model, specifically the maximum depth of
        a decision tree.
        """
        self.tuner.choice("max_depth", [3, 4, 5, 6, 7])
        
    def get_params(self) -> dict[str, Any]:
        """
        The function `get_params` returns the configuration of the model.
        Returns
        -------
            The configuration of the model.
        """
        return self.model.get_config()
    
    def load_dataset(self, dataset_df, label, test_ratio = 0.2):        
        """
        The function `load_dataset` takes a dataset dataframe, splits it into train, validation, and test
        sets, converts them into TensorFlow datasets, and calculates class weights.
        
        Parameters
        ----------
            dataset_df : pandas.DataFrame
                The dataset_df parameter is a pandas DataFrame that contains the dataset you want to load. It
                should have the features as columns and the corresponding labels as a separate column
                
            label : str
                The "label" parameter is the column name of the target variable in the dataset. It is the
                variable that you want to predict or classify   
                
            test_ratio : float, optional
                The `test_ratio` parameter is the ratio of the dataset that should be used for testing. It
                determines the proportion of the dataset that will be split into the test set. The remaining
                portion of the dataset will be used for training and validation, by default 0.2
        """

        train_df, test_df = self._split_dataset(dataset_df, test_ratio)
        train_df, val_df = self._split_dataset(train_df, test_ratio)

        self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label, task = self.task)
        self.val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_df, label=label, task = self.task)
        self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=label, task = self.task)
        if self.task == tfdf.keras.Task.CLASSIFICATION:
            self.class_weights = self._get_class_weights(dataset_df, label)

    def _get_class_weights(self, dataset_df, label) -> dict:
        """
        The function calculates and returns the class weights for a given dataset based on a specified
        label.
        
        Parameters
        ----------
        dataset_df : pandas.DataFrame
                The dataset_df parameter is a pandas DataFrame that represents the dataset. It contains the data
                for which we want to calculate the class weights
                
            label : str
                The "label" parameter represents the column name in the dataset_df dataframe that contains the
                class labels for each data point
        Returns
        -------
            dict
                a dictionary of class weights for a given dataset.

        """
        class_weights = {}
        for class_ in dataset_df[label].unique():
            class_weights[class_] = len(dataset_df[label]) / (len(dataset_df[dataset_df[label] == class_]) * len(dataset_df[label].unique()))
        return class_weights
                
    def fit(self, early_stopping_patience=5, learning_rate=0.001, momentum=0.9, _metrics = ["accuracy"]):
        """
        The `fit` function trains a model using stochastic gradient descent optimizer with early stopping
        and specified hyperparameters.
        
        Parameters
        ----------
            early_stopping_patience : int, optional
                The early_stopping_patience parameter determines the number of epochs to wait before stopping
                the training process if the validation loss does not improve. If the validation loss does not
                improve for the specified number of epochs, training will be stopped early, defaults to 5
                (optional)
                
            learning_rate : float, optional
                The learning rate determines the step size at each iteration while training the model. It
                controls how much the model's weights are updated based on the calculated gradients. A higher
                learning rate can result in faster convergence but may also cause the model to overshoot the
                optimal solution. On the other hand, a lower learning rate can result in slower convergence but
                may also result in a more stable model, defaults to 0.001 (optional)
                
            momentum : float, optional
                Momentum is a hyperparameter used in optimization algorithms, such as Stochastic Gradient
                Descent (SGD), to accelerate convergence and escape local minima. It determines the contribution
                of the previous update to the current update of the model's weights, defaults to 0.9 (optional)
                
            _metrics : list, optional
                _metrics is a list of metrics that will be used to evaluate the model's performance during
                training and validation. These metrics can include accuracy, precision, recall, F1 score, etc,
                defaults to ["accuracy"] (optional)     
        """
        self._setup_hyperparameters()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        self.model.compile(optimizer=optimizer, metrics=_metrics)
        self.model.fit(self.train_ds,
                       validation_data=self.val_ds,
                       callbacks=[early_stopping],
                       class_weight=self.class_weights
                       )
        
    
    def predict(self, length = 3, split = "test") -> Any:
        """
        The `predict` function takes in a length and split parameter, and returns the predictions made by
        the model on the specified dataset split.
        
        Parameters
        ----------
            length : int, optional
                The `length` parameter specifies the number of samples to be taken from the dataset. It
                determines how many samples will be used for prediction, defaults to 3 (optional)
                
            split : str, optional
                The "split" parameter determines whether to use the test or train dataset for prediction. If
                "split" is set to "test", the function will use the test dataset and if it is set to "train",
                the function will use the train dataset, defaults to test (optional)
        Returns
        -------
            The predictions made by the model on the specified dataset (either the test dataset or the train
            dataset).
        """
        if split == "test":
            ds = self.test_ds.take(length)
        elif split == "train":
            ds = self.train_ds.take(length)
        return self.model.predict(ds, verbose=self.verbose)
    
    def evaluate(self) -> dict:
        """
        The `evaluate` function evaluates a machine learning model on a test dataset and returns the
        evaluation metrics.
        
        Returns
        -------
            The evaluation results of the model on the test dataset. It returns a dictionary with the evaluation
            metrics as keys and their corresponding values.
        """
        evaluation = self.model.evaluate(self.test_ds, return_dict=True)
        
        if self.verbose:
            for name, value in evaluation.items():
                print(f"{name}: {value:.4f}")
        
        return evaluation
    
    def info(self):
        """
        The `info` function returns a summary of the model.
        Returns
        -------
            A summary of the model.
        """
        return self.model.summary()
    
    def _create_dataset(self, dataset_df, label):
        """
        The `_create_dataset` function takes a dataset dataframe and a label column, encodes the categorical
        labels as integers, splits the dataset into training and testing datasets, and returns the resulting
        datasets.
        
        Parameters
        ----------
            dataset_df : pandas.DataFrame
                The dataset_df parameter is a pandas DataFrame that represents the dataset. It contains the data
                for which we want to calculate the class weights
                
            label : str
                The "label" parameter represents the column name in the dataset_df dataframe that contains the
                class labels for each data point
        Returns
        -------
            train_ds : tf.data.Dataset
                The train_ds parameter is a TensorFlow dataset that contains the training data
                
            test_ds : tf.data.Dataset
                The test_ds parameter is a TensorFlow dataset that contains the testing data
            
        """
        # Name of the label column.
        classes = dataset_df[label].unique().tolist()
        
        # Encode the categorical labels as integers.
        dataset_df[label] = dataset_df[label].map(classes.index)
        
        # Split the dataset into a training and a testing dataset.
        train_ds_pd, test_ds_pd = self._split_dataset(dataset_df)
        
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
        test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)
        
        if self.verbose:
            print(f"Label classes: {classes}")
            print("{} examples in training, {} examples for testing.".format(len(train_ds_pd), len(test_ds_pd)))
            
        return train_ds, test_ds
                
    def _split_dataset(self, dataset, test_ratio = 0.2):
        """
        The function `_split_dataset` takes a pandas dataframe and splits it into two based on a given test
        ratio.
        
        Parameters
        ----------
            dataset : pandas.DataFrame
                The dataset parameter is a pandas DataFrame that represents the dataset. It contains the data
                for which we want to calculate the class weights
                
            test_ratio : float, optional
                The `test_ratio` parameter is the ratio of the dataset that should be used for testing. It
                determines the proportion of the dataset that will be split into the test set. The remaining
                portion of the dataset will be used for training and validation, by default 0.2 (optional)
                
        Returns
        -------
            train_ds : pandas.DataFrame
                The train_ds parameter is a pandas DataFrame that contains the training data
                
            test_ds : pandas.DataFrame
                The test_ds parameter is a pandas DataFrame that contains the testing data
        """
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]
    
    def _get_task(self, task):
        """
        The function `_get_task` returns the appropriate TensorFlow Decision Forest (TF-DF) task based on
        the input task string.
        
        Parameters
        ----------
            task : str
                The `task` parameter is a string that represents the type of task the model will be used for. It
                can take one of the following values: "task", "regression"
                
        Returns
        -------
            The appropriate TensorFlow Decision Forest (TF-DF) task based on the input task string.
        """
        if task == "classification":
            return tfdf.keras.Task.CLASSIFICATION
        elif task == "regression":
            return tfdf.keras.Task.REGRESSION     
        
    def _get_model(self, _model, task):
        """
        The function `_get_model` returns a TensorFlow Decision Forest model based on the input `_model` and
        `task` parameters.
        
        Parameters
        ----------
            _model : str
                The `_model` parameter is a string that represents the type of model to be used. It can take one
                of the following values: "rf", "gbt", "cart", "dgbt"
                
            task : str
                The `task` parameter is a string that represents the type of task the model will be used for. It
                can take one of the following values: "task", "regression"
                
        Returns
        -------
            A TensorFlow Decision Forest model based on the input `_model` and `task` parameters.  
        """
        if _model == "rf":
            return tfdf.keras.RandomForestModel(task=task)
        elif _model == "gbt":
            return tfdf.keras.GradientBoostedTreesModel(task=task)
        elif _model == "cart":
            return tfdf.keras.CartModel(task=task)
        elif _model == "dgbt":
            return tfdf.keras.DistributedGradientBoostedTreesModel(task=task)
        else:
            raise ValueError(f"Unknown model {_model}")