import tensorflow as tf
import numpy as np
import tensorflow_decision_forests as tfdf

# 
# 
class DecisionTree:
    """
    The above code defines a class that implements a machine learning model using
    TensorFlow Decision Forests (TF-DF) for classification or regression tasks,
    including methods for loading datasets, training the model, making predictions,
    and evaluating the model's performance.
    
    :param model: The `model` parameter is a string that represents the type of
    machine learning model to be used. It can take one of the following values:
    "random forest", "gradient boosted trees", "classification and regression trees",
    or "distributed gradient boosted trees". The default value is "random forest",
    defaults to random forest (optional)
    :param verbose: The `verbose` parameter is a boolean value that determines
    whether or not to print additional information during the execution of the code.
    If `verbose` is set to `True`, additional information will be printed. If
    `verbose` is set to `False`, no additional information will be printed, defaults
    to True (optional)
    :param _task: The `_task` parameter is used to specify the type of task the model
    will be used for. It can take one of the following values: "classification" or
    "regression". If the value is "classification", the model will be trained for
    classification tasks. If the value is "regression, defaults to classification
    (optional)
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
        
    def get_params(self):
        """
        The function `get_params` returns the configuration of the model.
        :return: The `get_params` method is returning the configuration of the model.
        """
        return self.model.get_config()
    
    def load_dataset(self, dataset_df, label, test_ratio = 0.2):        
        """
        The function `load_dataset` takes a dataset dataframe, splits it into train, validation, and test
        sets, converts them into TensorFlow datasets, and calculates class weights.
        
        :param dataset_df: The `dataset_df` parameter is a pandas DataFrame that contains the dataset you
        want to load. It should have the features as columns and the corresponding labels as a separate
        column
        :param label: The "label" parameter is the column name of the target variable in the dataset. It is
        the variable that you want to predict or classify
        :param test_ratio: The `test_ratio` parameter is the ratio of the dataset that should be used for
        testing. It determines the proportion of the dataset that will be split into the test set. The
        remaining portion of the dataset will be used for training and validation
        """
        train_df, test_df = self._split_dataset(dataset_df, test_ratio)
        train_df, val_df = self._split_dataset(train_df, test_ratio)

        self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label, task = self.task)
        self.val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_df, label=label, task = self.task)
        self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=label, task = self.task)
        if self.task == tfdf.keras.Task.CLASSIFICATION:
            self.class_weights = self._get_class_weights(dataset_df, label)

    def _get_class_weights(self, dataset_df, label):
        """
        The function calculates and returns the class weights for a given dataset based on a specified
        label.
        
        :param dataset_df: The dataset_df parameter is a pandas DataFrame that represents the dataset. It
        contains the data for which we want to calculate the class weights
        :param label: The "label" parameter represents the column name in the dataset_df dataframe that
        contains the class labels for each data point
        :return: a dictionary of class weights for a given dataset.
        """
        class_weights = {}
        for class_ in dataset_df[label].unique():
            class_weights[class_] = len(dataset_df[label]) / (len(dataset_df[dataset_df[label] == class_]) * len(dataset_df[label].unique()))
        return class_weights
                
    def fit(self, early_stopping_patience=5, learning_rate=0.001, momentum=0.9, _metrics = ["accuracy"], warmup_steps=100, total_steps=200, batch_size=32):
        """
        The `fit` function trains a model using stochastic gradient descent optimizer with early stopping
        and specified hyperparameters.
        
        :param early_stopping_patience: The early_stopping_patience parameter determines the number of
        epochs to wait before stopping the training process if the validation loss does not improve. If the
        validation loss does not improve for the specified number of epochs, training will be stopped early,
        defaults to 5 (optional)
        :param learning_rate: The learning rate determines the step size at each iteration while training
        the model. It controls how much the model's weights are updated based on the calculated gradients. A
        higher learning rate can result in faster convergence but may also cause the model to overshoot the
        optimal solution. On the other hand, a lower
        :param momentum: Momentum is a hyperparameter used in optimization algorithms, such as Stochastic
        Gradient Descent (SGD), to accelerate convergence and escape local minima. It determines the
        contribution of the previous update to the current update of the model's weights
        :param _metrics: _metrics is a list of metrics that will be used to evaluate the model's performance
        during training and validation. These metrics can include accuracy, precision, recall, F1 score, etc
        :param warmup_steps: The warmup_steps parameter is used to specify the number of steps during the
        warm-up phase of training. During the warm-up phase, the learning rate is gradually increased from a
        small value to the specified learning_rate. This helps the model to stabilize and avoid large
        updates in the early stages of training, defaults to 100 (optional)
        :param total_steps: The `total_steps` parameter represents the total number of steps or iterations
        to train the model. Each step typically corresponds to one batch of data being processed, defaults
        to 200 (optional)
        :param batch_size: The batch_size parameter determines the number of samples that will be propagated
        through the network at each training step. It is a hyperparameter that can be adjusted to balance
        between memory usage and training speed, defaults to 32 (optional)
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
        
    
    def predict(self, length = 3, split = "test"):
        """
        The `predict` function takes in a length and split parameter, and returns the predictions made by
        the model on the specified dataset split.
        
        :param length: The `length` parameter specifies the number of samples to be taken from the dataset.
        It determines how many samples will be used for prediction, defaults to 3 (optional)
        :param split: The "split" parameter determines whether to use the test or train dataset for
        prediction. If "split" is set to "test", the function will use the test dataset and if it is set to
        "train", the function will use the train dataset, defaults to test (optional)
        :return: The method is returning the predictions made by the model on the specified dataset (either
        the test dataset or the train dataset).
        """
        if split == "test":
            ds = self.test_ds.take(length)
        elif split == "train":
            ds = self.train_ds.take(length)
        return self.model.predict(ds, verbose=self.verbose)
    
    def evaluate(self):
        """
        The `evaluate` function evaluates a machine learning model on a test dataset and returns the
        evaluation metrics.
        :return: The `evaluate` method returns the evaluation results of the model on the test dataset. It
        returns a dictionary with the evaluation metrics as keys and their corresponding values.
        """
        evaluation = self.model.evaluate(self.test_ds, return_dict=True)
        
        if self.verbose:
            for name, value in evaluation.items():
                print(f"{name}: {value:.4f}")
        
        return evaluation
    
    def info(self):
        """
        The `info` function returns a summary of the model.
        :return: The summary of the model.
        """
        return self.model.summary()
    
    def _create_dataset(self, dataset_df, label):
        """
        The `_create_dataset` function takes a dataset dataframe and a label column, encodes the categorical
        labels as integers, splits the dataset into training and testing datasets, and returns the resulting
        datasets.
        
        :param dataset_df: The dataset_df parameter is a pandas DataFrame that contains the dataset you want
        to use for training and testing. It should have columns representing the features of the dataset and
        a column representing the label
        :param label: The `label` parameter is the name of the column in the dataset that contains the
        labels or target variable
        :return: two variables: `train_ds` and `test_ds`. These variables are the training and testing
        datasets, respectively, after performing some preprocessing steps.
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
        
        :param dataset: The dataset parameter is a pandas dataframe that you want to split into two parts
        :param test_ratio: The test_ratio parameter is the ratio of the dataset that will be used for
        testing. For example, if test_ratio is set to 0.2, it means that 20% of the dataset will be used for
        testing and the remaining 80% will be used for training
        :return: two panda dataframes. The first dataframe contains the rows from the original dataset that
        were not selected as test data, and the second dataframe contains the rows that were selected as
        test data.
        """
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]
    
    def _get_task(self, task):
        """
        The function `_get_task` returns the appropriate TensorFlow Decision Forest (TF-DF) task based on
        the input task string.
        
        :param task: The `task` parameter is a string that specifies the type of task for which you want to
        get the corresponding TensorFlow Decision Forests (TF-DF) task. It can have two possible values:
        :return: a task object based on the input parameter. If the input parameter is "classification", it
        returns the classification task object from the tfdf.keras.Task class. If the input parameter is
        "regression", it returns the regression task object from the tfdf.keras.Task class.
        """
        if task == "classification":
            return tfdf.keras.Task.CLASSIFICATION
        elif task == "regression":
            return tfdf.keras.Task.REGRESSION     
        
    def _get_model(self, _model, task):
        """
        The function `_get_model` returns a TensorFlow Decision Forest model based on the input `_model` and
        `task` parameters.
        
        :param _model: The `_model` parameter is a string that represents the type of model to be used. It
        can take one of the following values: "rf" (Random Forest), "gbt" (Gradient Boosted Trees), "cart"
        (Classification and Regression Trees), or "dgbt" (
        :param task: The `task` parameter is used to specify the type of task the model will be used for. It
        can take one of the following values:
        :return: The function `_get_model` returns a TensorFlow Decision Forest (TF-DF) model based on the
        input `_model` and `task` parameters. The specific model returned depends on the value of `_model`:
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