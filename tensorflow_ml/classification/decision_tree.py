import tensorflow as tf
import numpy as np
import tensorflow_decision_forests as tfdf

class DecisionTree:
    """A CART (Classification and Regression Trees) a decision tree.
    Uses the TensorFlow Decision Forests library.
    Refer to https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras for more information.
    """
    def __init__(self, model = "random forest", verbose = True, _task = "classification"):
        self.verbose = verbose
        self.train_ds = None
        self.test_ds = None
        self.val_ds = None
        self.tuner = tfdf.tuner.RandomSearch(num_trials=20)
        self.model = self._get_model(model, self._get_task(_task))
        self.class_weights = None
    
    def _setup_hyperparameters(self):
        self.tuner.choice("max_depth", [3, 4, 5, 6, 7])
        
    def get_params(self):
        return self.model.get_config()
    
    def load_dataset(self, dataset_df, label, test_ratio = 0.2):        
        train_df, test_df = self._split_dataset(dataset_df, test_ratio)
        train_df, val_df = self._split_dataset(train_df, test_ratio)

        self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label)
        self.val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_df, label=label)
        self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=label)
        self.class_weights = self._get_class_weights(dataset_df, label)

    def _get_class_weights(self, dataset_df, label):
        """Returns the class weights for a given dataset."""
        class_weights = {}
        for class_ in dataset_df[label].unique():
            class_weights[class_] = len(dataset_df[label]) / (len(dataset_df[dataset_df[label] == class_]) * len(dataset_df[label].unique()))
        return class_weights
                
    def fit(self, early_stopping_patience=5, learning_rate=0.001, momentum=0.9, _metrics = ["accuracy"], warmup_steps=100, total_steps=200, batch_size=32):
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
        if split == "test":
            ds = self.test_ds.take(length)
        elif split == "train":
            ds = self.train_ds.take(length)
        return self.model.predict(ds, verbose=self.verbose)
    
    def evaluate(self):
        evaluation = self.model.evaluate(self.test_ds, return_dict=True)
        
        if self.verbose:
            for name, value in evaluation.items():
                print(f"{name}: {value:.4f}")
        
        return evaluation
    
    def info(self):
        return self.model.summary()
    
    def _create_dataset(self, dataset_df, label):
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
        """Splits a panda dataframe in two."""
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]
    
    def _get_task(self, task):
        if task == "classification":
            return tfdf.keras.Task.CLASSIFICATION
        elif task == "regression":
            return tfdf.keras.Task.REGRESSION
        elif task == "ranking":
            return tfdf.keras.Task.RANKING      
        
    def _get_model(self, _model, task):
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