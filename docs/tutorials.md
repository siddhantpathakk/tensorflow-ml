## Tutorials

Here are some examples of how to use TensorFlow ML to create machine learning models:

**REGRESSION:**

* Linear Regression (Regular, Polynomial, Ridge, Lasso)

```python
from tensorflow_ml.regression.linear import LinearRegression

# Try other regression models if you feel like!
# from tensorflow_ml.regression.polynomial import PolynomialRegression
# from tensorflow_ml.regression.ridge import RidgeRegression
# from tensorflow_ml.regression.lasso import LassoRegression

lr = LinearRegression()

# Set the hyper-parameters such as learning-rate, number of epochs etc
lr.set_params(_params)

# Train the model
lr.fit(x, y)

# Get the MSE
lr.evaluate(x, y)

# Get the predictions
lr.predict(x)

# Get the R-squared value 
lr.score(x, y)

# Get the coefficients of the trained model
lr.get_coeff()
```

**CLASSIFICATION:**

* Logistic Regression

```python
from tensorflow_ml.classification.logistic_regression import LogisticRegression
logistic_regression = LogisticRegression()

# Set the hyper-parameters such as learning-rate, number of epochs etc
logistic_regression.set_params(params)

# Train the model
logistic_regression.fit(X_train, y_train, random_seed=42, X_val=X_val, y_val=y_val)

# Evaluate the model on the test set
accuracy, cross_entropy_loss = logistic_regression.score(X_test, y_test)
```

* Naive Bayes Classifier (Bernoulli NB, Gaussian NB)

```python
from tensorflow_ml.classification.naive_bayes.bernoulli import BernoulliNaiveBayes

# Set some additional hyper-parameters
bnb = BernoulliNaiveBayes(smoothing=1.0)  # Set the desired smoothing parameter

# Train the model
bnb.fit(training_features, training_labels)

# Get the accuracy of the model on test set
accuracy = bnb.evaluate(testing_features, testing_labels)
```

* Random Forests, Gradient Boosted Trees

```python
from tensorflow_ml.classification.decision_tree import DecisionTree

model = DecisionTree(model = "gbt", verbose = True) # Also can use 'rf' for Random Forests, 'cart' for Classification and Regression Tree

# If regression:
model = DecisionTree(model = "gbt", verbose = True, _task = 'regression') 

# Get the parameters pre-defined by the model
model.get_params()

# Load the dataset and convert it to TFDS
model.load_dataset(data, label)

# Train the model
model.fit(_metrics = ['mse', 'accuracy'])

# Evaluate the model and view metrics and loss
model.evaluate()

# Make predictions on test/train split of data
model.predict(length=5, split="test")
```
