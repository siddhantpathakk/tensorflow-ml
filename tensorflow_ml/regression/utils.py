import tensorflow as tf
import numpy as np

init_param_dict = {
    'n_examples': 0,
    'training_steps': 0,
    'display_step': 0,
    'learning_rate': 0
}

def prediction(x: np.ndarray, weight: tf.Variable, bias: tf.Variable):
    """Our predicted (learned)  m and c, expression is like y=m*x + c

    Args:
        x (np.ndarray): Data
        weights (tf.Variable): weight (m)
        biases (tf.Variable): bias (c)

    Returns:
        Predicted value
    """
    #convert x to tensor
    x = tf.convert_to_tensor(value=x, dtype=tf.float64)
    # using tensorflow to calculate predicted value x dot m + c
    return tf.add(tf.matmul(x, weight), bias)
    
def loss(x: np.ndarray, y: np.ndarray, weights: tf.Variable, biases:tf.Variable):
    """A loss function using mean-squared error

    Args:
        x (np.ndarray): Data
        y (np.ndarray): Actual values
        weights (tf.Variable): weight (m)
        biases (tf.Variable): bias (c)

    Returns:
        Overall mean of squared error
    """
    
    #  how 'wrong' our predicted (learned)  y is
    error = prediction(x, weights, biases) - y 
    
    squared_error = tf.square(error)
    
    # overall mean of squared error
    return tf.reduce_mean(input_tensor=squared_error)
  
 
def grad(x : np.ndarray, y:np.ndarray, weights : tf.Variable , biases : tf.Variable):
    """Calculate the derivative of loss with respect to weight and bias

    Args:
        x (np.ndarray): Data
        y (np.ndarray): Actual values
        weights (tf.Variable): weight (m)
        biases (tf.Variable): bias (c)

    Returns:
        Direction and value of the gradient of our loss w.r.t weight and bias
    """
    # magical feature of tensorflow to calculate gradient for backpropagation
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, weights, biases)
    
    # direction and value of the gradient of our loss w.r.t weight and bias
    return tape.gradient(loss_, [weights, biases])