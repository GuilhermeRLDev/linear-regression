'''
Created on Thurs Jan 04 21:50:42 2024

@Author: Guilherme Rossetti Lima
'''
import os 
import numpy as np
import tensorflow as tf

epislon = 1.0e-7

#Implementation for layer
class Layer:
  def __init__(self, shape, activation):
    self.weights = tf.Variable(tf.random.normal((shape[0], shape[1]), mean=0.0, stddev=0.05), dtype=tf.float32)
    self.bias = tf.Variable(tf.random.normal((1, shape[0]), mean=0.0, stddev=0.05), dtype=tf.float32)
    self.activation = activation
 
  def predict(self, X, weights, bias):
    #Since a dot product is calculated the sum for the weights is implicitly implemented
    return self.activation(tf.matmul(X, tf.transpose(weights))+bias)

def forward_pass(X, layer):
    return layer.predict(X, layer.weights, layer.bias)

def cross_entropy(predicted, y):
   # Calculate the logs for the predicted values, note that higher probabilities will result in lower logs and 
   # Lower probabilities will result in higher logs 
   log_predicted =  tf.math.log(predicted+epislon)
   # Multiplies the predicted values by Y note that Y is a one hot encoded
   #hence the logs corresponding to false responses will be zeroed out
   prd = tf.multiply(y, log_predicted)
   #Here the we apply a signal since the logs will be negative values
   #The division by the number of predicted item is applied to get the average loss for the batch since this is a vectorized implementation
   loss = -(tf.reduce_sum(prd, axis=1))
   #returns the final liss
   return tf.reduce_mean(loss)


def train_single_layer(X, y, layer, epochs, learning_rate=0.001, validation_split=0.1, calculate_accuracy=None):
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  #Calculate validation and split
  test_size = int(len(X) * validation_split)
  training_size = len(X) - test_size
  #Split dataset between training and validation
  x_training, x_validation = tf.split(X, [training_size, test_size])
  y_training, y_validation = tf.split(y, [training_size, test_size])

  # prepate the dictionary for the history
  history = {}
  history['training'] = {}
  history['training']['loss'] = []
  history['training']['accuracy'] = []
  history['validation'] = {}
  history['validation']['loss'] = []
  history['validation']['accuracy'] = []
  
  trainable_parameter = []
  trainable_parameter.append(layer.weights)
  trainable_parameter.append(layer.bias)

  #Epochs loop
  for epoch in range(epochs):
    #Creates a gradient tape to record the forward pass and apply autodiff to obtain 
    # the diferentials for each parameter update
    with tf.GradientTape() as tape:
      #Execute the forward pass
      predicted = forward_pass(x_training, layer)
      #Execute the loss function, which must be minimized during optimization
      loss = cross_entropy(predicted, y_training)
    
    #Calculates the model's accuracy
    accuracy = calculate_accuracy(predicted, y_training)
    #Calculate the grandient for each trainable parameter in the model
    gradients = tape.gradient(loss, trainable_parameter)

    #Save loss and accuracy in dictionary to keep track of training history
    history['training']['loss'].append(loss.numpy())
    history['training']['accuracy'].append(accuracy)

    #Execute forward pass for the validation model
    predicted_validation =  layer.predict(x_validation, layer.weights, layer.bias)
    
    #Execute loss function and accuracy for the validation
    validation_loss = cross_entropy(predicted_validation, y_validation)
    validation_accuracy = calculate_accuracy(predicted_validation, y_validation)
    #Save values on the history object for this iteration
    history['validation']['loss'].append(validation_loss.numpy())
    history['validation']['accuracy'].append(validation_accuracy)
    
    #Run optimization
    optimizer.apply_gradients(zip(gradients, trainable_parameter))

    print(f"Training epoch: {epoch} - Training accuracy:  {accuracy}  Training loss:{loss} Validaton: {validation_accuracy} Validation loss {validation_loss}.")
  
  #Once training is complete returns the training object
  return history
    

