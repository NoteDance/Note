import tensorflow as tf


# Define a function that takes y_true, y_pred, threshold as parameters
def binary_accuracy(y_true, y_pred, threshold=0.5):
  # Convert y_pred to a boolean tensor by comparing it with the threshold
  y_pred = tf.cast(y_pred > threshold, tf.bool)
  # Convert y_true to a boolean tensor by comparing it with the threshold
  y_true = tf.cast(y_true > threshold, tf.bool)
  # Count the number of elements that are equal in y_true and y_pred
  correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int32))
  # Get the total number of elements in y_true
  total = tf.size(y_true)
  # Calculate the ratio of correct elements to total elements, which is the accuracy
  accuracy = tf.cast(correct, tf.float32) / tf.cast(total, tf.float32)
  # Return the accuracy as a one-element tensor
  return tf.reshape(accuracy, (1,))


# Define a function that takes predictions and labels as parameters
def categorical_accuracy(y_true, y_pred):
  # Get the index of the maximum element in each row of predictions, which is the predicted class
  y_pred = tf.math.argmax(y_pred, axis=-1)
  # Get the index of the maximum element in each row of labels, which is the true class
  y_true = tf.math.argmax(y_true, axis=-1)
  # Compare the predicted class and the true class for each sample
  accuracy = tf.math.equal(y_pred, y_true)
  # Convert the comparison result to a float tensor
  accuracy = tf.cast(accuracy, tf.float32)
  # Return the accuracy as a one-dimensional tensor
  return accuracy


# Define a function that takes y_true and y_pred as parameters
def sparse_categorical_accuracy(y_true, y_pred):
  # Get the index of the maximum element in each row of y_pred, which is the predicted class
  y_pred = tf.argmax(y_pred, axis=-1)
  # Compare the predicted class and the true class for each sample
  equal = tf.equal(tf.cast(y_true, y_pred.dtype.name), y_pred)
  # Calculate the mean of the comparison result, which is the accuracy
  accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
  # Return the accuracy as a scalar tensor
  return accuracy


# Define a function that takes y_true, y_pred, k as parameters
def sparse_top_k_categorical_accuracy(y_true, y_pred, k):
  # Reshape y_true to a one-dimensional integer tensor
  y_true = tf.reshape(y_true, [-1])
  # Get the shape of y_pred, assume it is (batch_size, num_classes)
  shape = tf.shape(y_pred)
  # Calculate the top-k indices for each row (each sample) of y_pred, return an integer tensor of shape (batch_size, k)
  top_k_indices = tf.math.top_k(y_pred, k).indices
  # Create a boolean tensor of shape (batch_size, num_classes), where the element at row i and column j is True if and only if j is in top_k_indices[i]
  top_k_mask = tf.reduce_any(tf.equal(tf.expand_dims(top_k_indices, axis=-1), tf.range(shape[-1])), axis=-2)
  # Create a boolean tensor of shape (batch_size, num_classes), where the element at row i and column j is True if and only if j equals to y_true[i]
  true_mask = tf.equal(tf.expand_dims(y_true, axis=-1), tf.range(shape[-1]))
  # Calculate the logical and of two boolean tensors, get a boolean tensor of shape (batch_size, num_classes), where the element at row i and column j is True if and only if j is in top_k_indices[i] and equals to y_true[i]
  correct_mask = tf.logical_and(top_k_mask, true_mask)
  # Count the number of True elements in each row (each sample) of correct_mask, get an integer tensor of shape (batch_size,)
  num_corrects = tf.reduce_sum(tf.cast(correct_mask, dtype=tf.int32), axis=-1)
  # Check if each sample has at least one correct prediction, get a boolean tensor of shape (batch_size,)
  is_correct = tf.greater(num_corrects, 0)
  # Calculate the mean of the boolean tensor, which is the top-k accuracy
  accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))
  # Return the accuracy as a scalar tensor
  return accuracy


# Define a function that takes y_true, y_pred, k as parameters
def top_k_categorical_accuracy(y_true, y_pred, k):
  # Get the shape of y_pred, assume it is (batch_size, num_classes)
  shape = tf.shape(y_pred)
  # Calculate the top-k indices for each row (each sample) of y_pred, return an integer tensor of shape (batch_size, k)
  top_k_indices = tf.math.top_k(y_pred, k).indices
  # Create a boolean tensor of shape (batch_size, num_classes), where the element at row i and column j is True if and only if j is in top_k_indices[i]
  top_k_mask = tf.reduce_any(tf.equal(tf.expand_dims(top_k_indices, axis=-1), tf.range(shape[-1])), axis=-2)
  # Calculate the logical and of y_true and top_k_mask, get a boolean tensor of shape (batch_size, num_classes), where the element at row i and column j is True if and only if y_true[i][j] is 1 and j is in top_k_indices[i]
  correct_mask = tf.logical_and(top_k_mask, tf.cast(y_true, dtype=tf.bool))
  # Count the number of True elements in each row (each sample) of correct_mask, get an integer tensor of shape (batch_size,)
  num_corrects = tf.reduce_sum(tf.cast(correct_mask, dtype=tf.int32), axis=-1)
  # Check if each sample has at least one correct prediction, get a boolean tensor of shape (batch_size,)
  is_correct = tf.greater(num_corrects, 0)
  # Calculate the mean of the boolean tensor, which is the top-k accuracy
  accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))
  # Return the accuracy as a scalar tensor
  return accuracy
