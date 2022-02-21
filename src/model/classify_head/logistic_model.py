import tensorflow.contrib.slim as slim
import tensorflow as tf

class LogisticModel():
  """Logistic model with L2 regularization."""
  def __init__(self, num_classes, l2_penalty=None):
      self.num_classes = num_classes
      self.l2_penalty =  0.0 if l2_penalty==None else l2_penalty

  def __call__(self, model_input):
    """
    model_input: 'batch' x 'num_features' matrix of input features.
    Returns: The dimensions of the tensor are batch_size x num_classes."""
    logits = slim.fully_connected(
        model_input, self.num_classes, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(self.l2_penalty),
        biases_regularizer=slim.l2_regularizer(self.l2_penalty),
        weights_initializer=slim.variance_scaling_initializer())
    output = tf.nn.sigmoid(logits)
    return {"predictions": output, "logits": logits}
