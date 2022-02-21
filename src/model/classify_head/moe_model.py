import tensorflow.contrib.slim as slim
import tensorflow as tf

class MoeModel():
  """A softmax over a mixture of logistic models (with L2 regularization)."""
  def __init__(self, num_classes, num_mixtures=4, l2_penalty=0.0):
      self.vocab_size = num_classes
      self.num_mixtures = num_mixtures
      self.l2_penalty = l2_penalty

  def __call__(self, model_input):
    """Creates a Mixture of (Logistic) Experts model.
     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.
    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    gate_activations = slim.fully_connected(
        model_input,
        self.vocab_size * (self.num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(self.l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        self.vocab_size * self.num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(self.l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, self.num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, self.num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :self.num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, self.vocab_size])
    return {"predictions": final_probabilities}
