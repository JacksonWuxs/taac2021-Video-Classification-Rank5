import tensorflow.contrib.slim as slim
import tensorflow as tf

class SE():
    """Dropout + Channel Attention
    """
    def __init__(self, drop_rate, hidden1_size, gating_reduction, gating_last_bn=False):
        self.drop_rate = drop_rate
        self.hidden1_size = hidden1_size
        self.gating_reduction = gating_reduction
        self.gating_last_bn = gating_last_bn
        self.expansion = 1.5

    def __call__(self, input_list, is_training):
        #features = []
        #for feature in input_list:
        #    feature = slim.dropout(feature, keep_prob=1.0 - self.drop_rate, is_training=is_training)
        #    features.append(slim.fully_connected(feature, 1024, activation_fn=None))
        concat_feat = tf.concat(input_list, 1)
        concat_feat = slim.dropout(concat_feat, keep_prob=1. - self.drop_rate, is_training=is_training, scope="concat_feat_dropout")
        concat_feat_dim = concat_feat.get_shape().as_list()[1]

        hidden1_weights = tf.get_variable("hidden1_weights",[concat_feat_dim, self.hidden1_size],
                                          initializer=slim.variance_scaling_initializer())
        activation = tf.matmul(concat_feat, hidden1_weights)
        activation = slim.batch_norm(activation,center=True,scale=True,
                                     is_training=is_training,scope="hidden1_bn",fused=False)

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [self.hidden1_size, self.hidden1_size // self.gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(gates,center=True,scale=True,is_training=is_training,
                                activation_fn=slim.nn.relu, scope="gating_bn")
        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [self.hidden1_size // self.gating_reduction, self.hidden1_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        gates = tf.matmul(gates, gating_weights_2)
        if self.gating_last_bn:
            gates = slim.batch_norm(gates, center=True, scale=True, is_training=is_training, scope="gating_last_bn")

        gates = tf.sigmoid(gates)
        #tf.summary.histogram("final_gates", gates)
        activation = tf.multiply(activation, gates)
        return activation
