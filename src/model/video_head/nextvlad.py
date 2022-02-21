import tensorflow.contrib.slim as slim
import tensorflow as tf

class NeXtVLAD():
    def __init__(self, feature_size, max_frames, nextvlad_cluster_size, expansion, groups, directly=False):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.nextvlad_cluster_size = nextvlad_cluster_size
        self.expansion = expansion
        self.groups = groups
        self.directly = directly

    def __call__(self, input, is_training, mask=None):
        input = slim.fully_connected(input, self.expansion * self.feature_size, activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(input, self.groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, self.max_frames*self.groups, 1])
        feature_size = self.expansion * self.feature_size // self.groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion*self.feature_size, self.groups*self.nextvlad_cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn",
            fused=False)

        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.nextvlad_cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)
        activation = tf.multiply(activation, attention)
        # tf.summary.histogram("cluster_output", activation)
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.nextvlad_cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.nextvlad_cluster_size * feature_size])
        #return tf.reshape(vlad, (-1, 16, self.nextvlad_cluster_size * feature_size // 16))
        vlad = slim.batch_norm(vlad,
                center=True,
                scale=True,
                is_training = is_training,
                scope="vlad_bn",
                fused=False)
        return vlad
