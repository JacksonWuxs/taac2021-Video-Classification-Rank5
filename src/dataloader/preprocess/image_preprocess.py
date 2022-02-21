import os
import tensorflow as tf
import numpy as np

from cnn_preprocessing import inception_preprocessing

class Preprocess:

    def __init__(self, is_training, return_idx=False):
        self.is_training = is_training
        #with tf.get_default_graph():
        self.path_placeholder = tf.placeholder(shape=None,dtype=tf.string)
        image = tf.io.read_file(self.path_placeholder)
        image = tf.io.decode_image(image, channels=3)
        self.image_shape = (224, 224, 3)
        #TODO(jefxiong, 对不同模型预处理要通用)
        image.set_shape(self.image_shape)
        self.image = inception_preprocessing.preprocess_image(image, 224, 224,
                                                              is_training=self.is_training,
                                                              add_image_summaries=False,
                                                              crop_image=self.is_training)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.return_idx = return_idx

    def __call__(self, path, augment):
        if os.path.exists(path):
            image = self.sess.run(self.image,feed_dict={self.path_placeholder:path})
        else:
            image = np.zeros(self.image_shape)
        if self.return_idx:
            idx = os.path.basename(path).split('.')[0]
            return image, idx
        return image
