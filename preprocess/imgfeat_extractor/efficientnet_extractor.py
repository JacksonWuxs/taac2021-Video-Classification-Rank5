# Author: wuxsmail@163.com

import time

import numpy as np
import tensorflow as tf
import cv2

from efficientnet.tfkeras import EfficientNetB5
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input


def center_crop_and_resize(frame, size):
    """change shape of a frame with shape (h, w, 3) into shape (size, size, 3)
    """
    # prepare_frame
    assert len(frame.shape) == 3 and frame.shape[-1] == 3
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
        
    # center crop process
    y, x = frame.shape[0:2]
    if x != y:
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        frame = frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

    # resize process
    h, w = frame.shape[:2]
    if h * w < size ** 2:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
    elif not (h == w == size):
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return np.expand_dims(frame, 0).astype(np.float32)


class EfficientNetExtractor(object):
    """Extracts EfficientNet features for RGB frames.
    """

    def __init__(self, img_size=456, max_pooling=True):
        self.index = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        self.session = tf.compat.v1.Session(config=config)
        self.graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(self.session)
        self.model = EfficientNetB5(
                   weights='pretrained/efficientnet/efficientnet-b5_noisy-student_notop.h5',
                   include_top=False,
                   pooling='avg')
        self.img_size = img_size
        self.block7 = self.model.output
        self.block6 = self.model.layers[-48].output

    def extract_rgb_frame_features(self, frame_rgb):
        assert len(frame_rgb.shape) == 4
        assert frame_rgb.shape[3] == 3  # 3 channels (R, G, B)
        with self.graph.as_default():
            tf.keras.backend.set_session(self.session)
            block7, block6 = self.session.run([self.block7, self.block6], feed_dict={self.model.input: frame_rgb})
            return np.hstack([block7, np.reshape(block6, [block6.shape[0], -1, block6.shape[-1]]).mean(1)])

    def extract_rgb_frame_features_list(self, frame_rgb_list, batch_size):
        self.index += 1
        def _predict_batch():
            if len(frame_list) > 0:
                batch_inputs = preprocess_input(np.vstack(frame_list))
                batch_feat = self.extract_rgb_frame_features(batch_inputs)
                feature_list.extend(frame for frame in batch_feat)

        frame_list = []
        feature_list = []
        for frame in frame_rgb_list:
            frame_list.append(center_crop_and_resize(frame, self.img_size))
            if len(frame_list) == batch_size:
                _predict_batch()
                frame_list = []
        else:
            _predict_batch()
        msg = "[%s] Video-%d has Frames: %d | Feature Dimension: %s" % (time.asctime(), 
                                                                        self.index, 
                                                                        len(feature_list), 
                                                                        feature_list[-1].shape[-1])
        with open("/home/tione/notebook/log/extract_train.log", "a+") as f:
            f.write(msg + "\n")
        return feature_list
