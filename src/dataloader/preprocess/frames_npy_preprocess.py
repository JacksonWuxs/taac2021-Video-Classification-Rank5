import os
import tensorflow as tf
import numpy as np
import random
import jieba


def temporal_shift(src, shift_ratio=0.2):
    ts, fs = src.shape
    shift_dim = max(int(fs * shift_ratio) // 2, 1)
    out = np.zeros_like(src)
    out[1:, :shift_dim] = src[:-1, :shift_dim] # shift later
    out[:-1, -shift_dim:] = src[1:, -shift_dim:] # shift earlier
    out[:, shift_dim:-shift_dim] = src[:, shift_dim:-shift_dim] # no shift
    return out


def data_augment(src, noisy=0.5):
    return src + np.random.normal(0, noisy * np.std(src), size=src.shape)


def load_embeddings(path="/home/tione/notebook/VideoStructuring/taac2021_tagging_pytorchyyds/pretrained/word_embed/Tencent_AILab_ChineseEmbedding_cut100w.txt"):
    embeddings = {}
    with open(path) as f:
        for row in f:
            char, score = row.strip().split(" ", 1)
            embeddings[char] = np.fromstring(score, sep=" ")
    return embeddings


word2vec = load_embeddings()            
def concat_w2v(text_path, frames):
    with open(text_path.replace("video_npy/Youtube8M/", "text_txt/").replace(".npy", ".txt")) as f:
        data = eval(f.read().strip())
        tokens = list(jieba.cut(data["video_asr"].replace("|", "")))
        if len(tokens) < len(frames):
            text = data["video_ocr"]
            for char in '''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*|()~`·、[]【】？、。，,.';；‘:""''':
                text = text.replace(char, "")
            tokens = list(jieba.cut(text))
    window_size = max(len(tokens) // len(frames), 1)
    embeddings = []
    for i in range(len(frames)):
        start = i * window_size
        frame_embed = np.zeros((200,))
        k = 0.0
        for token in tokens[start:start + window_size]:
            if token in word2vec:
                frame_embed += word2vec[token]
                k += 1
        if k > 0:
            frame_embed = frame_embed / k
        embeddings.append(frame_embed)
    embeddings = np.vstack(embeddings)
    return np.hstack([frames, embeddings])


def resize_axis(tensor, axis, new_size, fill_value=0):
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))
  
    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])
  
    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)
  
    resized = tf.concat([
        tf.slice(tensor, tf.zeros_like(shape), shape),
        tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized


class Preprocess:
    
    def __init__(self, 
                 max_frames,
                 return_frames_num,
                 feat_dim = 128,
                 is_training=False,
                 return_idx = False):
        self.max_frames = max_frames
        self.return_frames_num = return_frames_num
        self.is_training = is_training
        self.return_idx = return_idx
        self.feat_dim = feat_dim
        self.frames_placeholder = tf.placeholder(shape=[None,None],dtype=tf.float32)
        self.num_frames = tf.minimum(tf.shape(self.frames_placeholder)[0], self.max_frames)
        self.feature_matrix = resize_axis(self.frames_placeholder,axis=0,new_size=self.max_frames)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

    def __call__(self, frames_npy_fn, augment):
        if os.path.exists(frames_npy_fn):
            frames = np.load(frames_npy_fn)
            assert frames.shape[-1] in (2352, 128)
            if augment > 0:
                if frames.shape[-1] == 2352:
                    frames = np.hstack([data_augment(frames[:, :2048], 0.5),
                                        data_augment(frames[:, 2048:], 0.5)])
                elif frames.shape[-1] == 128:
                    frames = data_augment(frames, 0.5)
            if frames.shape[-1] != 128:
                frames = concat_w2v(frames_npy_fn, frames)
            frames = temporal_shift(frames)
        else:
            print("!"*100+"\n Warning: file {} not exits".format(frames_npy_fn))
            frames = np.zeros((1, self.feat_dim))
        feature_matrix,num_frames = self.sess.run([self.feature_matrix, self.num_frames],feed_dict={self.frames_placeholder:frames})
        idx = os.path.basename(frames_npy_fn).split('.')[0]
        return_list = []
        return_list.append(feature_matrix)
        if self.return_frames_num:
            return_list.append(num_frames)
        if self.return_idx:
            return_list.append(idx)
        return tuple(return_list)
