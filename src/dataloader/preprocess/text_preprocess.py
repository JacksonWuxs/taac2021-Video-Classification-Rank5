import tokenization
import numpy as np
import random
import tensorflow as tf
import os

seed = 20210627
random.seed(seed)
tf.set_random_seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


class Preprocess:

    def __init__(self, vocab, max_len, is_training=False):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab)
        self.max_len = max_len
        self.is_training = is_training

    def __call__(self, text, augment):
        with open(text) as f:
            data = eval(f.read().strip())
            text = data['video_ocr'] + data['video_asr']
            text = text.replace("|", "")
            if augment > 0:
                text = text[random.randint(0, int(max(0, len(text) - 50))):]
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text)
        if augment == 2:
            tokens = ['[CLS]'] + [token for token in tokens[1:] if random.random() > 0.1]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)[:self.max_len]
        ids = ids + [0]*(self.max_len-len(ids))
        return np.array(ids).astype('int64')