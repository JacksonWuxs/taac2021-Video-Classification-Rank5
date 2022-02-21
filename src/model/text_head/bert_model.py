import tensorflow as tf
from src.model.text_head.bert_base import BertModel,BertConfig

class BERT():
    def __init__(self, bert_config, bert_emb_encode_size, reuse_variables=tf.AUTO_REUSE):
        self.reuse_variables = reuse_variables
        self.bert_emb_encode_size = bert_emb_encode_size
        self.bert_config = BertConfig(**bert_config)

    def __call__(self, input_ids, is_training):
        input_mask = tf.cast(tf.not_equal(input_ids,0),tf.int32)
        bert_model = BertModel(config = self.bert_config,
                                is_training = is_training,
                                input_ids = input_ids,
                                input_mask = input_mask,
                                reuse_variables = self.reuse_variables)
        
        text_features = bert_model.get_pooled_output()
        text_features = tf.layers.dense(text_features, self.bert_emb_encode_size, activation=None, name='text_features', reuse=self.reuse_variables)
        text_features = tf.layers.batch_normalization(text_features, training=is_training, reuse=self.reuse_variables)
        return text_features
