#coding=utf-8
#Author: jefxiong@tencent.com

import tensorflow as tf
import tensorflow.contrib.slim as slim
import src.model.video_head as video_head
import src.model.text_head as text_head
import src.model.image_head as image_head
import src.model.fusion_head as fusion_head
import src.model.classify_head as classcify_head
from src.model.models.base_model import BaseModel

class NextVladBERT(BaseModel):
    def __init__(self, model_config):
        self.with_video_head = model_config['with_video_head']
        self.with_audio_head = model_config['with_audio_head']
        self.with_text_head = model_config['with_text_head']
        self.with_image_head = model_config['with_image_head']
        
        self.use_modal_drop = model_config['use_modal_drop']
        self.modal_drop_rate = model_config['modal_drop_rate']
        self.with_embedding_bn = model_config['with_embedding_bn']

        self.modal_name_list = []
        if self.with_video_head:
            self.modal_name_list.append('video')
            self.video_max_frame = model_config['video_head_params']['max_frames']
        if self.with_audio_head:
            self.modal_name_list.append('audio')
            self.audio_max_frame = model_config['audio_head_params']['max_frames']
        if self.with_text_head: 
            self.modal_name_list.append('text')
        if self.with_image_head:
            self.modal_name_list.append('image')

        self.fusion_head_dict={}
        self.classifier_dict={}
        self.head_dict={}
        
        for modal in (self.modal_name_list+['fusion']):
            fusion_head_params = model_config['fusion_head_params'].copy()
            fusion_head_params['drop_rate'] = fusion_head_params['drop_rate'][modal]
            
            self.fusion_head_dict[modal] = fusion_head.get_instance(model_config['fusion_head_type'], fusion_head_params)
            self.classifier_dict[modal] = classcify_head.get_instance(model_config['tagging_classifier_type'], model_config['tagging_classifier_params'])
            if modal=='video':
                self.head_dict[modal] = video_head.get_instance(model_config['video_head_type'], model_config['video_head_params'])
            elif modal=='audio':
                self.head_dict[modal] = video_head.get_instance(model_config['audio_head_type'], model_config['audio_head_params'])
            elif modal == 'text':
                self.head_dict[modal] = text_head.get_instance(model_config['text_head_type'], model_config['text_head_params'])
            elif modal == 'image':
                self.head_dict[modal] = image_head.get_instance(model_config['image_head_type'], model_config['image_head_params'])
            elif modal == 'fusion':
                pass
            else:
                raise NotImplementedError

    def  _modal_drop(self, x, rate=0.0, noise_shape=None):
        """模态dropout"""
        random_scale = tf.random.uniform(noise_shape)
        keep_mask = tf.cast(random_scale >= rate, x.dtype)
        ret = x * keep_mask
        probs = tf.cast(keep_mask, tf.float32)
        return ret, probs

    def __call__(self, inputs_dict, is_training=False, train_batch_size=1):
        assert is_training is not None
        prob_dict = {}
        embedding_list = []
        
        for modal_name in self.modal_name_list:    
          #Modal Dropout
          if modal_name in ['video', 'audio']:
            drop_shape = [train_batch_size, 1, 1]
            mask = tf.sequence_mask(inputs_dict[modal_name+'_frames_num'], self.video_max_frame, dtype=tf.float32)   
          elif modal_name == 'text': 
            drop_shape = [train_batch_size, 1]
          elif modal_name == 'image': 
            drop_shape = [train_batch_size, 1, 1, 1]
            
          if is_training and self.use_modal_drop:
              inputs_dict[modal_name], prob_dict[modal_name+'_loss_weight'] = self._modal_drop(inputs_dict[modal_name], self.modal_drop_rate, drop_shape)
        
          with tf.variable_scope(modal_name):
            if modal_name in ['video', 'audio']:
                embedding = self.head_dict[modal_name](inputs_dict[modal_name], is_training=is_training, mask=mask)
            else:
                embedding =  self.head_dict[modal_name](inputs_dict[modal_name], is_training=is_training)
            
          with tf.variable_scope("tag_classifier/"+modal_name[0]):
            if self.with_embedding_bn:
                embedding = slim.batch_norm(embedding, center=True, scale=True, is_training=is_training, scope=modal_name[0]+"_feat_bn")
            encode_emb = self.fusion_head_dict[modal_name]([embedding], is_training=is_training)
            prob_dict['tagging_output_'+modal_name] = self.classifier_dict[modal_name](encode_emb)
            embedding_list.append(embedding)
            #if is_training:
            #    tf.summary.histogram("embedding/{}".format(modal_name), embedding)
            #    tf.summary.histogram("encode_emb/{}".format(modal_name), encode_emb)

        #fusion
        with tf.variable_scope("tag_classifier/fusion"):
            fusion_embedding = self.fusion_head_dict['fusion'](embedding_list, is_training = is_training)
            probs = self.classifier_dict['fusion'](fusion_embedding)
            prob_dict['tagging_output_fusion'] = probs
            prob_dict['video_embedding'] = fusion_embedding
        return prob_dict

    def build_loss(self, inputs, results, label_loss_fn_dict):
        loss_dict={}
        for key, loss_fn in label_loss_fn_dict.items():
            if key == 'tagging':
                labels = inputs['tagging']
                for modal in self.modal_name_list + ['fusion']:
                    loss_weight = results.get(modal+"_loss_weight", 1.0)
                    prediction = results["tagging_output_"+modal]["predictions"]
                    loss_dict["tagging_loss_"+modal] = loss_fn.calculate_loss(prediction, labels,
                                                              **dict(loss_weight = loss_weight))
            else:
                raise NotImplementedError
        return loss_dict
