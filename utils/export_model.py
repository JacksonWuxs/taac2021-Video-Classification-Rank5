# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to export a model for batch prediction."""

import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

_TOP_PREDICTIONS_IN_OUTPUT = 82

class ModelExporter(object):

  def __init__(self, model, reader):
    self.model = model
    self.reader = reader

    with tf.Graph().as_default() as graph:
      self.inputs, self.outputs = self.build_inputs_and_outputs()
      self.graph = graph
      self.saver = tf.train.Saver(tf.global_variables(), sharded=True)
      
  def export_model(self, model_dir, global_step_val, last_checkpoint):
    """Exports the model so that it can used for batch predictions."""

    with self.graph.as_default():
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        self.saver.restore(session, last_checkpoint)

        signature = signature_def_utils.build_signature_def(
            inputs=self.inputs,
            outputs=self.outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}

        model_builder = saved_model_builder.SavedModelBuilder(model_dir)
        model_builder.add_meta_graph_and_variables(session,
            tags=[tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save()

  def build_inputs_and_outputs(self):
    input_name_list = self.reader.dname_string_list #模型输入变量名
    inupt_shape_list = self.reader.data_shape_list  #模型输入shape
    input_dtype_list = self.reader.dtype_list       #模型输入类型
    
    inputs_dict={}
    for input_name,input_shape,input_dtype in zip(input_name_list, inupt_shape_list, input_dtype_list):
      inputs_dict[input_name] = tf.placeholder(shape=[None]+input_shape, dtype=input_dtype, name=input_name) #add batch size dim
        
    with tf.variable_scope("tower"):
      result = self.model(inputs_dict,is_training=False)
      predictions = result["tagging_output_fusion"]["predictions"]
      video_embedding = result["video_embedding"]
      top_predictions, top_indices = tf.nn.top_k(predictions, _TOP_PREDICTIONS_IN_OUTPUT)

    #inputs = {"video_input_placeholder": saved_model_utils.build_tensor_info(video_input_placeholder),
    #          "audio_input_placeholder": saved_model_utils.build_tensor_info(audio_input_placeholder),
    #          "text_input_placeholder":  saved_model_utils.build_tensor_info(text_input_placeholder),
    #          "num_frames_placeholder": saved_model_utils.build_tensor_info(num_frames_placeholder)}
    inputs = {key:saved_model_utils.build_tensor_info(val) for key,val in inputs_dict.items()}
    outputs = {
        "class_indexes": saved_model_utils.build_tensor_info(top_indices),
        "video_embedding": saved_model_utils.build_tensor_info(video_embedding),
        "predictions": saved_model_utils.build_tensor_info(top_predictions)}

    return inputs, outputs
