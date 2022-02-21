#coding=utf-8
#Author: jefxiong@tencent.com
#Author: xxx@tencent.com

"""
训练流程的基类实现
"""
import yaml
import json
import os
import shutil
import traceback

import tensorflow as tf
from tensorflow import logging
from tensorflow import gfile
from tensorflow.python.client import device_lib
import tensorflow.contrib.slim as slim

from munch import Munch
import time

import src.model.models as models
import src.dataloader.dataloader as dataloader
import src.loss as loss_lib
import utils.train_util as train_util
from utils.train_util import ParameterServer, task_as_string, start_server
from utils.export_model import ModelExporter

class Trainer(object):
    def __init__(self, cluster, task, model, reader, configs):
        self.model_config = Munch(configs['ModelConfig'])
        self.optimizer_config = Munch(configs['OptimizerConfig'])
        self.data_config = configs['DatasetConfig']
        self.reader = reader
        self.model = model
        self.cluster = cluster
        self.model_export_pb = ModelExporter(model=model, reader=reader)
        self.task = task
        self.is_master = (task.type == "master" and task.index == 0)
        self.config = tf.ConfigProto( allow_soft_placement=True,
                                      log_device_placement=self.optimizer_config.log_device_placement)
        self.config.gpu_options.allow_growth = self.optimizer_config.gpu_allow_growth

        #根据配置文件选择哪些数据模态构建模型(支持四种模态输入)
        self.modal_name_list = []
        if self.model_config.with_video_head: self.modal_name_list.append('video')
        if self.model_config.with_audio_head: self.modal_name_list.append('audio')
        if self.model_config.with_text_head: self.modal_name_list.append('text')
        if self.model_config.with_image_head: self.modal_name_list.append('image')
        print("!"*100,"modal_name_list:", self.modal_name_list)

    def get_train_fetch_dict(self):
        raise NotImplementedError("get_train_fetch_list")

    def get_val_fetch_dict(self):
        raise NotImplementedError

    def load_pretrained_model(self):
        """加载预训练权重"""
        raise NotImplementedError("load_pretrained_model")

    def train_metric_log(self, train_fetch_dict_eval):
        raise NotImplementedError("train_metric_log")

    def eval(self, sess, global_step_val, data_generater, data_source_nam):
        raise NotImplementedError("eval")


    def build_train_graph(self):
        """训练图构建"""
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        #GPU/CPU设置
        gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'][:self.optimizer_config.num_gpu]
        num_gpus = len(gpus)
        if num_gpus > 0:
            logging.info("Using the following GPUs to train: " + str(gpus))
            num_towers = num_gpus
            device_string = '/gpu:%d'
        else:
            logging.info("No GPUs found. Training on CPU.")
            num_towers = 1
            device_string = '/cpu:%d'
        num_towers = num_towers
        device_string = device_string

        #优化器构建
        learning_rate_dict = self.optimizer_config.learning_rate_dict
        optimizer_dict ={}
        optimizer_class = train_util.find_class_by_name(self.optimizer_config.optimizer, [tf.train])
        for k,v in learning_rate_dict.items():
            if k not in self.modal_name_list+['classifier']: #支持不同模块的学习率独立设置
                print("key:" ,k)
                print("!"*50, "Warning: learning rate dict does not match with modal_name_list")
                continue
            learning_rate_dict[k] = tf.train.exponential_decay(v, 
                                                               self.global_step,
                                                               3333,
                                                               #max(1, self.optimizer_config.max_step_num//3), #decay 3 times
                                                               self.optimizer_config.learning_rate_decay, 
                                                               staircase=True) #TODO(jefxiong, 扩展不同学习率衰减方法)
            tf.summary.scalar('lr/{}'.format(k), learning_rate_dict[k])
            optimizer_init_params = self.optimizer_config.optimizer_init_params.copy()
            optimizer_init_params['learning_rate'] = learning_rate_dict[k]
            optimizer_dict[k] = optimizer_class(**optimizer_init_params)

        #模型输入构建(字典形式)
        with tf.name_scope("train_input"):
            raw_inputs_dict = {data_name: self.reader.data_op_list[i] for i,data_name in enumerate(self.reader.dname_string_list)}
            tower_inputs = {key: tf.split(value, num_towers) for key, value in raw_inputs_dict.items()}
            tower_inputs = [{key: value[i] for key, value in tower_inputs.items()} for i in range(num_towers)] #放置到不同GPU
        tower_gradients = []
        
        tower_losses_dict={}
        tower_predictions_dict = {}
        for task_name in self.reader.label_num_dict:
            tower_predictions_dict[task_name] = []

        for i in range(num_towers):
            with tf.device(device_string % i):
                with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
                    with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus != 1 else "/gpu:0")):
                        result_dict = self.model(tower_inputs[i],train_batch_size = self.reader.batch_size//num_towers, is_training=True)

                        #遍历所有分类任务输出
                        for task_name in self.reader.label_num_dict:
                            if task_name in tower_predictions_dict: 
                                tower_predictions_dict[task_name].append(result_dict[task_name+'_output_fusion']["predictions"])
                            else: 
                                tower_predictions_dict[task_name] = [result_dict[task_name+'_output_fusion']["predictions"]]

                        #遍历所有分类任务损失函数
                        loss_fn_dict={}
                        for task_name, task_loss_type in self.optimizer_config.loss_type_dict.items():
                          loss_fn = loss_lib.get_instance(task_loss_type, paramters_dict={}) #TODO(jefxiong, 支持损失函数的额外参数输入)
                          loss_fn_dict[task_name] = loss_fn
                        loss_dict = self.model.build_loss(tower_inputs[i], result_dict, loss_fn_dict)

                        #聚合不同GPU上的loss
                        for key,loss in loss_dict.items():
                          if key in tower_losses_dict: 
                              tower_losses_dict[key].append(loss)
                          else: 
                              tower_losses_dict[key] = [loss]

                        #BN 相关
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        if update_ops:
                            with tf.control_dependencies(update_ops):
                                barrier = tf.no_op(name="gradient_barrier")
                                with tf.control_dependencies([barrier]):
                                    total_loss = sum([v for k,v in loss_dict.items()])
                                    if 'total_loss' not in tower_losses_dict:
                                        tower_losses_dict['total_loss'] = [total_loss] 
                                    else:
                                        tower_losses_dict['total_loss'].append(total_loss)
                        gradients = optimizer_dict['classifier'].compute_gradients(total_loss, colocate_gradients_with_ops=False)
                        tower_gradients.append(gradients)

        merged_gradients = train_util.combine_gradients(tower_gradients)
        if self.optimizer_config.clip_gradient_norm > 0:
            with tf.name_scope('clip_grads'):
                merged_gradients = train_util.clip_gradient_norms(merged_gradients, self.optimizer_config.clip_gradient_norm)

        vars_dict = {} #不同模态和分类器学习权重和字典
        for key in self.modal_name_list:
          vars_dict[key] =  []
        vars_dict['classifier'] = [] #新增分类器独立学习率参数设置
        for grad, var in merged_gradients:
            flag_in_key = False
            print(var.name)
            for key in vars_dict:
              if key in var.name:
                vars_dict[key].append((grad, var))
                flag_in_key = True
                continue
            assert flag_in_key == True
        var_dict_logging_info='|'.join(["{} vars size: {}".format(name, len(value)) for name,value in vars_dict.items()])  
        logging.info('!'*100)
        logging.info(var_dict_logging_info)

        train_op_list=[]
        for key,var in vars_dict.items():
          if len(var)>0:
            if key !='classifier':
                train_op_list.append(optimizer_dict[key].apply_gradients(vars_dict[key], global_step=None))
            else:
                train_op_list.append(optimizer_dict[key].apply_gradients(vars_dict[key], global_step=self.global_step))
        self.train_op = tf.group(*train_op_list)

        for k,v in tower_losses_dict.items():
            tower_losses_dict[k] = tf.reduce_mean(tf.stack(v))
            tf.summary.scalar("TrainLoss/{}".format(k), tower_losses_dict[k])

        #for inp_name,inp_val in raw_inputs_dict.items():
        #    tf.summary.histogram("inputs/{}".format(inp_name), inp_val)

        #添加相关变量到类成员, TODO(jefxiong, 扩展 taggging && classification)
        self.train_inputs_dict = raw_inputs_dict
        self.train_losses_dict = tower_losses_dict
        self.train_tagging_predictions = tf.concat(tower_predictions_dict['tagging'], 0)
        #self.train_classification_predictions = tf.concat(tower_predictions_dict['classification'], 0)

    def build_eval_graph(self):
        input_name_list = self.reader.dname_string_list #模型输入变量名
        inupt_shape_list = self.reader.data_shape_list  #模型输入shape
        input_dtype_list = self.reader.dtype_list       #模型输入类型
        inputs_dict={}

        #定义输入placeholder
        with tf.name_scope("eval_input"):
            for input_name,input_shape,input_dtype in zip(input_name_list, inupt_shape_list, input_dtype_list):
                print("input_name: {}, input_shape:{}, input_dtype: {}".format(input_name, input_shape, input_dtype))
                inputs_dict[input_name] = tf.placeholder(shape=[None]+input_shape, dtype=input_dtype, name=input_name) #add batch size dim

        val_tagging_predictions={}
        with tf.variable_scope("tower", reuse=True):
            result = self.model(inputs_dict, is_training=False)
            for task_name in self.reader.label_num_dict: #[tagging, classification]
                for modal_name in ['fusion'] + self.modal_name_list:
                    val_tagging_predictions[task_name+'_output_'+modal_name] = result[task_name+"_output_"+modal_name]["predictions"]
            loss_fn_dict={}
            for task_name, task_loss_type in self.optimizer_config.loss_type_dict.items():
                loss_fn = loss_lib.get_instance(task_loss_type, paramters_dict={}) #TODO(jefxiong, 支持损失函数的额外参数输入)
                loss_fn_dict[task_name] = loss_fn
            loss_dict = self.model.build_loss(inputs_dict, result, loss_fn_dict)

        for key,item in inputs_dict.items():
          tf.add_to_collection(key, item) ##将需要的模型输入变量放入collection中 e.g. video,video_frames_num, etc.
        
        for pred_name, pred_value in val_tagging_predictions.items():
          tf.add_to_collection(pred_name, pred_value) #将各个分支的预测输出放入collection中 e.g. tagging_pred_video
        
        for loss_name, loss_val in loss_dict.items():
          tf.add_to_collection(loss_name, loss_val) ##将损失函数加入collection中 e.g. tagging_loss_video
        
        tf.add_to_collection("val_summary_op", tf.summary.merge_all())

        self.best_validation_score = -1.0

    def run(self, config_path):
        """训练验证主流程"""
        #训练目录设置
        if self.is_master and self.optimizer_config.start_new_model:
            self.remove_training_directory(self.optimizer_config.train_dir)
        os.makedirs(self.optimizer_config.train_dir, exist_ok=True)
        shutil.copyfile(config_path, self.optimizer_config.train_dir+'/config.yaml')

        #分布式训练设置
        self.target, device_fn = self.start_server_if_distributed()

        #with tf.Graph().as_default() as graph:
        with tf.get_default_graph().as_default() as graph:
            #meta_filename = self.get_meta_filename(self.optimizer_config.start_new_model, self.optimizer_config.train_dir)
            #if meta_filename:
            #    saver = self.recover_model(meta_filename)
            with tf.device(device_fn):
                #if not meta_filename:
                #    saver = self.build_model()
                saver = self.build_model()
                global_init_op = tf.global_variables_initializer()
                dataset_init_op = self.reader.iterator.initializer
                #init_op = tf.group([global_init_op, dataset_init_op])

        self.sv = tf.train.Supervisor(graph=graph,
            logdir = self.optimizer_config.train_dir,
            init_op = global_init_op,
            local_init_op=dataset_init_op,
            is_chief = self.is_master,
            global_step = self.global_step,
            save_model_secs = 0,
            save_summaries_secs = 120,
            saver = saver)
        self.summary_writer = self.sv.summary_writer
        logging.info("%s: Starting managed session.", task_as_string(self.task))

        total_iteration = self.optimizer_config.max_step_num
        with self.sv.managed_session(self.target, config=self.config) as sess:
            try:
                logging.info("%s: Entering training loop.", task_as_string(self.task))
                while not self.sv.should_stop():
                    try:
                        batch_start_time = time.time()
                        train_fetch_dict_eval = sess.run(self.train_fetch_dict)
                        global_step_val = train_fetch_dict_eval['global_step']
                        train_losses_dict = train_fetch_dict_eval['train_losses_dict']
                        if global_step_val>total_iteration:
                            logging.info("step limit reached")
                            break

                        seconds_per_batch = time.time() - batch_start_time
                        examples_per_second = self.reader.batch_size / seconds_per_batch
                        if self.is_master and train_fetch_dict_eval['global_step'] % 10 == 0 and self.optimizer_config.train_dir:
                            self.train_metric_log(train_fetch_dict_eval, examples_per_second)
                            time_to_export = global_step_val % self.optimizer_config.export_model_steps == 0
                            if self.is_master and time_to_export:
                                valid_sample_generator_dict =  self.reader.get_valid_sample_generator_dict()
                                val_score_dict={}
                                for source_name, generator in valid_sample_generator_dict.items():
                                  val_score = self.eval(sess, global_step_val, generator, source_name) # 在验证集上测试模型
                                  val_score_dict[source_name] = val_score
                                  print("validation score on {} is : {:.4f}".format(source_name, val_score))
                                self.export_model(global_step_val, sess, val_score_dict)
                        else:
                            train_losses_info = "|".join(["{}: {:.3f}".format(k, v) for k,v in train_losses_dict.items()])
                            print("training step {} | {} | {:.2f} Examples/sec".format(global_step_val, train_losses_info, examples_per_second))
                    except tf.errors.DataLossError:
                        logging.info("ERROR: corrupted input tfrecord")
            except tf.errors.OutOfRangeError:
                logging.info("%s: Done training -- step limit reached.",
                             task_as_string(self.task))
            except:
                traceback.print_exc()

        logging.info("%s: Exited training loop.", task_as_string(self.task))
        self.sv.stop()

    def build_model(self):
        self.build_train_graph()        
        self.build_eval_graph()
        self.load_pretrained_model()
        
        self.train_fetch_dict = self.get_train_fetch_dict()
        self.val_fetch_dict = self.get_val_fetch_dict()
        
        return tf.train.Saver(max_to_keep=5)

    def export_model(self, global_step_val, session, validation_score_dict):
        # TODO(jefxiong, 改进对不同来源验证集上指标求平均策略)
        validation_score = sum(validation_score_dict.values()) 
        if validation_score <= self.best_validation_score:
            return
        self.best_validation_score = validation_score
        last_checkpoint = self.sv.saver.save(session, self.sv.save_path, global_step_val)
        model_dir = "{}/export/step_{}_{:.4f}".format(self.optimizer_config.train_dir, global_step_val, validation_score)        
        self.model_export_pb.export_model(model_dir=model_dir, global_step_val=global_step_val, last_checkpoint=last_checkpoint)       
        

    def start_server_if_distributed(self):
        if self.cluster:
            logging.info("%s: Starting trainer within cluster %s.",
                         task_as_string(self.task), self.cluster.as_dict())
            server = start_server(self.cluster, self.task)
            target = server.target
            device_fn = tf.train.replica_device_setter(
                ps_device="/job:ps",
                worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
                cluster=self.cluster)
        else:
            target = ""
            device_fn = ""
        return (target, device_fn)

    def remove_training_directory(self, train_dir):
        try:
            logging.info(
                "%s: Removing existing train directory.",
                task_as_string(self.task))
            gfile.DeleteRecursively(train_dir)
        except:
            logging.error(
                "%s: Failed to delete directory " + train_dir +
                " when starting a new model. Please delete it manually and" +
                " try again.", task_as_string(self.task))

    def get_meta_filename(self, start_new_model, train_dir):
        if start_new_model:
            logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                         task_as_string(self.task))
            return None

        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if not latest_checkpoint:
            logging.info("%s: No checkpoint file found. Building a new model.",
                         task_as_string(self.task))
            return None

        meta_filename = latest_checkpoint + ".meta"
        if not gfile.Exists(meta_filename):
            logging.info("%s: No meta graph file found. Building a new model.",
                         task_as_string(self.task))
            return None
        else:
            return meta_filename

    def recover_model(self, meta_filename):
        logging.info("%s: Restoring from meta graph file %s",
                     task_as_string(self.task), meta_filename)
        return tf.train.import_meta_graph(meta_filename)

#训练流程
def train_main(config_path, TrainerType):
    config = yaml.load(open(config_path))
    print(config)

    env = json.loads(os.environ.get("TF_CONFIG", "{}"))
    cluster_data = env.get("cluster", None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get("task", None) or {"type": "master", "index": 0}
    task = type("TaskSpec", (object,), task_data)

    # Logging the version.
    logging.set_verbosity(tf.logging.INFO)
    logging.info("%s: Tensorflow version: %s.",task_as_string(task), tf.__version__)

    # Dispatch to a master, a worker, or a parameter server.
    if not cluster or task.type == "master" or task.type == "worker":
        model = models.get_instance(config['ModelConfig']['model_type'],
                                    config['ModelConfig'])
        reader = dataloader.Data_Pipeline(config['DatasetConfig'])
        TrainerType(cluster, task, model, reader, config).run(config_path=config_path)
    elif task.type == "ps":
        ParameterServer(cluster, task).run()
    else:
        raise ValueError("%s: Invalid task_type: %s." %(task_as_string(task), task.type))
