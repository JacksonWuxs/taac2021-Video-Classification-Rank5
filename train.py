#coding=utf-8
#Author: jefxiong@tencent.com
#Author: xxx@tencent.com

import sys,os
sys.path.append(os.getcwd())
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import logging
import tensorflow.contrib.slim as slim
import utils.train_util as train_util
from utils.base_trainer import Trainer, train_main



class TaggingTrainer(Trainer):
    def __init__(self, cluster, task, model, reader, configs):
        super().__init__(cluster, task, model, reader, configs)

    def get_train_fetch_dict(self):
        fetch_dict = {}
        fetch_dict['global_step'] = self.global_step
        fetch_dict['train_losses_dict'] = self.train_losses_dict
        fetch_dict['trian_op'] = self.train_op
        #标签任务相关变量
        fetch_dict['train_tagging_predictions'] = self.train_tagging_predictions
        fetch_dict['train_tagging_labels'] = self.train_inputs_dict['tagging']
        return fetch_dict

    def get_val_fetch_dict(self):
        fetch_dict = {}
        for modal_name in ['fusion'] + self.modal_name_list:
            fetch_dict['tagging_output_'+modal_name] = tf.get_collection('tagging_output_'+modal_name)[0]
            fetch_dict['tagging_loss_'+modal_name] = tf.get_collection('tagging_loss_'+modal_name)[0]
            fetch_dict['val_summary_op'] = tf.get_collection("val_summary_op")[0]
        return fetch_dict

    def load_pretrained_model(self):
        text_pretrained_model = self.optimizer_config.pretrained_model['text_pretrained_model']
        assignment_map, _ = train_util.get_assignment_map_from_checkpoint(tf.global_variables(), 
                                                                         text_pretrained_model,
                                                                         var_prefix='tower/text/',
                                                                         show=True)
        tf.train.init_from_checkpoint(text_pretrained_model, assignment_map)
        print("load text_pretrained_model: {}".format(text_pretrained_model))

    def train_metric_log(self, train_fetch_dict_val, examples_per_second):
        """训练集上的结果验证和训练指标tensorboard输出"""

        predictions_val = train_fetch_dict_val['train_tagging_predictions']
        labels_val = train_fetch_dict_val['train_tagging_labels']
        global_step_val = train_fetch_dict_val['global_step']
        train_losses_dict = train_fetch_dict_val['train_losses_dict']

        train_pr_calculator = train_util.PRCalculator()
        gap = train_util.calculate_gap(predictions_val, labels_val)
        train_pr_calculator.accumulate(predictions_val, labels_val)
        precision_at_1 = train_pr_calculator.get_precision_at_conf(0.1)
        precision_at_5 = train_pr_calculator.get_precision_at_conf(0.5)
        recall_at_1 = train_pr_calculator.get_recall_at_conf(0.1)
        recall_at_5 = train_pr_calculator.get_recall_at_conf(0.5)
        train_pr_calculator.clear()
        
        train_losses_info = "|".join(["{}: {:.3f}".format(k, v) for k,v in train_losses_dict.items()])
        logging.info("training step {} |{} | Examples/sec: {:.2f}".format(global_step_val, train_losses_info, examples_per_second))
        logging.info("GAP: {:.2f} | precision@0.1: {:.2f} | precision@0.5: {:.2f} |recall@0.1: {:.2f} | recall@0.5: {:.2f}".format(gap,
                                                                             precision_at_1, precision_at_5,recall_at_1, recall_at_5))
        
        self.sv.summary_writer.add_summary(train_util.MakeSummary("TrainMetric/GAP", gap), global_step_val)
        self.sv.summary_writer.add_summary(train_util.MakeSummary("TrainMetric/precision@0.1", precision_at_1), global_step_val)
        self.sv.summary_writer.add_summary(train_util.MakeSummary("TrainMetric/precision@0.5", precision_at_5), global_step_val)
        self.sv.summary_writer.add_summary(train_util.MakeSummary("TrainMetric/recall@0.1", recall_at_1), global_step_val)
        self.sv.summary_writer.add_summary(train_util.MakeSummary("TrainMetric/recall@0.5", recall_at_5), global_step_val)
        self.sv.summary_writer.flush()

    def eval(self, sess, global_step_val, data_generater, data_source_name):
        #taggging eval
        tagging_class_num = self.reader.label_num_dict['tagging']
        self.evl_metrics = [train_util.EvaluationMetrics(tagging_class_num, top_k=20)
                           for i in range(len(self.modal_name_list)+1)] #+1 for fusion
        for i in range(len(self.evl_metrics)):
            self.evl_metrics[i].clear()

        examples_processed = 0
        
        for sample in data_generater:
          batch_start_time = time.time()
          feed_dict_data = {}
          for input_name in self.reader.dname_string_list:
            var_names = tf.get_collection(input_name)
            assert len(var_names)==1
            feed_dict_data[var_names[0]] = sample[input_name]
            
            
          fetch_dict_eval = sess.run(self.val_fetch_dict, feed_dict=feed_dict_data)
          seconds_per_batch = time.time() - batch_start_time
          example_per_second = self.reader.batch_size / seconds_per_batch
          examples_processed += self.reader.batch_size

          for index, modal_name in enumerate(self.modal_name_list+['fusion']):
            pred = fetch_dict_eval['tagging_output_'+modal_name]
            val_label = sample['tagging']
            gap = train_util.calculate_gap(pred, val_label)
            iteration_info_dict = self.evl_metrics[index].accumulate(pred, val_label, fetch_dict_eval['tagging_loss_'+modal_name])
            iteration_info_dict['GAP'] = gap
          iteration_info_dict["examples_per_second"] = example_per_second
          iterinfo = "|".join(["{}: {:.3f}".format(k,v) for k,v in iteration_info_dict.items()])
          logging.info("examples_processed: %d | %s", examples_processed, iterinfo)
        logging.info("Done with batched inference. Now calculating global performance metrics.")

        for index, modal_name in enumerate(self.modal_name_list+['fusion']):
            epoch_info_dict = self.evl_metrics[index].get()
            epoch_info_dict["epoch_id"] = global_step_val
            epochinfo = train_util.FormatEvalInfo(self.summary_writer, global_step_val, epoch_info_dict, prefix="val_"+modal_name)
            logging.info(epochinfo)
            self.evl_metrics[index].clear()
        self.summary_writer.add_summary(fetch_dict_eval['val_summary_op'], global_step_val)

        return epoch_info_dict['gap'] #融合特征的预测结果


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/config.example.yaml',type=str)
    args = parser.parse_args()
    train_main(args.config, TaggingTrainer)
