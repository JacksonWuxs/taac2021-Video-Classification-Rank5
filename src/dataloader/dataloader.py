# coding=utf-8
import yaml
import tensorflow as tf
import os,sys
sys.path.append(os.getcwd())
from src.dataloader.data_generator import Data_Generator

class Data_Pipeline:

    def __init__(self, data_config):

        self.data_config = data_config
        self.batch_size = data_config['batch_size']
        self.data_generator = Data_Generator(data_config=self.data_config)
        self.sample_generator = self.data_generator.get_train_sample_generator
        self.get_valid_sample_generator_dict = self.data_generator.get_valid_sample_generator_dict
        self.label_num_dict = self.data_generator.label_num_dict
        self.dname_string_list = self.data_generator.dname_string_list
        self.data_shape_list = self.data_generator.data_shape_list

        self.data_num = len(self.dname_string_list)
        self.dtype_map_dict = {'bool':tf.bool,
                               'int16':tf.int16,
                               'int32': tf.int32,
                               'int64': tf.int64,
                               'float16':tf.float16,
                               'float32': tf.float32,
                               'float64': tf.float64,
                                'string': tf.string}
        self.dtype_list = [self.dtype_map_dict[string] for string in self.data_generator.dtype_string_list]
        self.dataset = tf.data.Dataset.from_generator(self.sample_generator,
                                                           tuple(self.dtype_list),
                                                           tuple(self.data_shape_list))
        self.dataset = self.dataset.batch(self.batch_size).prefetch(20)
        self.iterator = self.dataset.make_initializable_iterator()
        self.data_op_lst = self.iterator.get_next()
        self.name_to_data_op = {}
        self.data_op_list = []
        for index in range(self.data_num):
            name = self.dname_string_list[index]
            self.name_to_data_op[name] = self.data_op_lst[index]
            self.data_op_list.append(self.name_to_data_op[name])

if __name__ == '__main__':
   import argparse
   import time

   parser = argparse.ArgumentParser()
   parser.add_argument('--data_config',type=str)
   args = parser.parse_args()
   
   data_config = yaml.load(open(args.data_config))
   data_pipeline =  Data_Pipeline(data_config = data_config['DatasetConfig'])

   for name in data_pipeline.name_to_data_op:
       print(name)
       print(data_pipeline.name_to_data_op[name])

   Sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
   Sess_config.gpu_options.allow_growth = True
   with tf.Session(config=Sess_config) as sess:
       sess.run(data_pipeline.iterator.initializer)
       sess.run(tf.local_variables_initializer())
       sess.run(tf.global_variables_initializer())
       for _ in range(10):
           print(data_pipeline.label_num_dict)
           start_time = time.time() 
           data_list = sess.run(data_pipeline.data_op_list)
           for data,name in zip(data_list,data_pipeline.dname_string_list):
               print(name,data.shape)
           #time.sleep(0.5)
           end_time = time.time()
           print(end_time-start_time)

   def valid():
       valid_sample_generator_dict =  data_pipeline.get_valid_sample_generator_dict()
       for source_name,generator in valid_sample_generator_dict.items():
           for sample in generator:
               for output_name, x in sample.items():
                   print(source_name, output_name,x.shape)
   valid()
