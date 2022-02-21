import os
import sys
import random
import numpy as np
import yaml
import linecache
import importlib
from tomorrow3 import threads

class Data_Generator:
    
    def __init__(self,
                 data_config):
        self.data_config = data_config
        self.shuffle = self.data_config['shuffle']
        self.feature_config = self.data_config['preprocess_config']['feature']
        self.feature_num_per_sample = len(self.feature_config)
        self.label_config = self.data_config['preprocess_config']['label']
        self.label_num_per_sample = len(self.label_config)
        self.index_to_input_name = {}
        self.label_num_dict = {}
        input_index = -1
        self.dname_string_list = []
        self.dtype_string_list = []
        self.data_shape_list = []
        for tmp_config in [self.feature_config, self.label_config]:
            for data_part in tmp_config:
                name = data_part['name']
                shape = data_part['shape']
                input_index += 1
                self.index_to_input_name[input_index] = name
                dtype_str = data_part['dtype']
                output_name_spt = name.split(',')
                output_dtype_spt = dtype_str.split(',')
                self.dname_string_list += output_name_spt
                self.dtype_string_list += output_dtype_spt
                self.data_shape_list +=shape

        # +1 blank line for seperate
        self.data_num_per_sample = self.feature_num_per_sample + self.label_num_per_sample + 1
        print(self.data_num_per_sample)

        self.train_data_source_list = self.data_config['train_data_source_list']
        for source_name in self.train_data_source_list:
            fn = self.train_data_source_list[source_name]['file']
            sample_count = self.fn_sample_count(fn)
            print('Train Source sample_count: ',source_name,sample_count)
            self.train_data_source_list[source_name]['sample_count'] = sample_count
            batch_size = self.train_data_source_list[source_name]['batch_size']
            batch_num = max(1, sample_count // batch_size)
            print('Train Source batch_num: ',source_name,batch_num)
            self.train_data_source_list[source_name]['batch_num'] = batch_num

        self.valid_data_source_list = self.data_config['valid_data_source_list']
        for source_name in self.valid_data_source_list:
            fn = self.valid_data_source_list[source_name]['file']
            sample_count = self.fn_sample_count(fn)
            print('Valid Source: ', source_name,sample_count)
            self.valid_data_source_list[source_name]['sample_count'] = sample_count
            batch_size = self.valid_data_source_list[source_name]['batch_size']
            batch_num = max(1, sample_count // batch_size)
            print('Valid Source batch_num: ',source_name,batch_num)
            self.valid_data_source_list[source_name]['batch_num'] = batch_num


        self.train_preprocess = self.get_preprocess_function(is_training=True)
        self.valid_preprocess = self.get_preprocess_function(is_training=False)

    def fn_sample_count(self, fn):
        line_count = 0
        for l in open(fn):
            line_count += 1
        assert line_count % self.data_num_per_sample == 0, "line_count: {} , data_num_per_sample: {}".format(line_count, self.data_num_per_sample)
        sample_count = line_count / self.data_num_per_sample
        sample_count = int(sample_count)
        return sample_count

    def get_single_sample_gen(self, data_source, preprocess_function, clip_batch=True):
        filename = data_source['file']
        count = data_source['sample_count']
        batch_size = data_source['batch_size']
        if clip_batch:
           count = count - (count%batch_size)
        while True:
            index_lst = list(range(0, count))
            if self.shuffle:
               random.shuffle(index_lst)
            result_list_queue = []
            for i in index_lst:
                return_list = []
                for line_i in range(self.data_num_per_sample*i+1,
                               self.data_num_per_sample*(i+1)):
                    line = linecache.getline(filename, line_i)
                    line = line.strip('\r\n')
                    return_list.append(line)
                result_list = preprocess_function(*return_list)
                result_list_queue.append(result_list)

                if len(result_list_queue) == 50:
                   for result_list in result_list_queue:
                       yield result_list.result()
                   result_list_queue = []
            
            for result_list in result_list_queue:
               yield result_list.result()
            result_list_queue = []
            
    def get_batch_generator(self, generator,batch_num, batch_size, return_dict=True):
        batch_sample_list = []
        for _ in range(batch_num*batch_size):
            sample = generator.__next__()
            batch_sample_list.append(sample)
            if len(batch_sample_list) == batch_size:
               batch_sample = []
               data_size = len(batch_sample_list[0])
               for data_i in range(data_size):
                   data_i_batch = []
                   for batch_i in range(batch_size):
                       data_i_batch.append(batch_sample_list[batch_i][data_i])
                   data_i_batch = np.array(data_i_batch)
                   batch_sample.append(data_i_batch)
               if return_dict:
                  batch_sample = {name:data for name,data in zip(self.dname_string_list,batch_sample)}
               yield batch_sample
               batch_sample_list = []

    def get_train_sample_generator(self):
        self.train_source_generator = {}
        for source_name in self.train_data_source_list:
            self.train_source_generator[source_name] = self.get_single_sample_gen(self.train_data_source_list[source_name],
                                                                                  self.train_preprocess)

        while True:
            for source_name in self.train_data_source_list:
                source_batch_size = self.train_data_source_list[source_name]['batch_size']
                for _ in range(source_batch_size):
                    return_list = self.train_source_generator[source_name].__next__()
                    yield return_list

    def get_valid_sample_generator_dict(self):
        self.valid_source_generator_dict = {}
        for source_name in self.valid_data_source_list:
            generator = self.get_single_sample_gen(self.valid_data_source_list[source_name],
                                                   self.valid_preprocess)
            source_batch_num = self.valid_data_source_list[source_name]['batch_num']
            source_batch_size = self.valid_data_source_list[source_name]['batch_size']
            self.valid_source_generator_dict[source_name] = self.get_batch_generator(generator=generator,
                                                                           batch_num=source_batch_num,
                                                                           batch_size=source_batch_size)
        return self.valid_source_generator_dict

    def get_preprocess_function(self,is_training):
        root = self.data_config['preprocess_root']
        sys.path.append(root)
        index_to_preprocess = []

        for data_part in self.feature_config:
            package_name, preprocess_class_name = data_part['class'].split('.')
            if 'extra_args' in data_part:
                init_args = data_part['extra_args']
            else:
                init_args = {}
            init_args['is_training'] = is_training
            preprocess_module = importlib.import_module(package_name)
            Preprocess_Class = getattr(preprocess_module, preprocess_class_name)
            preprocess_instance = Preprocess_Class(**init_args)
            index_to_preprocess.append(preprocess_instance)

        for data_part in self.label_config:
            package_name, preprocess_class_name = data_part['class'].split('.')
            if 'extra_args' in data_part:
                init_args = data_part['extra_args']
            else:
                init_args = {}
            preprocess_module = importlib.import_module(package_name)
            Preprocess_Class = getattr(preprocess_module, preprocess_class_name)
            preprocess_instance = Preprocess_Class(**init_args)
            name = data_part['name']
            self.label_num_dict[name] = preprocess_instance.label_num
            index_to_preprocess.append(preprocess_instance)
        self.index_to_preprocess = index_to_preprocess
        
        @threads(20)
        def preprocess_fn(*args):
            preprocess_data_list = []
            for index, data in enumerate(args):
                augment = random.choice([0] * 7 + [1, 1, 2]) if is_training else 0
                preprocess_data = self.index_to_preprocess[index](data, augment)
                if isinstance(preprocess_data,np.ndarray):
                     preprocess_data_list.append(preprocess_data)
                elif isinstance(preprocess_data, tuple):
                     for preprocess_data_element in preprocess_data:
                         preprocess_data_list.append(preprocess_data_element)
            return tuple(preprocess_data_list)

        return preprocess_fn


if __name__ == '__main__':
   import argparse
   import time
   #import cProfile

   parser = argparse.ArgumentParser()
   parser.add_argument('--data_config',type=str)
   args = parser.parse_args()

   data_config = yaml.load(open(args.data_config))
   data_generator =  Data_Generator(data_config = data_config['DatasetConfig'])
   train_sample_generator =  data_generator.get_train_sample_generator()
   def train():
       time_list_sum = 0
       time_count = 0
       for _ in range(10):
           start_time = time.time()
           sample = train_sample_generator.__next__()
           end_time = time.time()
           time_list_sum += (end_time-start_time)
           time_count += 1
           print(time_count,np.mean(time_list_sum)/time_count)
           for x,output_name in zip(sample,data_generator.dname_string_list):
               print(x, output_name)

   def valid():
       valid_sample_generator_dict =  data_generator.get_valid_sample_generator_dict()
       for source_name,generator in valid_sample_generator_dict.items():
           for sample in generator:
               for output_name, x in sample.items():
                   print('valid', output_name,x)
   #cProfile.run('test()')
   #print(data_config)
   train()
   valid()
   train()
   valid()
