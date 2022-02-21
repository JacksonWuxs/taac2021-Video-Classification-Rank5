import numpy as np
import codecs

def extract_dict(dict_file):
    index_to_tag = {}
    tag_to_index = {}
    for i, line in enumerate(codecs.open(dict_file, 'r', encoding='utf-8')):
        line = line.strip()
        if '\t' in line:
            index, tag = line.split('\t')[:2]
        elif ' ' in line:
            index, tag = i, line.rsplit(' ', 1)[0]
        else:
            index, tag = i, line

        try:
            index = int(index)
        except:
            index, tag = int(tag), index

        index_to_tag[index] = tag
        tag_to_index[tag] = index
    return index_to_tag, tag_to_index

class Preprocess_index_indentity:
    
    def __init__(self,
                 index_dict,
                 label_num,
                 sep_token=',',
                 is_training=False):
        self.index_to_tag,self.tag_to_index = extract_dict(index_dict)
        self.label_num = label_num
        self.sep_token = sep_token
        self.is_training = is_training

    def __call__(self, index_str):
        index_lst = index_str.split(self.sep_token)
        index_lst = [int(index) for index in index_lst]
        for index in index_lst:
            assert index in self.index_to_tag
        return np.array(index_lst).astype('int32')

class Preprocess_index_sparse_to_dense:
        
    def __init__(self,
                 index_dict,
                 sep_token=',',
                 is_training=False):
        self.index_to_tag,self.tag_to_index = extract_dict(index_dict)
        self.sep_token = sep_token
        self.is_training = is_training
        self.max_index = 0
        for index in self.index_to_tag:
            self.max_index = max(index, self.max_index)
        self.seq_size = self.max_index + 1
        self.label_num = self.seq_size

    def __call__(self, index_str):
        dense_array = np.zeros(self.seq_size)
        index_lst = index_str.split(self.sep_token)
        index_lst = [int(index) for index in index_lst]
        for index in index_lst:
            if index == -1:
               continue
            assert index in self.index_to_tag
            dense_array[index] = 1.0
        return dense_array.astype('float32')

class Preprocess_label_sparse_to_dense:
        
    def __init__(self,
                 index_dict,
                 sep_token=',',
                 is_training=False):
        self.index_to_tag,self.tag_to_index = extract_dict(index_dict)
        self.sep_token = sep_token
        self.is_training = is_training
        self.max_index = 0
        for index in self.index_to_tag:
            self.max_index = max(index, self.max_index)
        self.seq_size = self.max_index + 1
        self.label_num = self.seq_size

    def __call__(self, index_str, augment):
        dense_array = np.zeros(self.seq_size)
        label_lst = index_str.split(self.sep_token)
        for label in label_lst:
            if label in self.tag_to_index:
                index = self.tag_to_index[label]
                dense_array[index] = 1.0
        return dense_array.astype('float32')
