# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class BasicTFRecord(object):
    def __init__(self):
        self.to_type = {'int': self.int64_feature, 
                        'float': self.float_feature,
                        'byte': self.bytes_feature}
    
    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def bytes_feature(self, value):
        value = value.tobytes()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    
    def to_record(self, writer, feature_list, save_types, feature_keys, tf_record_name=None):
        """ convert data to tfrecored 
        Arguments:
            feature_list: list of numpy array, the features to be convered, the first dimension must same
            save_types: list of save types, can be 'int', 'float', 'byte'
            feature_keys: features in tfrecord are stored in dict, the key for the dict
            tf_record_name: tfrecord file name
        """
        #writer = tf.python_io.TFRecordWriter(tf_record_name)
        
        feature_num = feature_list[0].shape[0]
        
        features_dict = {}
        for i in range(feature_num):
            for j , feature in enumerate(feature_list):
                features_dict[feature_keys[j]] = self.to_type[save_types[j]](feature[i])
            
            example = tf.train.Example(features=tf.train.Features(feature=features_dict))
            writer.write(example.SerializeToString())
        
        #writer.close()
    



