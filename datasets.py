# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class DataFeeding(object):
    def __init__(self, features, shuttle=True, epochs_completed=0, index_in_epoch=0):
        '''Reading data using feeding (placeholder) method
        Arguments:
            features: list of features or one feature
        
        '''  
        self.features = features
        self.shuttle = shuttle
        self.is_features_list = isinstance(features, list)
        
        self.data_num = features[0].shape[0]  if self.is_features_list else features.shape[0]
        
        self.epochs_completed = epochs_completed
        self.index_in_epoch = index_in_epoch
    
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.data_num:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            if self.shuttle:
                perm = np.arange(self.data_num)
                np.random.shuffle(perm)
                
                self.features = [feature[perm] for feature in self.features] if self.is_features_list else self.features[perm]
                
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.data_num
        end = self.index_in_epoch
        return [feature[start:end] for feature in self.features] if self.is_features_list else self.features[start:end]    

def read_mnist_feed(data_dir='./datasets/mnist', dtype=np.float32, reshape=True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_dataset = input_data.read_data_sets(data_dir, dtype=dtype, reshape=reshape, validation_size=0)
    return mnist_dataset


