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

def read_mnist_feed(data_dir, dtype=np.float32, reshape=True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_dataset = input_data.read_data_sets(data_dir, dtype=dtype, reshape=reshape, validation_size=0)
    return mnist_dataset.train

def read_and_decode_mnist(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'images': tf.FixedLenFeature([], tf.string),
            #'labels': tf.FixedLenFeature([], tf.int64), 
        })

    image = tf.decode_raw(features['images'], tf.uint8)
    #label = tf.cast(features['labels'], tf.int32)
    return image

def generate_batch(filename_queue, min_queue_examples, batch_size, num_threads=8, shuffle=True):
    image = read_and_decode_mnist(filename_queue)
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    image = tf.reshape(image, [28, 28, 1])
    if shuffle:
        img_batch = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_threads,
            min_after_dequeue=min_queue_examples)
    else:
        img_batch, depth_batch, mask_batch = tf.train.batch(
            [image],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_threads,
            min_after_dequeue=min_queue_examples) 
    return img_batch
    


#filename_queue  = tf.train.string_input_producer(['./datasets/mnist/train.tfrecords'], shuffle=False)
#image, label = read_and_decode_mnist(filename_queue)
#with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #coord=tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #for i in range(10):
        #img_raw, label_raw = sess.run([image, label])
        #key = sess.run(k)
        ##img_raw = Image.fromarray(img_raw)
        ##img_raw.show()
    
    #coord.request_stop()
    #coord.join(threads)


