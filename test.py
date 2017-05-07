import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tf_reader import BasicTFRecord

dataset = mnist.read_data_sets('./dataset/mnist', dtype=tf.uint8, validation_size=0)
train_x = dataset.train.images
train_y = dataset.train.labels

test_x = dataset.test.images
test_y = dataset.test.labels

tfrecorder = BasicTFRecord()
writer = tf.python_io.TFRecordWriter('./dataset/mnist/train.tfrecords')
tfrecorder.to_record(writer, [train_x, train_y], ['byte', 'int'], ['images', 'labels'])
writer.close()
