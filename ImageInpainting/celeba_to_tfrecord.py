import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as misc
import glob
import sys

sys.path.append('../')
from tf_reader import BasicTFRecord

file_list = glob.glob('/home/ppliu/python/DCGAN-tensorflow/data/img_align_celeba/*.jpg')
file_list = sorted(file_list)
file_list = np.array(file_list)
#np.random.seed(0)
#perm = np.random.permutation(len(file_list))
#train_id = perm[:200000]
#test_id = perm[200000:]
#np.savez('train_test_split.npz', train=train_id, test=test_id)
train_id = np.load('../datasets/celeba/train.npy')
train_list = file_list[train_id]
test_id = np.load('../datasets/celeba/test.npy')
test_list = file_list[test_id]
tfrecorder = BasicTFRecord()
writer = tf.python_io.TFRecordWriter('../datasets/celeba/train.tfrecords')
for filename in train_list:
    img = misc.imread(filename)
    img = np.array(img)
    img = np.expand_dims(img, 0)
    tfrecorder.to_record(writer, [img], ['byte'], ['img'] )

writer.close()
    

