import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim

x = tf.placeholder(tf.float32, shape=[None, 3, 3, 1])
y = slim.conv2d_transpose(x, 64, kernel_size=3, stride=2)
a
