# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from activations import lrelu

def batch_norm(x, train=True, reuse=None, decay=0.999, scale=True, epsilon=1e-3, scope='batch_norm'):
    return slim.batch_norm(x, decay=decay, scale=scale, epsilon=epsilon,  updates_collections=None,
                is_training=train, reuse=reuse, scope=scope)

def upsampling_d2s(inputs, num_outputs, kernel_size=3, stride=2, 
                   padding='SAME', activation_fn=lrelu, normalizer_fn=None, 
                   normalizer_params=None, reuse=None, trainable=True, scope=None):
    with tf.variable_scope(scope, reuse=reuse):
        conv = slim.conv2d(inputs, num_outputs*stride*stride, kernel_size=kernel_size, padding=padding, stride=1,
                           activation_fn=activation_fn, normalizer_fn=normalizer_fn, 
                           normalizer_params=normalizer_params, trainable=trainable, scope='conv')
        upsampling = tf.depth_to_space(conv, stride, name='upsampling')
    
    return upsampling




        