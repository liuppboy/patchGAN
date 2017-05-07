# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.contrib import slim
from layers import batch_norm

def basic_resnet(x, out_channels, kernel_size=[3, 3], first_activation=tf.nn.relu, last_activation=tf.nn.relu, 
                 preactive=True, train=True, downsample=False, projection=True, reuse=None, name='resnet'):
    """ Basic Resnet
    
        It can either be preactive or not.
        If not preactive and last_activation is None, then assume that last batch normalization is also None
    
    # Arguments
            
        
    """    
    
    with tf.variable_scope(name, reuse=reuse):        
        first_stride = 2 if downsample else 1
        res = x        
        if preactive:
            batch_norm_1 = batch_norm(x, is_train=train)
            act = first_activation(batch_norm_1)
            conv_1 = slim.conv2d(act, out_channels, kernel_size=kernel_size, stride=first_stride, 
                                 activation_fn=last_activation, normalizer_fn=batch_norm, normalizer_params={'is_train': train}, trainable=train, scope='conv_1')
            
            conv_2 = slim.conv2d(conv_1, out_channels, kernel_size=kernel_size, stride=1, activation_fn=None, 
                                 trainable=train, scope='conv_2')
        else:
            conv_1 = slim.conv2d(x, out_channels, kernel_size=kernel_size, stride=first_stride, activation_fn=first_activation, 
                                 normalizer_fn=batch_norm, normalizer_params={'is_train': train}, trainable=train, scope='conv_1')
            
            last_normalizier = batch_norm if last_activation else None
            conv_2 = slim.conv2d(conv_1, out_channels, kernel_size=kernel_size, stride=1, activation_fn=last_activation, 
                                 normalizer_fn=last_normalizier, normalizer_params={'is_train': train}, trainable=train, scope='conv_2') 
        
        
        if projection:
            res = slim.conv2d(x, out_channels, kernel_size=[1, 1], stride=first_stride, activation_fn=None, scope='residual')
        else:
            in_channels = x.get_shape().as_list()[-1]
            if in_channels != out_channels:
                ch = (out_channels - in_channels) // 2
                res = tf.pad(res, [[0, 0], [0, 0], [0, 0], [ch, ch]])
        res = tf.add(res, conv_2) 
        
        return res
            
              
def bottleneck(x, channels, strides=[1, 1, 1], rate=1, activation_fn=tf.nn.relu, train=True, scope='bottleneck'):
    with tf.variable_scope(scope):
        in_channels = x.get_shape().as_list()[-1]
        if in_channels != channels[-1] or max(strides) != 1:
            shortcut = slim.conv2d(x, channels[-1], [1, 1], max(strides), activation_fn=None, trainable=train, scope='shortcut')
        else:
            shortcut = x
        
        with slim.arg_scope([slim.conv2d], activation_fn=activation_fn, trainable=train, padding='SAME', 
                            normalizer_fn=batch_norm, normalizer_params={'is_train': train}):
            conv1 = slim.conv2d(x, channels[0], [1, 1], strides[0], scope='conv1')
            conv2 = slim.conv2d(conv1, channels[1], [3, 3], strides[1], rate=rate, scope='conv2')
            conv3 = slim.conv2d(conv2, channels[2], [1, 1], strides[2], activation_fn=None, scope='conv3')
        
        resnet = activation_fn(tf.add(shortcut, conv3), name='activation')
    
    return resnet
            
                
                
            
            
            
        
    