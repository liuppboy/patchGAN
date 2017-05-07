# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from six.moves import xrange
import time
import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.contrib import slim

from nets import vgg16
from resnet import basic_resnet
from layers import batch_norm
from activation import lrelu
from nyu_v2_tfreader import generate_batch


class patchGAN(object):
    def __init__(self, sess, batch_size=4, dataset_name='nyu_depth_v2', sample_size=100):
        self.sess = sess
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        tfrecord_name = [r'E:\nyudepth\tfrecord\test_list\classroom_0001.tfrecord', r'E:\nyudepth\tfrecord\test_list\dining_room_0026.tfrecord']
        filename_queue  = tf.train.string_input_producer(tfrecord_name, shuffle=True)
        self.img_batch, self.depth_batch, self.norm_batch, self.mask_batch = generate_batch(filename_queue, min_queue_examples=1000, batch_size=batch_size)         
        
    def encoder(self, x, train=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as scope:
            net = {}
            net['resnet_1'] = basic_resnet(x, 16, [3, 3], first_activation=lrelu, last_activation=None, 
                                        preactive=False, train=train, downsample=False, projection=True, name='resnet_1')
            net['resnet_2'] = basic_resnet(net['resnet_1'], 32, [3, 3], first_activation=lrelu, last_activation=lrelu, 
                                        preactive=True, train=train, downsample=True, projection=True, name='resnet_2')
            net['resnet_3'] = basic_resnet(net['resnet_2'], 64, [3, 3], first_activation=lrelu, last_activation=lrelu, 
                                        preactive=True, train=train, downsample=True, projection=True, name='resnet_3')
            net['resnet_4'] = basic_resnet(net['resnet_3'], 128, [3, 3], first_activation=lrelu, last_activation=lrelu, 
                                        preactive=True, train=train, downsample=True, projection=True, name='resnet_4')   
            net['resnet_5'] = basic_resnet(net['resnet_4'], 256, [3, 3], first_activation=lrelu, last_activation=lrelu, 
                                        preactive=True, train=train, downsample=True, projection=True, name='resnet_5')   

        return net
    
    def generator(self, x, train=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse) as scope:
            net = {}
            net['dconv_1'] = slim.conv2d_transpose(x, 128, [3, 3], stride=2, activation_fn=lrelu, 
                                            normalizer_fn=batch_norm, trainable=train, scope='deconv_1')
            net['dconv_2'] = slim.conv2d_transpose(net['dconv_1'], 64, [3, 3], stride=2, activation_fn=lrelu, 
                                            normalizer_fn=batch_norm, trainable=train, scope='deconv_2')  
            
            # for depth
            #net['depth_resnet'] = basic_resnet(net['deconv_2'], 64, [3, 3], first_activation=lrelu, last_activation=lrelu, 
                                            #preactive=False, train=train, downsample=False, preactive=False, name='depth_resnet')
            net['depth_dconv_3'] = slim.conv2d_transpose(net['dconv_2'], 32, [3, 3], stride=2, activation_fn=lrelu, 
                                            normalizer_fn=batch_norm, trainable=train, scope='depth_dconv_3')
            net['depth_dconv_4'] = slim.conv2d_transpose(net['depth_dconv_3'], 1, [3, 3], stride=2, activation_fn=lrelu, 
                                                        normalizer_fn=batch_norm, trainable=train, scope='depth_dconv_4')     
            
            # for norm
            #net['norm_resnet'] = basic_resnet(net['deconv_2'], 64, [3, 3], first_activation=lrelu, last_activation=lrelu, 
                                            #preactive=False, train=train, downsample=False, preactive=False, name='norm_resnet')
            net['norm_dconv_3'] = slim.conv2d_transpose(net['dconv_2'], 32, [3, 3], stride=2, activation_fn=lrelu, 
                                            normalizer_fn=batch_norm, trainable=train, scope='norm_dconv_3')
            net['norm_dconv_4'] = slim.conv2d_transpose(net['norm_dconv_3'], 3, [3, 3], stride=2, activation_fn=lrelu, 
                                                        normalizer_fn=batch_norm, trainable=train, scope='norm_dconv_4')            
            
        return net
    
    def U_net(self, encoder_net, generator_net, train=True, reuse=False):
        with tf.variable_scope('U_net', reuse=reuse) as scope:
            with tf.variable_scope('share'):
                generator_net['dconv_1'] = generator_net['dconv_1'] + slim.conv2d(encoder_net['resnet_4'], 128, [1, 1], 1, 
                                         activation_fn=None, trainable=train, scope='skip_connection_1')
                generator_net['dconv_2'] = generator_net['dconv_2'] + slim.conv2d(encoder_net['resnet_3'], 64, [1, 1], 1, 
                                         activation_fn=None, trainable=train, scope='skip_connection_2')
            
            with tf.variable_scope('depth'):
                generator_net['depth_dconv_3'] = generator_net['depth_dconv_3'] + slim.conv2d(encoder_net['resnet_2'], 32, [1, 1], 1, 
                                         activation_fn=None, trainable=train, scope='skip_connection_depth_3')
            
            with tf.variable_scope('norm'):
                generator_net['norm_dconv_3'] = generator_net['norm_dconv_3'] + slim.conv2d(encoder_net['resnet_2'], 32, [1, 1], 1, 
                                         activation_fn=None, trainable=train, scope='skip_connection_norm_3')        
    
    
    def discriminator(self, x, train, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope: 
            net = {}
            net['conv_1'] = slim.conv2d(x, 16, [3, 3], 2, activation_fn=lrelu, normalizer_fn=batch_norm, trainable=train, scope='conv_1')
            net['resnet_1'] = slim.conv2d(net['conv_1'], 32, [3, 3], 2, activation_fn=lrelu, normalizer_fn=batch_norm, trainable=train, scope='resnet_1')
            
        return net
    
    def bulid_model(self):        
        self.global_step = tf.Variable(0, trainable=False)
        self.train = tf.placeholder(tf.bool, shape=[1], name='train')
        
        self.encoder_net = self.encoder(self.img_batch, train=True)
        self.generator_net = self.generator(self.encoder_net['resnet_5'], train=True)
        self.U_net(self.encoder_net, self.generator_net, train=True)
        
        depth_diff = self.generator_net['depth_dconv_4'] - self.depth_batch
        self.depth_loss = tf.reduce_mean(tf.reduce_mean(tf.square(depth_diff), axis=[1, 2, 3]) - 
                                         0.5 * tf.square(tf.reduce_mean(depth_diff, axis=[1, 2, 3])), name='depth_loss')
        
        norm_normalization = tf.div(self.generator_net['norm_dconv_4'], 
                                    tf.reduce_sum(tf.square(self.generator_net['norm_dconv_4']), -1, keep_dims=True))
        
        norm_mul = tf.multiply(self.mask_batch, tf.multiply(norm_normalization, self.norm_batch))
        
        self.norm_loss = 1-tf.reduce_mean(tf.div(tf.reduce_sum(norm_mul, axis=[1, 2, 3]), tf.reduce_sum(self.mask_batch, axis=[1, 2, 3])), name='norm_loss')
        
        t_vars = tf.trainable_variables()
        self.e_vars = [var for var in t_vars if 'encoder' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.u_share_vars = [var for var in t_vars if 'U_net/share' in var.name]
        self.u_depth_vars = [var for var in t_vars if 'U_net/depth' in var.name]
        self.u_norm_vars = [var for var in t_vars if 'U_net/norm' in var.name]
            
        self.decay_steps = 100
        self.decay_rate = 0.9
        
    def train(self, config):
        self.bulid_model()
        lr = tf.train.exponential_decay(config.learning_rate, self.global_step, 
                                        self.decay_steps, self.decay_rate, staircase=True) 
        tf.summary.scalar('learning_rate', lr)
        
        depth_optim = tf.train.AdamOptimizer(lr) \
                          .minimize(self.depth_loss, global_step=self.global_step, var_list=self.e_vars+self.g_vars+self.u_share_vars+self.u_depth_vars)
        norm_optim = tf.train.AdamOptimizer(lr) \
                          .minimize(self.norm_loss, var_list=self.e_vars+self.g_vars+self.u_share_vars+self.u_norm_vars)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        if config.is_load_model:
            if self.load(config.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        
        counter = self.sess.run(self.global_step) + 1
        start_time = time.time()
        
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)   
        
        for step in xrange(config.epoch):
            _, _, depth_loss, norm_loss = self.sess.run([depth_optim, norm_optim, self.depth_loss, self.norm_loss])
            
            
            counter += 1
            print('setp: %2d time: %4.4fs depth_loss: %.6f, norm_loss: %.6f' % (counter, time.time() - start_time, depth_loss, norm_loss))


    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False    
        
        
