# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


import tensorflow as tf
import numpy as np
import os
import sys
import time

from six.moves import xrange
from tensorflow.contrib import slim
from tensorflow.contrib.learn.python.learn.datasets import mnist

from resnet import basic_resnet
from layers import batch_norm
from activation import lrelu


LAMBDA = 10 # Gradient penalty lambda hyperparameter

class patchGAN(object):
    def __init__(self, sess, batch_size=4, dataset_name='mnist', mode='dcgan',
                 checkpoint_dir='checkpoints', summary_dir='summary', 
                 sample_size=100, z_dim=100, learning_rate=1e-3, beta1=0.5):
        self.sess = sess
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.sample_size = sample_size
        self.summary_dir = summary_dir
        if self.dataset_name == 'mnist':
            self.data_h = 28
            self.data_w = 28
            self.data_c = 1
        
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.beta1 = beta1
        tfrecord_name = [r'E:\nyudepth\tfrecord\test_list\classroom_0001.tfrecord', r'E:\nyudepth\tfrecord\test_list\dining_room_0026.tfrecord']
        filename_queue  = tf.train.string_input_producer(tfrecord_name, shuffle=True)
        self.img_batch, self.depth_batch, self.norm_batch, self.mask_batch = generate_batch(filename_queue, min_queue_examples=1000, batch_size=batch_size)         
        
    
    def generator(self, x, train=True, reuse=None):
        with tf.variable_scope('generator', reuse=reuse) as scope:
            with slim.arg_scope([slim.conv2d_transpose], activation_fn=lrelu, kernel_size=4, stride=2, padding='SAME', 
                                normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train):
                net = {}
                net['fc'] = slim.fully_connected(x, 4*4*256, activation_fn=lrelu, 
                                normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train, scope='fc')
                net['fc'] = tf.reshape(net['fc'], [-1, 4, 4, 256])
                net['deconv1'] = slim.conv2d_transpose(net['fc1'], 128, scope='deconv1')
                net['deconv1'] = net['deconv1'][:, :7, :7, :]
                net['deconv2'] = slim.conv2d_transpose(net['deconv1'], 64, scope='deconv2')
                net['deconv3'] = slim.conv2d_transpose(net['deconv2'], 1, activation_fn=tf.nn.tanh, 
                                                       normalizer_fn=None, normalizer_params=None, scope='deconv3')            
            
        return net

    
    def discriminator(self, x, train, reuse=None, mode='dcgan'):
        if mode == 'wgan-gp': # without BN
            with tf.variable_scope('discriminator', reuse=reuse) as scope: 
                with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, stride=2, padding='SAME', trainable=train):
                    net = {}
                    net['conv1'] = slim.conv2d(x, 64, scope='conv1')
                    net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
                    net['conv3'] = slim.conv2d(net['conv2'], 235, scope='conv3')
                    net['conv3'] = slim.flatten(net['conv3'])
                    net['fc'] = slim.fully_connected(net['conv3'], 1, activation_fn=None, trainable=train, scope='fc')                    
                    
        else:
            with tf.variable_scope('discriminator', reuse=reuse) as scope: 
                with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, stride=2, padding='SAME', 
                                    normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train):
                    net = {}
                    net['conv1'] = slim.conv2d(x, 64, scope='conv1')
                    net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
                    net['conv3'] = slim.conv2d(net['conv2'], 235, scope='conv3')
                    net['conv3'] = slim.flatten(net['conv3'])
                    net['fc'] = slim.fully_connected(net['conv3'], 1, activation_fn=None, trainable=train, scope='fc')
                    

        return net
    
    def bulid_model(self, mode='dcgan'):        
        self.global_step = tf.Variable(0, trainable=False)
        
        self.z_input = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z_input')
        self.data_true = tf.placeholder(tf.float32, shape=[None, self.data_h, self.data_w, self.data_c], name='data_true')
        
        self.g_net = self.generator(self.z_input, train=True)
        self.G = self.g_net['deconv3']
        
        self.d_net_ture = self.discriminator(self.data_true, train=True, reuse=True, mode=mode)
        self.d_net_fake = self.discriminator(self.G, train=True, mode=mode)
        self.D_true = self.d_net_ture['fc']
        self.D_fake = self.d_net_fake['fc']
        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        
        if mode == 'dcgan':
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
            self.d_loss_true =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_true, labels=tf.ones_like(self.D_true)))
            self.d_loss_fake =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake))) 
            self.d_loss = self.d_loss_true + self.d_loss_fake
            
            tf.summary.scalar('d_loss_true', self.d_loss_true)
            tf.summary.scalar('d_loss_fake', self.d_loss_fake)
            
            
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.d_loss, 
                                                global_step=self.global_step, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.g_loss, var_list=self.g_vars)
            
        
        elif mode == 'wgan':
            self.g_loss = -tf.reduce_mean(self.D_fake)
            self.d_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_true) 
            self.d_optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.d_loss, var_list=self.d_vars) 
            self.g_optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)
            
            clip_ops = []
            for var in self.d_vars:
                clip_bounds = [-.01, .01]
                clip_ops.append(
                    tf.assign(
                        var, 
                        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            clip_disc_weights = tf.group(*clip_ops)            
        elif mode == 'wgan-gp':
            self.g_loss = -tf.reduce_mean(self.D_fake)
            self.d_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_true)  
            
            # Gradient penalty
            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
            differences = self.G - self.data_true
            interpolates = self.data_true + alpha * differences
            gradients = tf.gradients(self.discriminator(interpolates, train=True, reuse=True, mode=mode)['fc'], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=-1))
            self.gradient_penalty = tf.reduce_mean(tf.square((slopes-1.)))
            
            tf.summary.scalar('gradient_penalty', self.gradient_penalty)
            
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.d_loss + LAMBDA * self.gradient_penalty, 
                                                global_step=self.global_step, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        
            
    def train(self):
        self.bulid_model(mode=self.mode)
        
        self.sess.run(tf.global_variables_initializer())
        
        if config.is_load_model:
            if self.load('/'.join([self.checkpoint_dir, self.dataset_name])):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        
        counter = self.sess.run(self.global_step)
        start_time = time.time()
        
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)   
        
        for step in xrange(config.epoch):
            _, _, depth_loss, norm_loss = self.sess.run([depth_optim, norm_optim, self.depth_loss, self.norm_loss])
            
            
            counter += 1
            print('setp: %2d time: %4.4fs depth_loss: %.6f, norm_loss: %.6f' % (counter, time.time() - start_time, depth_loss, norm_loss))


    def save(self, checkpoint_dir, step, model_name='model'):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    
    def load(self, checkpoint_dir='checkpoints', step=None, verbose=True):
        if verbose:
            print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(self.train_dir, checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if step:
                basename = ckpt_name.split('-')[0]
                ckpt_name = '-'.join([basename, str(step)])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            if verbose:
                print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            if verbose:
                print(" [*] Failed to load checkpoint")
            return False
        
        
