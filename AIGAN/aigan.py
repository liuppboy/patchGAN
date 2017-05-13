# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import sys
import time

from six.moves import xrange
from tensorflow.contrib import slim


sys.path.append('../')
from layers import batch_norm
from activations import lrelu
from utils import load_model, save_grid

class AIGAN(object):
    def __init__(self, sess, batch_size=64, iter_steps=100000, z_dim=100, learning_rate=1e-3, 
                 sample_size=100, beta1=0.5, is_load_model=False, dataset_name='mnist', 
                 model_name='aigan_model', checkpoint_dir='checkpoints', summary_dir='summary', 
                 sample_dir='sample', read_mode='from_file'):
        '''
        read_mode: the method of reading data, 'feed' or 'from_file', latter one is more efficient.
        '''
        self.sess = sess
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.sample_size = sample_size
        self.beta1 = beta1        
        
        self.is_load_model = is_load_model
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir) 
            
        self.sample_dir = sample_dir
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)        
        
        self.summary_dir = summary_dir
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir) 
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.read_mode = read_mode
        if self.dataset_name == 'mnist':
            self.data_h = 28
            self.data_w = 28
            self.data_c = 1
            
            if self.read_mode == 'feed':
                from datasets import read_mnist_feed
                self.dataset = read_mnist_feed('../datasets/mnist')
                self.batch_data = tf.placeholder(tf.float32, shape=[None, self.data_h, self.data_w, self.data_c], name='batch_data')
            elif self.read_mode == 'from_file':
                from datasets import generate_batch
                filename_queue = tf.train.string_input_producer(['../datasets/mnist/train.tfrecords'])
                self.batch_data = generate_batch(filename_queue, 
                                        min_queue_examples=5000, batch_size=self.batch_size)
            else:
                raise ValueError('No such mode!')
                
        
        np.random.seed(100)
        self.sample_z = tf.constant(np.random.normal(size=[400, self.z_dim]).astype(np.float32)) 
        self.z_input = tf.random_normal(shape=[self.batch_size, self.z_dim], name='z_input')
        
    
    def generator(self, x, train=True, reuse=None):
        with tf.variable_scope('generator', reuse=reuse) as scope:
            with slim.arg_scope([slim.conv2d_transpose], activation_fn=lrelu, kernel_size=4, stride=2, padding='SAME', 
                                normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train):
                net = {}
                net['fc'] = slim.fully_connected(x, 4*4*256, activation_fn=lrelu, 
                                normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train, scope='fc')
                net['fc'] = tf.reshape(net['fc'], [-1, 4, 4, 256])
                net['deconv1'] = slim.conv2d_transpose(net['fc'], 128, scope='deconv1')
                net['deconv1'] = net['deconv1'][:, :7, :7, :]
                net['deconv2'] = slim.conv2d_transpose(net['deconv1'], 64, scope='deconv2')
                net['deconv3'] = slim.conv2d_transpose(net['deconv2'], 1, activation_fn=tf.nn.tanh, 
                                                       normalizer_fn=None, normalizer_params=None, scope='deconv3')            
            
        return net

    
    def discriminator(self, x, train, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as scope: 
            with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, stride=2, padding='SAME', 
                                normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train):
                net = {}
                net['conv1'] = slim.conv2d(x, 64, scope='conv1')
                net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
                net['conv3'] = slim.conv2d(net['conv2'], 256, scope='conv3')
                
                # true or false
                net['conv3'] = slim.flatten(net['conv3'])
                net['fc'] = slim.fully_connected(net['conv3'], 1, activation_fn=None, trainable=train, scope='fc')
                    
        return net
    
    def encoder(self, x, train, reuse=None):
        with tf.variable_scope('encoder', reuse=reuse) as scope: 
            with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, stride=2, padding='SAME', 
                                normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train):
                net = {}
                net['conv1'] = slim.conv2d(x, 64, scope='conv1')
                net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
                net['conv3'] = slim.conv2d(net['conv2'], 256, scope='conv3')
                
                # regression z
                net['conv3'] = slim.flatten(net['conv3'])
                net['z_regression'] = slim.fully_connected(net['conv3'], self.z_dim, activation_fn=None, trainable=train, scope='z_regression')       
        return net        
    
    def bulid_model(self):        
        self.global_step = tf.Variable(0, trainable=False)
        
        self.g_net = self.generator(self.z_input, train=True)
        self.G = self.g_net['deconv3']
        
        self.d_net_fake = self.discriminator(self.G, train=True)
        self.d_net_ture = self.discriminator(self.batch_data, train=True, reuse=True)
        self.e_net_true = self.encoder(self.batch_data, train=True)
        self.e_net_fake = self.encoder(self.G, train=True, reuse=True)
        self.img_reconstruct = self.generator(self.e_net_true['z_regression'], train=True, reuse=True)['deconv3']
        
        self.D_true = self.d_net_ture['fc']
        self.D_fake = self.d_net_fake['fc']
        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        
        all_vars = tf.global_variables()
        self.moving_mean_vars = [var for var in all_vars if 'moving_mean' in var.name]
        self.moving_variance_vars = [var for var in all_vars if 'moving_variance' in var.name]
        self.save_vars = self.d_vars + self.g_vars + self.e_vars + self.moving_mean_vars + self.moving_variance_vars + [self.global_step]
        self.saver = tf.train.Saver(var_list=self.save_vars, max_to_keep=200)
        
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.d_loss_true =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_true, labels=tf.ones_like(self.D_true)))
        self.d_loss_fake =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake))) 
        self.d_loss = self.d_loss_true + self.d_loss_fake
        self.z_regression_loss = tf.reduce_mean(tf.abs(self.z_input - self.e_net_fake['z_regression']), name='z_regression_loss')
        self.autoencoder_loss = tf.reduce_mean(tf.square(self.batch_data - self.img_reconstruct), name='autoencoder_loss')
        
        tf.summary.scalar('d_loss_true', self.d_loss_true)
        tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('z_regression_loss', self.z_regression_loss)
        tf.summary.scalar('autoencoder_loss', self.autoencoder_loss)
        
        lr = tf.train.polynomial_decay(self.learning_rate, self.global_step, decay_steps=1e5, end_learning_rate=1e-5)
        self.d_optim = tf.train.AdamOptimizer(lr, self.beta1).minimize(self.d_loss, 
                                            global_step=self.global_step, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(lr, self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        self.e_optim = tf.train.AdamOptimizer(lr, self.beta1).minimize(10 * self.z_regression_loss + self.autoencoder_loss, 
                                                                       var_list=self.g_vars + self.e_vars)
        
    def sampler(self, sample_z):
        sample_fake_images = self.generator(sample_z, train=False, reuse=True)['deconv3']
        sample_fake_images = (sample_fake_images + 1) * 127.5
        sample_fake_images = tf.cast(sample_fake_images, tf.uint8)
        return sample_fake_images
        
        
        
    def train(self):
        self.bulid_model()
        merge_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./summary')
        self.sample_fake_images = self.sampler(self.sample_z)
        
        self.sess.run(tf.global_variables_initializer())
        
        if self.is_load_model:
            if load_model(self.sess, self.saver, '/'.join([self.checkpoint_dir, self.dataset_name])):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        
        start_step = self.sess.run(self.global_step)
        start_time = time.time()
        
        if self.read_mode == 'feed':
            for step in range(start_step, self.iter_steps):
                batch_data, _ = self.dataset.next_batch(self.batch_size)
                batch_data = batch_data * 2 - 1.
                batch_data = np.reshape(batch_data, [self.batch_size, 28, 28, 1])
                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
                _, _, _, d_loss, g_loss, d_loss_true, d_loss_fake = self.sess.run(
                    [self.d_optim, self.g_optim, self.e_optim, self.d_loss, self.g_loss, self.d_loss_true, self.d_loss_fake], 
                feed_dict={self.batch_data: batch_data})
        
                if np.mod(step, 20) == 0:
                    print('step: %d time: %.6fs d_loss: %.6f, g_loss: %.6f, d_loss_true: %.6f, d_loss_fake: %.6f' % 
                          (step, time.time() - start_time, d_loss, g_loss, d_loss_true, d_loss_fake))
                
                if np.mod(step, 100)==0:
                    summary_str = self.sess.run(merge_summary, feed_dict={self.batch_data: batch_data})
                    summary_writer.add_summary(summary_str, global_step=step)
                    
                    sample_fake_images = self.sess.run(self.sample_fake_images)
                    save_path = '%s/step_%d.png' %(self.sample_dir, step)
                    save_grid(sample_fake_images, [20, 20], save_path)
                
                if np.mod(step, 1000) == 0:
                    self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name]), global_step=step)
        elif self.read_mode == 'from_file':
            coord=tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord) 
            for step in range(start_step, self.iter_steps):
                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
                _, _, _, d_loss, g_loss, d_loss_true, d_loss_fake, z_regression_loss, autoencoder_loss = \
                    self.sess.run([self.d_optim, self.g_optim, self.e_optim, self.d_loss, self.g_loss, self.d_loss_true, 
                     self.d_loss_fake, self.z_regression_loss, self.autoencoder_loss])
                self.sess.run([self.g_optim, self.e_optim])
                self.sess.run([self.g_optim, self.e_optim])
                if np.mod(step, 20) == 0:
                    print('step: %d time: %.6fs d_loss: %.6f, g_loss: %.6f, d_loss_true: %.6f, d_loss_fake: %.6f, z_regression_loss: %.6f, autoencoder_loss: %.6f' % 
                          (step, time.time() - start_time, d_loss, g_loss, d_loss_true, 
                           d_loss_fake, z_regression_loss, autoencoder_loss))
                
                if np.mod(step, 100)==0:
                    #summary_str = self.sess.run(merge_summary)
                    #summary_writer.add_summary(summary_str, global_step=step)
                    
                    sample_fake_images = self.sess.run(self.sample_fake_images)
                    save_path = '%s/step_%d.png' %(self.sample_dir, step)
                    save_grid(sample_fake_images, [20, 20], save_path)
                
                #if np.mod(step, 1000) == 0:
                    #self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name]), global_step=step)            