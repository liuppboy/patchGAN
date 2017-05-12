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
from datasets import read_mnist_feed
mnist_dataset = read_mnist_feed()


class DCGAN(object):
    def __init__(self, sess, batch_size=32, iter_steps=100000, z_dim=100, learning_rate=1e-3, 
                 sample_size=100, beta1=0.5, is_load_model=False, dataset_name='mnist', 
                 model_name='dcgan_model', checkpoint_dir='checkpoints', summary_dir='summary', 
                 sample_dir='sample'):
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
        if self.dataset_name == 'mnist':
            self.data_h = 28
            self.data_w = 28
            self.data_c = 1
        
        np.random.seed(100)
        self.sample_z = tf.constant(np.random.normal(size=[100, self.z_dim]).astype(np.float32)) 
        
        #tfrecord_name = [r'E:\nyudepth\tfrecord\test_list\classroom_0001.tfrecord', r'E:\nyudepth\tfrecord\test_list\dining_room_0026.tfrecord']
        #filename_queue  = tf.train.string_input_producer(tfrecord_name, shuffle=True)
        #self.img_batch, self.depth_batch, self.norm_batch, self.mask_batch = generate_batch(filename_queue, min_queue_examples=1000, batch_size=batch_size)         
        
    
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
                net['conv3'] = slim.conv2d(net['conv2'], 235, scope='conv3')
                net['conv3'] = slim.flatten(net['conv3'])
                net['fc'] = slim.fully_connected(net['conv3'], 1, activation_fn=None, trainable=train, scope='fc')
                    
        return net
    
    def bulid_model(self):        
        self.global_step = tf.Variable(0, trainable=False)
        
        self.z_input = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z_input')
        self.data_true = tf.placeholder(tf.float32, shape=[None, self.data_h, self.data_w, self.data_c], name='data_true')
        
        self.g_net = self.generator(self.z_input, train=True)
        self.G = self.g_net['deconv3']
        
        self.d_net_ture = self.discriminator(self.data_true, train=True, reuse=True)
        self.d_net_fake = self.discriminator(self.G, train=True)
        self.D_true = self.d_net_ture['fc']
        self.D_fake = self.d_net_fake['fc']
        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        
        all_vars = tf.global_variables()
        self.moving_mean_vars = [var for var in all_vars if 'moving_mean' in var.name]
        self.moving_variance_vars = [var for var in all_vars if 'moving_variance' in var.name]
        self.save_vars = self.d_vars + self.g_vars + self.moving_mean_vars + self.moving_variance_vars + [self.global_step]
        self.saver = tf.train.Saver(var_list=self.save_vars, max_to_keep=200)
        
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.d_loss_true =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_true, labels=tf.ones_like(self.D_true)))
        self.d_loss_fake =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake))) 
        self.d_loss = self.d_loss_true + self.d_loss_fake
        
        tf.summary.scalar('d_loss_true', self.d_loss_true)
        tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)        
        
        
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.d_loss, 
                                            global_step=self.global_step, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        
        
    def train(self):
        self.bulid_model()
        merge_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./summary')
        self.sess.run(tf.global_variables_initializer())
        self.sample_fake_images = self.generator(self.sample_z, train=False, reuse=True)['deconv3']
        if self.is_load_model:
            if load_model(self.sess, self.saver, '/'.join([self.checkpoint_dir, self.dataset_name])):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        
        start_step = self.sess.run(self.global_step)
        start_time = time.time()
        
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)   
        
        
        for step in range(start_step, self.iter_steps):
            batch_data, _ = mnist_dataset.train.next_batch(self.batch_size)
            batch_data = batch_data * 2 - 1.
            batch_data = np.reshape(batch_data, [self.batch_size, 28, 28, 1])
            batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
            _, _, d_loss, g_loss, d_loss_true, d_loss_fake = self.sess.run(
                [self.d_optim, self.g_optim, self.d_loss, self.g_loss, self.d_loss_true, self.d_loss_fake], 
            feed_dict={self.z_input: batch_z, self.data_true: batch_data})
            if np.mod(step, 10) == 0:
                print('step: %d time: %.6fs d_loss: %.6f, g_loss: %.6f, d_loss_true: %.6f, d_loss_fake: %.6f' % 
                      (step, time.time() - start_time, d_loss, g_loss, d_loss_true, d_loss_fake))
            
            if np.mod(step, 100)==0:
                summary_str = self.sess.run(merge_summary, feed_dict={self.z_input: batch_z, self.data_true: batch_data})
                summary_writer.add_summary(summary_str, global_step=step)
                
                sample_fake_images = self.sess.run(self.sample_fake_images)
                sample_fake_images = (sample_fake_images + 1) * 127.5
                save_path = '%s/step_%d.png' %(self.sample_dir, step)
                save_grid(sample_fake_images.astype('uint8'), [10, 10], save_path)
            
            
            if np.mod(step, 1000) == 0:
                self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name]), global_step=step)
                
        

        
            



        
        
