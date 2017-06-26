# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import sys
import time
import glob
from six.moves import xrange
from tensorflow.contrib import slim

sys.path.append('../')
from layers import batch_norm, upsampling_d2s
from activations import lrelu
from utils import load_model, save_grid, add_gaussian_noise
from read_celeba import generate_batch
from scipy import misc

class Inpainting(object):
    def __init__(self, sess, batch_size=32, iter_steps=50000, learning_rate=1e-3, 
                 sample_size=100, beta1=0.5, is_load_model=False, dataset_name='celeba', 
                 model_name='dcgan_model', checkpoint_dir='checkpoints', summary_dir='summary', 
                 sample_dir='sample'):
        self.sess = sess
        self.batch_size = batch_size
        self.iter_steps = iter_steps
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

    def generator(self, x, train=True, reuse=None):
        with tf.variable_scope('generator'):
            with tf.variable_scope('encoder', reuse=reuse) as scope:
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, kernel_size=3, stride=1, padding='SAME', 
                                    normalizer_fn=batch_norm, normalizer_params={'train': train}):
                    net = {}
                    net['conv1'] = slim.conv2d(x, 64, kernel_size=5, scope='conv1')
                    net['conv2'] = slim.conv2d(net['conv1'], 128, stride=2, scope='conv2')
                    net['conv3'] = slim.conv2d(net['conv2'], 128, scope='conv3')
                    net['conv4'] = slim.conv2d(net['conv3'], 256, stride=2, scope='conv4')
                    net['conv5'] = slim.conv2d(net['conv4'], 256, scope='conv5')
                    net['conv6'] = slim.conv2d(net['conv5'], 256, scope='conv6')
                    
                    net['conv7'] = slim.conv2d(net['conv6'], 256, rate=2, scope='conv7')
                    net['conv8'] = slim.conv2d(net['conv7'], 256, rate=4, scope='conv8')
                    net['conv9'] = slim.conv2d(net['conv8'], 256, rate=8, scope='conv9')
                    
                    net['conv10'] = slim.conv2d(net['conv9'], 256, scope='conv10')
                    net['conv11'] = slim.conv2d(net['conv10'], 256, scope='conv11')
                    
                        
                with slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.relu, kernel_size=4, stride=2, padding='SAME', 
                                    normalizer_fn=batch_norm, normalizer_params={'train': train}):
                    
                    net['deconv1'] = slim.conv2d_transpose(net['conv11'], 128, scope='deconv1')
                    net['conv12'] = slim.conv2d(net['deconv1'], 128, 3, normalizer_fn=batch_norm, 
                                                normalizer_params={'train': train}, scope='conv12')
                    net['deconv2'] = slim.conv2d_transpose(net['conv12'], 64, scope='deconv2')
                    net['conv13'] = slim.conv2d(net['deconv2'], 32, 3, normalizer_fn=batch_norm, 
                                                normalizer_params={'train': train}, scope='conv13')
                    net['conv14'] = slim.conv2d(net['conv13'], 3, 3, activation_fn=tf.nn.tanh, scope='conv14')
                #slim.add_arg_scope(upsampling_d2s)
                #with slim.arg_scope([upsampling_d2s], kernel_size=3, stride=2, padding='SAME', 
                                    #activation_fn=tf.nn.relu, normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train):
                    #net = {}
                    #net['fc'] = slim.fully_connected(x, 7*7*128, activation_fn=tf.nn.relu, 
                                    #normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train, scope='fc')
                    #net['fc'] = tf.reshape(net['fc'], [-1, 7, 7, 128])
                    ##net['deconv1'] = upsampling_d2s(net['fc'], 128, scope='deconv1')
                    ##net['deconv1'] = net['deconv1'][:, :7, :7, :]
                    #net['deconv2'] = upsampling_d2s(net['fc'], 64, scope='deconv2')
                    #net['deconv3'] = upsampling_d2s(net['deconv2'], 1, activation_fn=tf.nn.tanh, 
                                            #normalizer_fn=None, normalizer_params=None, scope='deconv3')                
            
        return net

    
    def discriminator(self, x, train, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as scope: 
            with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, stride=2, padding='SAME', 
                                normalizer_fn=batch_norm, normalizer_params={'train': train}, trainable=train):
                net = {}
                net['conv1'] = slim.conv2d(x, 64, scope='conv1')
                net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
                net['conv3'] = slim.conv2d(net['conv2'], 256, padding='VALID', scope='conv3')
                net['conv3'] = slim.flatten(net['conv3'])
                
            with slim.arg_scope([slim.fully_connected], activation_fn=lrelu, normalizer_fn=batch_norm, 
                                normalizer_params={'train': train}, trainable=train):
                #net['fc1'] = slim.fully_connected(net['conv3'], 256, scope='fc1')
                #net['fc2'] = slim.fully_connected(net['conv3'], 128, scope='fc2')
                net['fc'] = slim.fully_connected(net['conv3'], 1, normalizer_fn=None, activation_fn=None, scope='fc')
                    
        return net
    
    def bulid_model(self, train=True):  
        self.global_step = tf.Variable(0, trainable=False)
        
        self.g_net = self.generator(self.batch_img_mask, train=train)
        self.G = self.g_net['conv14']
        #self.G_ = tf.add(self.G, add_gaussian_noise(self.G.get_shape().as_list(), 
                                        #1, 0.01, 50000, self.global_step))
        #self.batch_data_ = tf.add(self.batch_data, add_gaussian_noise(self.batch_data.get_shape().as_list(), 
                                        #1, 0.01, 50000, self.global_step))
        
        #self.d_net_fake = self.discriminator(self.G_, train=True)
        #self.d_net_ture = self.discriminator(self.batch_data_, train=True, reuse=True)
        #self.D_true = self.d_net_ture['fc']
        #self.D_fake = self.d_net_fake['fc']
        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        
        all_vars = tf.global_variables()
        self.moving_mean_vars = [var for var in all_vars if 'moving_mean' in var.name]
        self.moving_variance_vars = [var for var in all_vars if 'moving_variance' in var.name]
        self.save_vars = self.d_vars + self.g_vars + self.moving_mean_vars + self.moving_variance_vars + [self.global_step]
        self.saver = tf.train.Saver(var_list=self.save_vars, max_to_keep=200)
        
        self.weighted_mse_loss = tf.divide(tf.reduce_sum(tf.multiply(self.batch_mask, tf.square(self.G - self.batch_img))), 
                                           tf.reduce_sum(self.batch_mask) * 3, name='weighted_mse_loss')
        #self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        #self.d_loss_true =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #logits=self.D_true, labels=tf.ones_like(self.D_true)))
        #self.d_loss_fake =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #logits=self.D_fake, labels=tf.zeros_like(self.D_fake))) 
        #self.d_loss = self.d_loss_true + self.d_loss_fake
        tf.summary.scalar('weighted_mse_loss', self.weighted_mse_loss)
        #tf.summary.scalar('d_loss_true', self.d_loss_true)
        #tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        #tf.summary.scalar('g_loss', self.g_loss)
        #tf.summary.scalar('d_loss', self.d_loss)        
        
        
    def train(self):
        tfrecord_name = ''.join(['../datasets/', self.dataset_name, '/train.tfrecords'])
        filename_queue  = tf.train.string_input_producer([tfrecord_name], capacity=32, shuffle=True)
        self.batch_img, self.batch_mask, self.batch_img_mask = generate_batch(filename_queue, min_queue_examples=2000, 
                                                                                  batch_size=self.batch_size, num_threads=16)                
        self.bulid_model(train=True)
        self.lr_decay = tf.train.polynomial_decay(self.learning_rate, self.global_step, decay_steps=4e4, end_learning_rate=1e-5)
        #self.d_optim = tf.train.AdamOptimizer(self.lr_decay * 0.1, self.beta1).minimize(self.d_loss, 
                            #global_step=self.global_step, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr_decay, self.beta1).minimize(self.weighted_mse_loss, 
                                                                                  var_list=self.g_vars)
    
        #self.g_grad = tf.gradients(self.g_loss, self.G)
        #tf.summary.histogram('g_grad_histogram', self.g_grad)
        #tf.summary.scalar('g_grad_abs', tf.reduce_mean(tf.abs(self.g_grad)))        
        merge_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./summary')
        
        self.sess.run(tf.global_variables_initializer())
        
        if self.is_load_model:
            if load_model(self.sess, self.saver, '/'.join([self.checkpoint_dir, self.dataset_name])):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        
        start_step = self.sess.run(self.global_step)
        start_time = time.time()
        

        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord) 
        for step in range(start_step+1, self.iter_steps+1):
            _, weighted_mse_loss = self.sess.run([self.g_optim, self.weighted_mse_loss])
            if np.mod(step, 10) == 0:
                print('step: %d time: %.6fs weighted_mse_loss: %.6f' % 
                      (step, time.time() - start_time, weighted_mse_loss))
            
            if np.mod(step, 100) == 0:
                summary_str = self.sess.run(merge_summary)
                summary_writer.add_summary(summary_str, global_step=step)
                
                #sample_fake_images = self.sess.run(self.sample_fake_images)
                #save_path = '%s/step_%d.png' %(self.sample_dir, step)
                #save_grid(sample_fake_images, [10, 10], save_path)
            
            if np.mod(step, 2000) == 0:
                self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name]), global_step=step) 
        coord.request_stop()
        coord.join(threads)
    
    def test(self):
        test_img_list = glob.glob('../datasets/celeba/test/img/*.jpg')
        test_mask_list = glob.glob('../datasets/celeba/test/mask/*.jpg')
        save_img_num = 32
        sample_img_dir = []
        for i in range(save_img_num):
            sample_img_dir.append('/'.join(['sample/test', test_img_list[i].split('/')[-1][:-4]]))
            if not os.path.exists(sample_img_dir[-1]):
                os.mkdir(sample_img_dir[-1])
        sample_img_dir = np.array(sample_img_dir)  
        self.batch_img = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.batch_mask = tf.placeholder(tf.float32, [None, 128, 128, 1])
        self.batch_img_mask = tf.placeholder(tf.float32, [None, 128, 128, 4])        
        self.bulid_model(train=False)
        batch_size = 32
        img_num = len(test_img_list)
        img = np.zeros([img_num, 128, 128, 3])
        mask = np.zeros([img_num, 128, 128, 1])
        for i in range(img_num):
            img[i] = misc.imread(test_img_list[i])
            mask[i, :, :, 0] = misc.imread(test_mask_list[i])
        img = img.astype('float32')
        mask = mask.astype('float32')
        
        sample_result = np.copy(img[:batch_size])
        img = img / 127.5 - 1
        img_mask = np.copy(img)
        img_mask[np.concatenate([mask, mask, mask], -1) > 0] = 0
        
        img_mask = np.concatenate([img, mask], -1)
        
        summary_writer = tf.summary.FileWriter(logdir='summary')
        steps = np.arange(2000, 50000+1, 2000, dtype='int32')
        self.sess.run(tf.global_variables_initializer())
        for step in steps:
            model_name = 'checkpoints/dcgan_model-%d' % step
            
            self.saver.restore(self.sess, model_name)
            
            samples = self.sess.run(self.G, feed_dict={self.batch_img_mask: img_mask[:batch_size]})
            samples = (samples + 1) * 127.5
            samples = samples.astype('uint8')
            sample_mask = np.concatenate([mask[:batch_size], mask[:batch_size], mask[:batch_size]], -1)
            sample_result[sample_mask > 0] = samples[sample_mask > 0]
            for i in range(batch_size):
                img_name = '/'.join([sample_img_dir[i], '%06d.jpg' % step])
                misc.imsave(img_name, sample_result[i])
            
            batch_num = int(img_num / batch_size)
            
            
            batch_losses = np.zeros(batch_num)
            for i in range(batch_num):
                batch_losses[i] = self.sess.run(self.weighted_mse_loss, 
                                                feed_dict={self.batch_img_mask: img_mask[i*batch_size:(i+1)*batch_size], 
                                                           self.batch_img: img[i*batch_size:(i+1)*batch_size], 
                                                           self.batch_mask: mask[i*batch_size:(i+1)*batch_size]})
            
            left_img_num = np.mod(img_num, batch_size)
            left_loss = self.sess.run(self.weighted_mse_loss, 
                                    feed_dict={self.batch_img_mask: img_mask[-left_img_num:], 
                                               self.batch_img: img[-left_img_num:], 
                                               self.batch_mask: mask[-left_img_num:]})
            
            weighted_mse_loss = (np.sum(batch_losses) * batch_size + left_loss * left_img_num) / img_num
            summary = tf.Summary()
            summary.value.add(tag='test_weighted_mse_loss', simple_value=weighted_mse_loss)
            summary_writer.add_summary(summary, global_step=step)            
            
            print('step: %d weighted_mse_loss: %.6f' % 
                                  (step, weighted_mse_loss))  

                
        
        
        
        
        
        
        
            
            
        
        
        
        
        