import os
import numpy as np


import tensorflow as tf
from model import patchGAN

class Config():
    def __init__(self, epoch=2000, learning_rate=0.001, beta1=0.9, train_size=2000, validation_size=100, is_load_model=False,
                 checkpoint_dir='checkpoint_dir',):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.train_size = train_size
        self.validation_size = validation_size
        self.is_load_model = is_load_model
        self.checkpoint_dir = checkpoint_dir

def main(_):
    
    config = Config()
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    with tf.Session(graph=tf.get_default_graph()) as sess:
        tf.set_random_seed(100)
        gan = patchGAN(sess)
        gan.train(config)


if __name__ == '__main__':
    tf.app.run()
