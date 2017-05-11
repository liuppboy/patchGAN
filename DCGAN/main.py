import tensorflow as tf
import numpy as np
import os

from dcgan import DCGAN

def main(_):
    

    with tf.Session(graph=tf.get_default_graph()) as sess:
        tf.set_random_seed(100)
        dcgan_model = DCGAN(sess)
        dcgan_model.train()


if __name__ == '__main__':
    tf.app.run()
