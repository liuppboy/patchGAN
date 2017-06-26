import tensorflow as tf
import numpy as np
import os

from inpainting import Inpainting

def main(_):

    with tf.Session(graph=tf.get_default_graph()) as sess:
        tf.set_random_seed(100)
        model = Inpainting(sess)
        #model.train()
        #model.test()
        #model.train_feed()
        model.train_discriminator_wgan_gp()


if __name__ == '__main__':
    tf.app.run()
