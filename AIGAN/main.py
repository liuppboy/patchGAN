import tensorflow as tf
import numpy as np
import os

from aigan import AIGAN

def main(_):
    

    with tf.Session(graph=tf.get_default_graph()) as sess:
        tf.set_random_seed(100)
        aigan_model = AIGAN(sess)
        aigan_model.train()


if __name__ == '__main__':
    tf.app.run()
