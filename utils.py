# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np


def get_shape(x):
    """ Returns data shape """
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    elif type(x) in [np.array, list, tuple]:
        return np.shape(x)
    else:
        raise Exception("Invalid layer.")