# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def batch_norm(x, train=True, reuse=None, decay=0.999, scale=True, epsilon=1e-3, scope='batch_norm'):
    return slim.batch_norm(x, decay=decay, scale=scale, epsilon=epsilon, 
                is_training=train, reuse=reuse, scope=scope)




        