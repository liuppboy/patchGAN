# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)