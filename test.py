import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
from collections import namedtuple

LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss", "const_loss", "l1_loss",
                                       "category_loss", "cheat_loss", "tv_loss"])

a