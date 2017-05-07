import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

depth = loadmat('./dataset/depth.mat')['depth']
plt.imshow(depth)