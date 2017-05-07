import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob
import h5py

from scipy.io import loadmat
from scipy.misc import imresize

from tf_reader import BasicTFRecord

shuffle_train_list = np.load('./dataset/shuffle_train_list.npy')

tfrecorder = BasicTFRecord()

#train_list = np.squeeze(loadmat('./dataset/splits.mat')['train_list'])
#nyu_files = glob.glob('/media/ppliu/JackLI/nyudepth/crop_data/train_list/*.mat')
#nyu_files = glob.glob(r'E:\nyudepth\crop_data\test_list\*.mat')
#nyu_files = glob.glob(r'E:\nyudepth\crop_data\other_list\*.mat')
#writer = tf.python_io.TFRecordWriter('/media/ppliu/JackLI/nyudepth/tfrecord/shuffle_train/train_4.tfrecord')
tfrecorder_name = None
count = 0
trainNdxs = np.squeeze(loadmat('./dataset/splits.mat')['testNdxs'])
data = h5py.File('./dataset/nyu_depth_v2_labeled.mat')
depths = data['depths'][()]
imgs = data['images'][()]
imgs = np.transpose(imgs, [0, 3, 2, 1])
depths = np.transpose(depths, [0, 2, 1])
imgs = imgs[trainNdxs-1]
depths = depths[trainNdxs-1]
masks = np.zeros_like(depths, dtype=np.uint8)
masks[:, 8:-8, 8:-8] = 1
tfrecorder.to_record(writer, [imgs, depths, masks], ['byte', 'byte', 'byte'], ['img', 'depth', 'mask'], tfrecorder_name)
for train_scene in shuffle_train_list[24000:]:
    #train_scene = np.squeeze(train_scene)
    
    #for i in train_scene:
        data = loadmat(train_scene)
        
        mask = np.zeros([480, 640])
        mask[44:471, 40:601] = 1
        
        img = data['img'][:480, :640]
        depth = data['depth'][:480, :640]
    #img = np.zeros([len(data), H, W, 3])
    #depth = np.zeros([len(data), H, W])
    #mask = np.zeros([len(data), H, W])
    #norm = np.zeros([len(data), H, W, 3])
    #for i, mat in enumerate(data):
        #img[i] = mat['img']
        #depth[i] = mat['depth']
        #mask[i] = mat['mask']
        #norm[i] = mat['norm']
    
        img = img.astype(np.uint8)
        depth = depth.astype(np.float32)
        mask = mask.astype(np.uint8)
        
        img = np.expand_dims(img, 0)
        depth = np.expand_dims(depth, 0)
        mask = np.expand_dims(mask, 0)
        
        tfrecorder.to_record(writer, [img, depth, mask], ['byte', 'byte', 'byte'], ['img', 'depth', 'mask'], tfrecorder_name)
        count = count+1
        print(count)


writer.close()
    
        
    