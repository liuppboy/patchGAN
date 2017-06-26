import numpy as np
import glob
from scipy import misc

file_list = glob.glob('/home/ppliu/python/DCGAN-tensorflow/data/img_align_celeba/*.jpg')
file_list = sorted(file_list)
file_list = np.array(file_list)
test_id = np.load('../datasets/celeba/test.npy')
test_list = file_list[test_id]
np.random.seed(0)
for filename in test_list:
    img = misc.imread(filename)
    rand_h_offset = np.random.randint(0, 58)
    rand_w_offset = np.random.randint(0, 18)
    img = img[rand_h_offset:rand_h_offset+160, rand_w_offset:rand_w_offset+160, :] 
    img = misc.imresize(img, [128, 128])
    target_name = '/'.join(['../datasets/celeba/test/img', filename.split('/')[-1]])
    misc.imsave(target_name, img)
    
    mask = np.zeros([128, 128])
    mask_size = np.random.randint(32, 65)
    mask_offset = np.random.randint(8, 120-mask_size, [2])
    mask[mask_offset[0]:mask_offset[0]+mask_size, mask_offset[1]:mask_offset[1]+mask_size] = 1
    mask = mask.astype('uint8')
    target_name = '/'.join(['../datasets/celeba/test/mask', filename.split('/')[-1]])
    misc.imsave(target_name, mask)
    
    
    



