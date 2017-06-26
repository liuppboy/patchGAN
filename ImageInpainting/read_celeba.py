import tensorflow as tf
import numpy as np
from PIL import Image
import glob
crop_size = 160
resize_size = tf.constant(128, tf.int32, shape=[2])
imgsize_zero = tf.zeros([128, 128, 3])
np.random.seed(0)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'img': tf.FixedLenFeature([], tf.string),
        })

    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(tf.cast(img, tf.float32), [218, 178, 3])
    rand_h_offset = np.random.randint(0, 58)
    rand_w_offset = np.random.randint(0, 18)
    img = tf.image.crop_to_bounding_box(img, rand_h_offset, rand_w_offset, 
                                       crop_size, crop_size)
    img = tf.image.resize_images(img, resize_size, align_corners=True)
    img = img / 127.5 - 1.
    
    mask = np.zeros([128, 128, 1])
    mask_size = np.random.randint(32, 65)
    mask_offset = np.random.randint(8, 120-mask_size, [2])
    mask[mask_offset[0]:mask_offset[0]+mask_size, mask_offset[1]:mask_offset[1]+mask_size] = 1
    
    local_patch_coord = np.maximum(np.zeros(2), mask_offset+mask_size/2-32)
    local_patch_coord[local_patch_coord > 64] = 64
    local_patch_coord = tf.convert_to_tensor(local_patch_coord, dtype=tf.int32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    img_destroy = tf.multiply(img, tf.subtract(1., mask))
    
    img_mask = tf.concat([img_destroy, mask], -1)
    return img, mask, img_mask, local_patch_coord

def generate_batch(filename_queue, min_queue_examples, batch_size,  num_threads=16, shuffle=True):
    img, mask, img_mask, local_patch_coord = read_and_decode(filename_queue)
    if shuffle:
        batch_img, batch_mask, batch_img_mask, batch_local_patch_coord = tf.train.shuffle_batch([img, mask, img_mask, local_patch_coord], batch_size, num_threads=num_threads,
                                           capacity=min_queue_examples + 3 * batch_size,
                                           min_after_dequeue=min_queue_examples)
    else:
        batch_img, batch_mask, batch_img_mask, batch_local_patch_coord = tf.train.batch([img, mask, img_mask, local_patch_coord], batch_size, num_threads=num_threads,
                                           capacity=min_queue_examples + 3 * batch_size,
                                           min_after_dequeue=min_queue_examples)
    
    return batch_img, batch_mask, batch_img_mask, batch_local_patch_coord

#filename_queue  = tf.train.string_input_producer(['../datasets/celeba/test.tfrecords'], shuffle=False)
#img = read_and_decode(filename_queue)
#with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #coord=tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #for i in range(10):
        #img_raw = sess.run(img)
        
        #img_raw = Image.fromarray(img_raw)
        #img_raw.show()
    
    #coord.request_stop()
    #coord.join(threads)