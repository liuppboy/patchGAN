import tensorflow as tf
import numpy as np
from PIL import Image
import glob


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'img': tf.FixedLenFeature([], tf.string),
            'depth': tf.FixedLenFeature([], tf.string), 
            'norm': tf.FixedLenFeature([], tf.string), 
            'mask': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['img'], tf.uint8)
    depth = tf.decode_raw(features['depth'], tf.float32)
    norm = tf.decode_raw(features['norm'], tf.float32)
    mask = tf.decode_raw(features['mask'], tf.uint8)
    
    return image, depth, norm, mask
                
                

filepath = glob.glob(r'./test_data/*.png')
features = np.zeros([len(filepath), 640, 640], dtype=np.uint8)
for i in range(len(filepath)):
    features[i] = np.array(Image.open(filepath[i]))
    
tfrecorder = BasicTFRecord()
tfrecorder.to_record([features], ['byte'], ['img'], './test_data/test.tfrecords')


filename_queue  = tf.train.string_input_producer(['./test_data/test.tfrecords'], shuffle=False)
img, k = read_and_decode(filename_queue)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for i in range(10):
        img_raw = sess.run(img)
        key = sess.run(k)
        img_raw = Image.fromarray(img_raw)
        img_raw.show()
    
    coord.request_stop()
    coord.join(threads)