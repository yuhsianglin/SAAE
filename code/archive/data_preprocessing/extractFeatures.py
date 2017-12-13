import numpy as np
import os
import tensorflow as tf
from PIL import Image

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet, dataset_utils
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

img_dir = '../data/CUB_200_2011/images/'
preprocessed_dir = '../data/preprocessed'
checkpoints_dir = 'checkpoints/'
image_size = inception.inception_v1.default_image_size

# url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"

# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)

# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

for foldername in os.listdir(img_dir):
    print(foldername)
    
    if not tf.gfile.Exists(os.path.join(preprocessed_dir, foldername)):
        tf.gfile.MakeDirs(os.path.join(preprocessed_dir, foldername))

    for filename in os.listdir(os.path.join(img_dir, foldername)):
        with tf.Graph().as_default():
            image_file = tf.read_file(os.path.join(img_dir, foldername, filename))
            image = tf.image.decode_jpeg(image_file, channels=3)
            processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_images = tf.expand_dims(processed_image, 0)
            
            with slim.arg_scope(inception.inception_v1_arg_scope()):
                logits, end_points = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
            probabilities = tf.nn.softmax(logits)                

            init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoints_dir, 'inception_v1.ckpt'), 
                slim.get_model_variables('InceptionV1'))

            with tf.Session() as sess:
                init_fn(sess)
                
                np_image, probabilities = sess.run([image, probabilities])
                array = end_points['AvgPool_0a_7x7'].eval()
                np.save(os.path.join(preprocessed_dir, foldername, filename), array)